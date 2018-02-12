# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

import numpy as np
import pandas as pd

from cgpm.crosscat.engine import Engine
from cgpm.utils.parallel_map import parallel_map


class TRCRP_Mixture(object):
    """Temporally-Reweighted Chinese Restaurant Process Mixture.

    The data frame being modeled has an integer-valued index indicating the
    discrete time step, and has one column per time-varying variable, as shown
    below:

        +------+----------+----------+----------+
        | Time |   Var A  |   Var B  |   Var C  |
        +------+----------+----------+----------+
        | 1997 |   0.62   |   0.38   |   1.34   |
        | 1998 |   0.82   |   0.23   |    nan   |
        | 1999 |    nan   |   0.13   |   2.19   |
        | 2000 |   1.62   |   0.22   |   1.70   |
        | 2001 |   0.78   |   2.89   |    nan   |
        +------+----------+----------+----------+
    """

    def __init__(self, chains, lag, variables, rng, dependencies=None):
        """Initialize a TRCRP Mixture instance.
            chains::int
                Number of MCMC chains.
            lag::int
                Order of the AR process.
            variables::list<str>
                Name of time series variables.
            rng::np.random.RandomState
                Source of entropy.
            dependencies::list<tuple<str>>
                Blocks of variables which are constrained to be dependent.
        """
        # From constructor.
        self.chains = chains
        self.lag = lag
        self.variables = list(variables)
        self.rng = rng
        self.dependencies = self._make_dependencies(dependencies)
        # Derived fields.
        self.window = self.lag + 1
        self.variables_lagged = list(itertools.chain.from_iterable([
            ['%s.lag.%d' % (varname, i,) for i in xrange(self.lag, -1, -1)]
            for varname in self.variables
        ]))
        for variable in self.variables:
            variable_idx = self._variable_to_index(variable)
            assert self.variables_lagged[variable_idx]=='%s.lag.0' % (variable,)
        # Internal fields.
        self.dataset = pd.DataFrame()
        self.engine = None
        self.initialized = None

    def incorporate(self, frame):
        """Incorporate observations."""
        assert set(frame.columns) == set(self.variables)
        self._incorporate_new_sampids(frame)
        # XXX Improve this function.
        self._incorporate_existing_sampids(frame)
        assert self.engine.states[0].n_rows() == len(self.dataset)

    def transition(self, **kwargs):
        backend = kwargs.pop('backend', None)
        kwargs['cols'] = self._variable_indexes()
        if backend in ['cgpm', None]:
            self.engine.transition(**kwargs)
        elif backend in ['lovecat']:
            self.engine.transition_lovecat(**kwargs)
        elif backend in ['loom']:
            self.engine.transition_loom(**kwargs)
        else:
            raise ValueError('Unknown backend: %s' % (backend,))

    def simulate(self, sampids, variables, nsamples, multiprocess=1):
        """Simulate sampids and variables in a non-ancestral manner.

        // panelcat has 2 chains, so chains * nsamples = 4 samples returned.
        >> panelcat.simulate([1, 4], ['a', 'b'], 2)

        #  |<--- chain 0 --->| |<--- chain 1 ---->|
        [sample0.0, sample0.1, sample1.0, sample1.1]

        sample0.0: ((val_a1, val_b1), (val_a4, val_b4))
        sample0.1: ((val_a1, val_b1), (val_a4, val_b4))
        sample1.0: ((val_a1, val_b1), (val_a4, val_b4))
        sample1.1: ((val_a1, val_b1), (val_a4, val_b4))
        """
        cgpm_rowids = [self._sampid_to_rowid(sampid) for sampid in sampids]
        constraints_list = [self._get_cgpm_constraints(sampid) for sampid in sampids]
        query = [self._variable_to_index(var) for var in variables]
        targets_list = [query] * len(cgpm_rowids)
        Ns = [nsamples] * len(cgpm_rowids)
        samples_raw = self.engine.simulate_bulk(cgpm_rowids, targets_list,
            constraints_list, Ns=Ns, multiprocess=multiprocess)
        samples = list(itertools.chain.from_iterable(
            zip(*sample) for sample in samples_raw))
        extract_vals = lambda sample: tuple(sample[col] for col in query)
        return [tuple(extract_vals(s) for s in sample) for sample in samples]

    def simulate_ancestral(self, sampids, variables, nsamples, multiprocess=1):
        """Simulate sampids and variables in an ancestral manner.

        Returned samples follow same convention as PanelCat.simulate
        """
        assert sampids == sorted(sampids)
        query = [self._variable_to_index(var) for var in variables]
        rowids = [self._sampid_to_rowid(sampid) for sampid in sampids]
        constraints = [self._get_cgpm_constraints(sampid) for sampid in sampids]
        windows = {sampid: set(self._get_sampid_window(sampid))
            for sampid in sampids}
        parents = {sampid: self._get_parents_from_windows(sampid, windows)
            for sampid in sampids}
        args = [
            (state, sampids, variables, rowids, query,
                constraints, parents, self._variable_to_index, nsamples)
            for state in self.engine.states
        ]
        mapper = parallel_map if multiprocess else map
        self.engine._seed_states()
        samples_raw_list = mapper(_simulate_ancestral_mp, args)
        samples_raw = itertools.chain.from_iterable(samples_raw_list)
        return [
            [[sample[sampid][col] for col in query] for sampid in sampids]
            for sample in samples_raw
        ]

    def dependence_probability_pairwise(self, colnames=None):
        if colnames is None:
            colnames = self.variables
        colnos = [self._variable_to_index(c) for c in colnames]
        return self.engine.dependence_probability_pairwise(colnos)

    def row_similarity_pairwise(self, colnames=None):
        if colnames is None:
            colnames = self.variables
        colnos = [self._variable_to_index(c) for c in colnames]
        return self.engine.row_similarity_pairwise(cols=colnos)

    def _incorporate_new_sampids(self, frame):
        """Incorporate fresh sample ids as new cgpm rows."""
        new_sampids = self._get_new_sampids(frame)
        self.dataset = self.dataset.append(frame[self.variables].loc[new_sampids])
        new_rows = [self._get_sampid_row(sampid) for sampid in new_sampids]
        if self.initialized:
            outputs = self.engine.states[0].outputs
            for row, sampid in zip(new_rows, new_sampids):
                rowid_cgpm = self.engine.states[0].n_rows()
                assert len(row) == len(outputs)
                assert rowid_cgpm == self._sampid_to_rowid(sampid)
                row_cgpm = {i: row[i] for i in outputs if not np.isnan(row[i])}
                self.engine.incorporate(rowid_cgpm, row_cgpm)
        # XXX Do not initialize here! Instead, consider including a dummy row of
        # all zeros or something. The reason that we initialize with the full
        # training set is to ensure that we have a good initial set of
        # hyperparameter grids.
        else:
            self.engine = Engine(
                np.asarray(new_rows),
                num_states=self.chains,
                cctypes=['normal']*len(self.variables_lagged),
                Cd=self._get_variable_dependence_constraints(),
                rng=self.rng,
            )
            self.initialized = True

    def _incorporate_existing_sampids(self, frame):
        """Update existing sampids with NaN entries in cgpm cells."""
        nan_mask = pd.isnull(self.dataset) & ~pd.isnull(frame)
        nan_mask = nan_mask[nan_mask.any(axis=1)]
        if len(nan_mask) == 0:
            return
        cgpm_rowids_cells = []
        # For each new sampid, get the cgpm rowids and cell values to force.
        for nan_sampid, nan_sampid_mask in nan_mask.iterrows():
            self._update_dataset_nan_sampid(frame, nan_sampid, nan_sampid_mask)
            sampid_rowids_cells = self._convert_nan_sampid_to_cgpm_rowid_cells(
                frame, nan_sampid, nan_sampid_mask)
            cgpm_rowids_cells.extend(sampid_rowids_cells)
        # Force the cells in bulk.
        cgpm_rowids, cgpm_cells = zip(*cgpm_rowids_cells)
        self.engine.force_cell_bulk(cgpm_rowids, cgpm_cells)
        # XXX Also force any other sample ids which may have the new sample ids
        # in the window set at nan. Refer to the test case in
        # tests/test_data_transforms.test_incorporate_sampleid_wedged.

    def _update_dataset_nan_sampid(self, frame, nan_sampid, nan_sampid_mask):
        """Populates existing sampid with nan values using values from frame."""
        nan_col_names = nan_sampid_mask[nan_sampid_mask].index
        nan_col_values = frame.loc[nan_sampid, nan_col_names]
        self.dataset.loc[nan_sampid, nan_col_names] = nan_col_values

    def _convert_nan_sampid_to_cgpm_rowid_cells(
            self, frame, nan_sampid, nan_sampid_mask):
        """Returns the cgpm rowid of all windows that nan_sampid participates
        in, and dict containing columns and values to populate."""
        nan_col_names = nan_sampid_mask[nan_sampid_mask].index
        nan_col_idxs = [self._variable_to_index(col) for col in nan_col_names]
        nan_col_values = frame.loc[nan_sampid, nan_col_names].as_matrix()
        cgpm_rowids = self._sampid_to_rowids(nan_sampid)
        cgpm_rowids_cells = [
            {col_idx - lag: value
                for col_idx, value in zip(nan_col_idxs, nan_col_values)}
            if rowid is not None else None
            for lag, rowid in enumerate(cgpm_rowids)
        ]
        return [
            (rowid, cells)
            for rowid, cells in zip(cgpm_rowids, cgpm_rowids_cells)
            if rowid is not None
        ]

    def _get_parents_from_windows(self, sampid, windows):
        """Return list of sampids of parents of the given sampid."""
        return [
            sampid2 for sampid2 in windows
            if sampid2 != sampid and sampid in windows[sampid2]
        ]

    def _get_new_sampids(self, frame):
        """Return sampids in the frame which are not in the dataset."""
        return [
            sampid for sampid in frame.index
            if sampid not in self.dataset.index
        ]

    def _sampid_to_rowid(self, sampid):
        """Return the cgpm rowid representing the sampid."""
        try:
            return self.dataset.index.get_loc(sampid)
        except KeyError:
            return None

    def _sampid_to_rowids(self, sampid):
        """Return the list of cgpm rowids that sampid participates in."""
        # Assuming self.window = 3, the first cgpm rowid that sampid of value 13
        # participates in is the rowid of sampid, and the last cgpm rowid is the
        # rowid of sampid+lag.
        # Example:
        #   lag       L2,L1,L0
        #   rowid=7   11,12,13
        #   rowid=8   12,13,14
        #   rowid=9   13,14,15
        sampids_window = self._get_sampid_window(sampid)
        return [self._sampid_to_rowid(sampid) for sampid in sampids_window]

    def _get_sampid_window(self, sampid):
        """Return the previous sampids in the window of this sampid."""
        return range(sampid, sampid + self.window)

    def _variable_to_index(self, variable, lag=0):
        """Convert variable name to cgpm output index."""
        assert 0 <= lag <= self.lag
        return self.variables.index(variable) * self.window + (self.lag - lag)

    def _variable_to_window_indexes(self, variable):
        """Convert variable name to list of cgpm output indexes in its window."""
        return [self._variable_to_index(variable, l) for l in xrange(self.window)]

    def _variable_indexes(self):
        """Return list of cgpm output indexes, one per variable at lag 0."""
        return [self._variable_to_index(var) for var in self.variables]

    def _get_variable_dependence_constraints(self):
        """Ensure lagged columns and user constraints are modeled as a block."""
        dependencies = [
            list(itertools.chain.from_iterable(
                [self._variable_to_window_indexes(c) for c in block
            ]))
            for block in self.dependencies
        ]
        # Filter out any singleton dependencies.
        return [colnos for colnos in dependencies if len(colnos) > 1]

    def _get_cgpm_constraints(self, sampid):
        # An already incorporated sampid requires no constraints.
        if sampid in self.dataset.index:
            return None
        # Retrieve existing observations in window of a fresh sampid.
        row_values = self._get_sampid_row(sampid)
        assert len(row_values) == len(self.variables_lagged)
        # XXX Require user to specify columns to ignore.
        return {i : v for i, v in enumerate(row_values) if not np.isnan(v)}

    def _get_sampid_row(self, sampid):
        """Convert sampid to row representation with sampid at lag0."""
        sampids_lag = range(sampid - self.lag, sampid + 1)
        return list(itertools.chain.from_iterable(
            (self.dataset[col].get(s, float('nan')) for s in sampids_lag)
            for col in self.variables
        ))

    def _make_dependencies(self, dependencies):
        if dependencies is None:
            dependencies = []
        seen = set(col for block in dependencies for col in block)
        deps_default = [[col] for col in self.variables if col not in seen]
        deps_external = [block for block in dependencies]
        return list(itertools.chain(deps_default, deps_external))

    def to_metadata(self):
        metadata = dict()
        # From constructor.
        metadata['chains'] = self.chains
        metadata['lag'] = self.lag
        metadata['variables'] = self.variables
        # Internal fields.
        metadata['initialized'] = self.initialized
        metadata['engine'] = self.engine.to_metadata() \
            if self.initialized else None
        metadata['dataset.values'] = self.dataset.values.tolist()
        metadata['dataset.index'] = list(self.dataset.index)
        metadata['dataset.columns'] = list(self.dataset.columns)
        # Factory.
        metadata['factory'] = ('trcrpm', 'TRCRP_Mixture')
        return metadata


    @staticmethod
    def from_metadata(metadata, seed):
        model = TRCRP_Mixture(
            chains=metadata['chains'],
            lag=metadata['lag'],
            variables=metadata['variables'],
            rng=np.random.RandomState(seed),
        )
        # Internal fields.
        model.initialized = metadata['initialized']
        model.dataset = pd.DataFrame(
            metadata['dataset.values'],
            index=metadata['dataset.index'],
            columns=metadata['dataset.columns'])
        model.engine = Engine.from_metadata(metadata['engine']) \
            if model.initialized else None
        return model


def _simulate_ancestral_mp((
        state, sampids, variables, rowids, query, constraints, parents,
        variable_to_index, nsamples)):
    return [
        _simulate_ancestral_one(
            state, sampids, variables, rowids, query, constraints, parents,
            variable_to_index)
        for _i in xrange(nsamples)
    ]


def _simulate_ancestral_one(
        state, sampids, variables, rowids, query, constraints, parents,
        variable_to_index):
    """Simulate sampids and variables in an ancestral manner."""
    samples = dict()
    for i, sampid in enumerate(sampids):
        simulated_parents = {
            variable_to_index(var, sampid-sampid_parent) :
                samples[sampid_parent][var_idx]
            for sampid_parent in parents[sampid]
            for var, var_idx in zip(variables, query)
        }
        if constraints[i] is not None:
            constraints[i].update(simulated_parents)
        sample = state.simulate(rowids[i], query, constraints[i])
        samples[sampid] = sample
    return samples
