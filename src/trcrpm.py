# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

import numpy as np
import pandas as pd

from cgpm.crosscat.engine import Engine
from cgpm.utils.parallel_map import parallel_map


class Hierarchical_TRCRP_Mixture(object):
    """Hierarchical Temporally-Reweighted Chinese Restaurant Process Mixture.

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

        Parameters
        ----------
        chains: int
            Number of parallel MCMC chains to use for inference.
        lag : int
            Number of time points in the history to use for reweighting the
            CRP. If lag is zero, then all temporal dependencies are removed
            and the model becomes a standard CRP mixture.
        variables : list of str
            Human-readable names of the time series to be modeled.
        rng : np.random.RandomState
            Source of entropy
        dependencies : list of tuple<string>, optional
            Blocks of variables which are deterministically constrained to be
            modeled jointly. Defaults to no deterministic constraints.
        """
        # From constructor.
        self.chains = chains
        self.lag = lag
        self.variables = list(variables)
        self.rng = rng
        self.dependencies = self._make_dependencies(dependencies)
        # Derived attributes.
        self.window = self.lag + 1
        self.variables_lagged = list(itertools.chain.from_iterable([
            ['%s.lag.%d' % (varname, i,) for i in xrange(self.lag, -1, -1)]
            for varname in self.variables
        ]))
        self.variable_index = {var: i for i, var in enumerate(self.variables)}
        for variable in self.variables:
            variable_idx = self._variable_to_index(variable)
            assert self.variables_lagged[variable_idx]=='%s.lag.0' % (variable,)
        # Internal attributes.
        self.dataset = pd.DataFrame()
        self.engine = None
        self.initialized = None

    def incorporate(self, frame):
        """Incorporate new observations.

        Parameters
        ----------
        frame : pd.DataFrame
            DataFrame containing new observations.
        """
        assert set(frame.columns) == set(self.variables)
        self._incorporate_new_timepoints(frame)
        # XXX Improve this function.
        self._incorporate_existing_timepoints(frame)
        assert self.engine.states[0].n_rows() == len(self.dataset)

    def resample_all(self, steps=None, seconds=None):
        """Run MCMC inference on entire latent state

        Parameters
        ----------
        steps : int, optional
            Number of full Gibbs sweeps through all kernels, default is 1.
        seconds : int, optional
            Maximum number of seconds to run inference before timing out,
            default is None.

        If both `steps` and `seconds` are specified, then the min is taken. That
        is, MCMC inference will run until the given number Gibbs steps are taken
        or until the given number of seconds elapse, whichever comes first.
        """
        self._transition(N=steps, S=seconds, backend='lovecat')

    def resample_hyperparameters(self, steps=None, seconds=None, variables=None):
        """Run MCMC inference on variable hyperparameters.

        Parameters
        ----------
        steps : int, optional
            Number of full Gibbs sweeps through all kernels, default is 1.
        seconds : int, optional
            Maximum number of seconds to run inference before timing out,
            default is None.
        variables : list of str
            List of time series variables whose hyperparameters to target,
            default is all.

        See Also
        --------
        resample_all
        """
        variables_transition = variables or self.variables
        variable_indexes = list(itertools.chain.from_iterable([
            self._variable_to_window_indexes(v) for v in variables_transition
        ]))
        self._transition(N=steps, S=seconds, cols=variable_indexes,
            kernels=['view_alphas','column_hypers'], backend='cgpm')

    def simulate(self, timepoints, variables, nsamples, multiprocess=1):
        """Generate simulations from the posterior distribution.

        Parameters
        ----------
        timepoints : list of int
            List of integer-valued time steps to simulate
        variables : list of str
            Names of time series which to simulate from.
        nsamples : int
            Number of predictive samples to generate from each chain.

        Returns
        -------
        np.ndarray
            3D array of generated samples. The dimensions of the returned list
            are `(self.chains*nsamples, len(timepoints), len(variables))`, so
            that `result[i][j][k]` contains a simulation of `variables[k],` at
            timepoint `j`, from chain `i`.

            // model has 2 chains, so chains * nsamples = 6 samples returned.
            >> model.simulate([1, 4], ['a', 'b'], 3)

            |<-----------chain 0----------->||<-----------chain 1----------->|
            [sample0.0, sample0.1, sample0.2, sample1.0, sample1.1, sample1.2]

            sample0.0: ((sim0.0_a1, sim0.0_b1), (sim0.0_a40, sim0.0_b40))
            sample0.1: ((sim0.1_a1, sim0.1_b1), (sim0.1_a40, sim0.1_b40))
            sample0.2: ((sim0.2_a1, sim0.2_b1), (sim0.2_a40, sim0.2_b40))

            sample1.0: ((sim1.0_a1, sim1.0_b1), (sim1.0_a40, sim1.0_b40))
            sample1.1: ((sim1.1_a1, sim1.1_b1), (sim1.1_a40, sim1.1_b40))
            sample1.2: ((sim1.2_a1, sim1.2_b1), (sim1.2_a40, sim1.2_b40))
        """
        cgpm_rowids = [self._timepoint_to_rowid(t) for t in timepoints]
        constraints_list = [self._get_cgpm_constraints(t) for t in timepoints]
        targets = [self._variable_to_index(var) for var in variables]
        targets_list = [targets] * len(cgpm_rowids)
        Ns = [nsamples] * len(cgpm_rowids)
        samples_raw_bulk = self.engine.simulate_bulk(cgpm_rowids, targets_list,
            constraints_list, Ns=Ns, multiprocess=multiprocess)
        samples_raw = list(itertools.chain.from_iterable(
            zip(*sample) for sample in samples_raw_bulk))
        samples = np.asarray([
            [[sample[t] for t in targets] for sample in sample_chain]
            for sample_chain in samples_raw
        ])
        return samples

    def simulate_ancestral(self, timepoints, variables, nsamples, multiprocess=1):
        """Generate simulations from the posterior distribution ancestrally.

        See Also
        --------
        simulate
        """
        assert timepoints == sorted(timepoints)
        targets = [self._variable_to_index(var) for var in variables]
        rowids = [self._timepoint_to_rowid(t) for t in timepoints]
        constraints = [self._get_cgpm_constraints(t) for t in timepoints]
        windows = {timepoint: set(self._get_timepoint_window(timepoint))
            for timepoint in timepoints}
        parents = {timepoint: self._get_parents_from_windows(timepoint, windows)
            for timepoint in timepoints}
        args = [
            (state, timepoints, variables, rowids, targets,
                constraints, parents, self._variable_to_index, nsamples)
            for state in self.engine.states
        ]
        mapper = parallel_map if multiprocess else map
        self.engine._seed_states()
        samples_raw_list = mapper(_simulate_ancestral_mp, args)
        samples_raw = itertools.chain.from_iterable(samples_raw_list)
        samples = np.asarray([
            [[sample[t][variable] for variable in targets] for t in timepoints]
            for sample in samples_raw
        ])
        return samples

    def dependence_probability_pairwise(self, variables=None):
        """Compute posterior dependence probabilities between variables.

        Parameters
        ----------
        variables : list of str, optional
            List of variables to include in the returned array. Defaults to
            `self.variables`.

        Returns
        -------
        np.ndarray
            3D array containing pairwise dependence probabilities of
            `variables` from each chain. The dimensions of the returned
            array are `(self.chains, len(variables), len(variables))`, so
            that result[i,j,k] = 1 if `variables[j]` and `variables[k]` are
            dependent according to chain `i`, and 0 otherwise.
        """
        if variables is None:
            variables = self.variables
        varnos = [self._variable_to_index(var) for var in variables]
        D = self.engine.dependence_probability_pairwise(cols=varnos)
        return np.asarray(D)

    def get_temporal_regimes(self, variable, timepoints=None):
        """Return latent temporal regime at `timepoints` of the given `variable`.

        Parameters
        ----------
        variable : str
            Name of the time series variable to query.
        timepoints : list of int, optional
            List of timepoints at which to get the latent temporal regime value,
            defaults to all observed timesteps.

        Returns
        -------
        np.ndarray
            2D array containing latent temporal regime at `timepoints` of the
            given variable, for each chain. The dimensions of the returned array
            are `(self.chains, len(timepoints))`, where `result[i][t]` is the
            value of the hidden temporal regime at `timepoints[t]`, according to
            chain `i`.

            NOTE: The numerical (integer) values of the regimes are immaterial.
        """
        if timepoints is None:
            timepoints = self.dataset.index
        rowids = [self._timepoint_to_rowid(t) for t in timepoints]
        varno = self._variable_to_index(variable)
        regimes = [[state.view_for(varno).Zr(rowid) for rowid in rowids]
            for state in self.engine.states]
        return np.asarray(regimes)

    def _transition(self, **kwargs):
        """Helper for MCMC resample methods (full interface not exposed)."""
        if self.engine is None:
            raise ValueError('No data incorporate yet.')
        backend = kwargs.pop('backend', None)
        kwargs['cols'] = kwargs.pop('cols', self._variable_indexes())
        if backend in ['cgpm', None]:
            self.engine.transition(**kwargs)
        elif backend in ['lovecat']:
            self.engine.transition_lovecat(**kwargs)
        elif backend in ['loom']:
            self.engine.transition_loom(**kwargs)
        else:
            raise ValueError('Unknown backend: %s' % (backend,))

    def _incorporate_new_timepoints(self, frame):
        """Incorporate fresh sample ids as new cgpm rows."""
        new_timepoints = frame.index[~frame.index.isin(self.dataset.index)]
        new_observations = frame[self.variables].loc[new_timepoints]
        self.dataset = self.dataset.append(new_observations)
        new_rows = [self._get_timepoint_row(t) for t in new_timepoints]
        if self.initialized:
            outputs = self.engine.states[0].outputs
            assert all(len(row) == len(outputs) for row in new_rows)
            rowids_cgpm = range(
                self.engine.states[0].n_rows(),
                self.engine.states[0].n_rows() + len(new_rows)
            )
            observations_cgpm = [
                {i: row[i] for i in outputs if not np.isnan(row[i])}
                for row in new_rows
            ]
            assert all(
                rowid_cgpm == self._timepoint_to_rowid(timepoint)
                for timepoint, rowid_cgpm in zip(new_timepoints, rowids_cgpm)
            )
            self.engine.incorporate_bulk(rowids_cgpm, observations_cgpm)
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

    def _incorporate_existing_timepoints(self, frame):
        """Update existing timepoints with NaN entries in cgpm cells."""
        nan_mask = pd.isnull(self.dataset) & ~pd.isnull(frame)
        nan_mask = nan_mask[nan_mask.any(axis=1)]
        if len(nan_mask) == 0:
            return
        cgpm_rowids_cells = []
        # For each new timepoint, get the cgpm rowids and cell values to force.
        for nan_timepoint, nan_timepoint_mask in nan_mask.iterrows():
            self._update_dataset_nan_timepoint(
                frame, nan_timepoint, nan_timepoint_mask)
            timepoint_rowids_cells = \
                self._convert_nan_timepoint_to_cgpm_rowid_cells(
                        frame, nan_timepoint, nan_timepoint_mask)
            cgpm_rowids_cells.extend(timepoint_rowids_cells)
        # Force the cells in bulk.
        cgpm_rowids, cgpm_cells = zip(*cgpm_rowids_cells)
        self.engine.force_cell_bulk(cgpm_rowids, cgpm_cells)
        # XXX Also force any other sample ids which may have the new sample ids
        # in the window set at nan. Refer to the test case in
        # tests/test_data_transforms.test_incorporate_sampleid_wedged.

    def _update_dataset_nan_timepoint(
            self, frame, nan_timepoint, nan_timepoint_mask):
        """Populates timepoint with nan values in self.dataset using frame."""
        nan_col_names = nan_timepoint_mask[nan_timepoint_mask].index
        nan_col_values = frame.loc[nan_timepoint, nan_col_names]
        self.dataset.loc[nan_timepoint, nan_col_names] = nan_col_values

    def _convert_nan_timepoint_to_cgpm_rowid_cells(
            self, frame, nan_timepoint, nan_timepoint_mask):
        """Returns the cgpm rowid of all windows that nan_timepoint participates
        in, and dict containing columns and values to populate."""
        nan_col_names = nan_timepoint_mask[nan_timepoint_mask].index
        nan_col_idxs = [self._variable_to_index(col) for col in nan_col_names]
        nan_col_values = frame.loc[nan_timepoint, nan_col_names].as_matrix()
        cgpm_rowids = self._timepoint_to_rowids(nan_timepoint)
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

    def _get_parents_from_windows(self, timepoint, windows):
        """Return list of timepoints of parents of the given timepoint."""
        return [
            timepoint2 for timepoint2 in windows
            if timepoint2 != timepoint and timepoint in windows[timepoint2]
        ]

    def _timepoint_to_rowid(self, timepoint):
        """Return the cgpm rowid representing the timepoint."""
        try:
            return self.dataset.index.get_loc(timepoint)
        except KeyError:
            return None

    def _timepoint_to_rowids(self, timepoint):
        """Return the list of cgpm rowids that timepoint participates in."""

        # Assuming self.window = 3, the first cgpm rowid that timepoint of value
        # 13 participates in is the rowid of timepoint, and the last cgpm rowid
        # is the rowid of timepoint+lag.

        # Example:
        #   lag       L2,L1,L0
        #   rowid=7   11,12,13
        #   rowid=8   12,13,14
        #   rowid=9   13,14,15
        timepoints_window = self._get_timepoint_window(timepoint)
        return [self._timepoint_to_rowid(t) for t in timepoints_window]

    def _get_timepoint_window(self, timepoint):
        """Return the previous timepoints in the window of this timepoint."""
        return range(timepoint, timepoint + self.window)

    def _variable_to_index(self, variable, lag=0):
        """Convert variable name to cgpm output index."""
        assert 0 <= lag <= self.lag
        return self.variable_index[variable] * self.window + (self.lag - lag)

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

    def _get_cgpm_constraints(self, timepoint):
        # An already incorporated timepoint requires no constraints.
        if timepoint in self.dataset.index:
            return None
        # Retrieve existing observations in window of a fresh timepoint.
        row_values = self._get_timepoint_row(timepoint)
        assert len(row_values) == len(self.variables_lagged)
        # XXX Require user to specify columns to ignore.
        return {i : v for i, v in enumerate(row_values) if not np.isnan(v)}

    def _get_timepoint_row(self, timepoint):
        """Convert timepoint to row representation with timepoint at lag0."""
        timepoints_lag = range(timepoint - self.lag, timepoint + 1)
        return list(itertools.chain.from_iterable(
            (self.dataset[col].get(t, float('nan')) for t in timepoints_lag)
            for col in self.variables
        ))

    def _make_dependencies(self, dependencies):
        """Return combination of default and user's dependence constraints."""
        if dependencies is None:
            dependencies = []
        seen = set(col for block in dependencies for col in block)
        deps_default = [[col] for col in self.variables if col not in seen]
        deps_external = [block for block in dependencies]
        return tuple(tuple(itertools.chain(deps_default, deps_external)))

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


class TRCRP_Mixture(Hierarchical_TRCRP_Mixture):
    """Temporally-Reweighted Chinese Restaurant Process Mixture.

    The TRCRP_Mixture is a special case of the Hierarchical_TRCRP_Mixture where
    all time series are deterministically constrained to be modeled jointly.

    See Also
    --------
    Hierarchical_TRCRP_Mixture
    """
    def __init__(self, chains, lag, variables, rng):
        super(TRCRP_Mixture, self).__init__(
            chains, lag, variables, rng, dependencies=[variables])


# Multiprocessing helpers.
# These functions must be defined top-level in the module to work with
# parallel_map.

def _simulate_ancestral_mp((state, timepoints, variables, rowids, targets,
        constraints, parents, variable_to_index, nsamples)):
    """Simulate timepoints and variables ancestrally (multiple samples)."""
    return [
        _simulate_ancestral_one(
            state, timepoints, variables, rowids, targets, constraints, parents,
            variable_to_index)
        for _i in xrange(nsamples)
    ]


def _simulate_ancestral_one(state, timepoints, variables, rowids, targets,
        constraints, parents, variable_to_index):
    """Simulate timepoints and variables ancestrally (one sample)."""
    samples = dict()
    for i, timepoint in enumerate(timepoints):
        simulated_parents = {
            variable_to_index(var, timepoint-timepoint_parent) :
                samples[timepoint_parent][var_idx]
            for timepoint_parent in parents[timepoint]
            for var, var_idx in zip(variables, targets)
        }
        if constraints[i] is not None:
            constraints[i].update(simulated_parents)
        sample = state.simulate(rowids[i], targets, constraints[i])
        samples[timepoint] = sample
    return samples
