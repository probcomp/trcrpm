# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pandas as pd
import pytest

from trcrpm import TRCRP_Mixture

nan = float('nan')

DATA_RAW = [
    [86.,    57.,     65.],
    [19.,    62.,     nan],
    [17.,    41.,     17.],
    [nan,     7.,     30.],
    [75.,     1.,     12.],
    [45.,    nan,     72.],
    [48.,    77.,      8.],
    [29.,    nan,     86.],
    [83.,    46.,     38.],
    [nan,    54.,     nan]
]

FRAME = pd.DataFrame(DATA_RAW, columns=['a', 'b', 'c'])

def test_column_indexing():
    rng = np.random.RandomState(2)

    trcrpm = TRCRP_Mixture(chains=1, lag=0, variables=FRAME.columns, rng=rng)
    assert trcrpm.variables_lagged == [
        'a.lag.0',
        'b.lag.0',
        'c.lag.0',
    ]
    # lag 0
    assert trcrpm._variable_to_index('a') == 0
    assert trcrpm._variable_to_index('b') == 1
    assert trcrpm._variable_to_index('c') == 2
    assert trcrpm._variable_indexes() == [0, 1, 2]

    assert trcrpm._get_variable_dependence_constraints() == []

    trcrpm = TRCRP_Mixture(chains=1, lag=2, variables=FRAME.columns, rng=rng)
    assert trcrpm.variables_lagged == [
        'a.lag.2', 'a.lag.1', 'a.lag.0',
        'b.lag.2', 'b.lag.1', 'b.lag.0',
        'c.lag.2', 'c.lag.1', 'c.lag.0'
    ]
    # lag 0
    assert trcrpm._variable_to_index('a') == 2
    assert trcrpm._variable_to_index('b') == 5
    assert trcrpm._variable_to_index('c') == 8
    # lag 1
    assert trcrpm._variable_to_index('a', lag=1) == 1
    assert trcrpm._variable_to_index('b', lag=1) == 4
    assert trcrpm._variable_to_index('c', lag=1) == 7
    # lag 2
    assert trcrpm._variable_to_index('a', lag=2) == 0
    assert trcrpm._variable_to_index('b', lag=2) == 3
    assert trcrpm._variable_to_index('c', lag=2) == 6

    assert trcrpm._variable_indexes() == [2, 5, 8]
    assert trcrpm._variable_to_window_indexes('a') == [2,1,0]
    assert trcrpm._variable_to_window_indexes('b') == [5,4,3]
    assert trcrpm._variable_to_window_indexes('c') == [8,7,6]

    assert trcrpm._get_variable_dependence_constraints() == \
        [[2,1,0], [5,4,3], [8,7,6]]

    trcrpm = TRCRP_Mixture(chains=1, lag=5, variables=FRAME.columns, rng=rng)
    assert trcrpm.variables_lagged == [
        'a.lag.5', 'a.lag.4', 'a.lag.3', 'a.lag.2', 'a.lag.1', 'a.lag.0',
        'b.lag.5', 'b.lag.4', 'b.lag.3', 'b.lag.2', 'b.lag.1', 'b.lag.0',
        'c.lag.5', 'c.lag.4', 'c.lag.3', 'c.lag.2', 'c.lag.1', 'c.lag.0'
    ]
    # lag 0
    assert trcrpm._variable_to_index('a') == 5
    assert trcrpm._variable_to_index('b') == 11
    assert trcrpm._variable_to_index('c') == 17
    # lag 1
    assert trcrpm._variable_to_index('a', lag=1) == 4
    assert trcrpm._variable_to_index('b', lag=1) == 10
    assert trcrpm._variable_to_index('c', lag=1) == 16
    # lag 2
    assert trcrpm._variable_to_index('a', lag=2) == 3
    assert trcrpm._variable_to_index('b', lag=2) == 9
    assert trcrpm._variable_to_index('c', lag=2) == 15
    # lag 5
    assert trcrpm._variable_to_index('a', lag=5) == 0
    assert trcrpm._variable_to_index('b', lag=5) == 6
    assert trcrpm._variable_to_index('c', lag=5) == 12

    assert trcrpm._variable_indexes() == [5, 11, 17]
    assert trcrpm._variable_to_window_indexes('a') == [5,4,3,2,1,0]
    assert trcrpm._variable_to_window_indexes('b') == [11,10,9,8,7,6]
    assert trcrpm._variable_to_window_indexes('c') == [17,16,15,14,13,12]

    assert trcrpm._get_variable_dependence_constraints() == \
        [[5,4,3,2,1,0], [11,10,9,8,7,6], [17,16,15,14,13,12]]


def test_dependence_constraints():
    rng = np.random.RandomState(2)

    # Lag 0 dependencies should skip singleton 'c'.
    trcrpm = TRCRP_Mixture(chains=1, lag=0, variables=FRAME.columns, rng=rng,
        dependencies=[['a','b'],['c']])
    assert trcrpm._get_variable_dependence_constraints() == [[0,1]]

    # Lag 0 dependencies with all constrained.
    trcrpm = TRCRP_Mixture(chains=1, lag=0, variables=FRAME.columns, rng=rng,
        dependencies=[['a','b','c']])
    assert trcrpm._get_variable_dependence_constraints() == [[0,1,2]]

    # Lag 1 dependencies with two constraints.
    trcrpm = TRCRP_Mixture(chains=1, lag=1, variables=FRAME.columns, rng=rng,
        dependencies=[['a','b']])
    assert trcrpm._get_variable_dependence_constraints() == [[5,4], [1,0,3,2]]

    # Lag 1 dependency with single constraint.
    trcrpm = TRCRP_Mixture(chains=1, lag=1, variables=FRAME.columns, rng=rng,
        dependencies=[['a']])
    assert trcrpm._get_variable_dependence_constraints() == [
        [3,2] , [5,4], [1,0]
    ]

    # Multiple customer dependencies. Capturing a runtime error since
    # incorporate is going to use multiprocess so parallel_map captures the
    # ValueError and throws a RuntimeError.
    with pytest.raises(RuntimeError):
        trcrpm = TRCRP_Mixture(chains=1, lag=0, variables=FRAME.columns, rng=rng,
            dependencies=[['a','c'], ['a','b']])
        trcrpm.incorporate(FRAME)


def test_tabulate_lagged_data():
    rng = np.random.RandomState(2)

    trcrpm = TRCRP_Mixture(chains=1, lag=0, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)
    tabulated_data = trcrpm.engine.states[0].data_array()
    expected_data = DATA_RAW
    assert np.allclose(tabulated_data, DATA_RAW, equal_nan=True)

    trcrpm = TRCRP_Mixture(chains=1, lag=2, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)
    tabulated_data = trcrpm.engine.states[0].data_array()
    expected_data = [
        [nan,  nan,  86.,  nan,  nan,  57.,  nan,  nan,  65.],
        [nan,  86.,  19.,  nan,  57.,  62.,  nan,  65.,  nan],
        [86.,  19.,  17.,  57.,  62.,  41.,  65.,  nan,  17.],
        [19.,  17.,  nan,  62.,  41.,   7.,  nan,  17.,  30.],
        [17.,  nan,  75.,  41.,   7.,   1.,  17.,  30.,  12.],
        [nan,  75.,  45.,   7.,   1.,  nan,  30.,  12.,  72.],
        [75.,  45.,  48.,   1.,  nan,  77.,  12.,  72.,   8.],
        [45.,  48.,  29.,  nan,  77.,  nan,  72.,   8.,  86.],
        [48.,  29.,  83.,  77.,  nan,  46.,   8.,  86.,  38.],
        [29.,  83.,  nan,  nan,  46.,  54.,  86.,  38.,  nan],
    ]
    assert np.allclose(tabulated_data, expected_data, equal_nan=True)


def test_incorporate_sampleid_existing():
    rng = np.random.RandomState()
    trcrpm = TRCRP_Mixture(chains=1, lag=2, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    new_frame = pd.DataFrame([
            [nan, nan, 99.],
            [44., nan, nan],
            [21., nan, 88.],
        ],
        index=[1, 3, 9],
        columns=['a', 'b', 'c'],
    )
    trcrpm.incorporate(new_frame)
    tabulated_data = trcrpm.engine.states[0].data_array()
    expected_data = [
        [nan,  nan,  86.,  nan,  nan,  57.,  nan,  nan,  65.],
        [nan,  86.,  19.,  nan,  57.,  62.,  nan,  65.,  99.],
        [86.,  19.,  17.,  57.,  62.,  41.,  65.,  99.,  17.],
        [19.,  17.,  44.,  62.,  41.,   7.,  99.,  17.,  30.],
        [17.,  44.,  75.,  41.,   7.,   1.,  17.,  30.,  12.],
        [44.,  75.,  45.,   7.,   1.,  nan,  30.,  12.,  72.],
        [75.,  45.,  48.,   1.,  nan,  77.,  12.,  72.,   8.],
        [45.,  48.,  29.,  nan,  77.,  nan,  72.,   8.,  86.],
        [48.,  29.,  83.,  77.,  nan,  46.,   8.,  86.,  38.],
        [29.,  83.,  21.,  nan,  46.,  54.,  86.,  38.,  88.],
    ]
    assert np.allclose(tabulated_data, expected_data, equal_nan=True)


def test_incorporate_sampleid_non_contiguous():
    rng = np.random.RandomState(1)
    trcrpm = TRCRP_Mixture(chains=1, lag=2, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    # Now incorporate an non-contiguous row skipping last_sampled_id+1.
    last_sampled_id = max(FRAME.index)
    frame_new = pd.DataFrame(
        [[11, 12, float('nan')]],
        columns=['a', 'b', 'c'],
        index=[last_sampled_id+2]
    )
    trcrpm.incorporate(frame_new)
    tabulated_data = trcrpm.engine.states[0].data_array()
    expected_data = [
        [nan,  nan,  86.,  nan,  nan,  57.,  nan,  nan,  65.],
        [nan,  86.,  19.,  nan,  57.,  62.,  nan,  65.,  nan],
        [86.,  19.,  17.,  57.,  62.,  41.,  65.,  nan,  17.],
        [19.,  17.,  nan,  62.,  41.,   7.,  nan,  17.,  30.],
        [17.,  nan,  75.,  41.,   7.,   1.,  17.,  30.,  12.],
        [nan,  75.,  45.,   7.,   1.,  nan,  30.,  12.,  72.],
        [75.,  45.,  48.,   1.,  nan,  77.,  12.,  72.,   8.],
        [45.,  48.,  29.,  nan,  77.,  nan,  72.,   8.,  86.],
        [48.,  29.,  83.,  77.,  nan,  46.,   8.,  86.,  38.],
        [29.,  83.,  nan,  nan,  46.,  54.,  86.,  38.,  nan],
        [nan,  nan,  11,   54.,  nan,  12,   nan,  nan,  nan],
    ]
    assert np.allclose(tabulated_data, expected_data, equal_nan=True)

@pytest.mark.xfail(strict=True, reason='Feature not implemented.')
def test_incorporate_sampleid_wedged():
    rng = np.random.RandomState(1)
    trcrpm = TRCRP_Mixture(chains=1, lag=2, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    # Now incorporate an non-contiguous row skipping last_sampled_id+1.
    last_sampled_id = max(FRAME.index)
    frame_new = pd.DataFrame(
        [[11, 12, float('nan')]],
        columns=['a', 'b', 'c'],
        index=[last_sampled_id+2]
    )
    trcrpm.incorporate(frame_new)

    # Now bring back the missing observation at last_sampled_id+1.
    # XXX The xfailure happens in this step.
    frame_new = pd.DataFrame(
        [[3, 3, 3]],
        columns=['a', 'b', 'c'],
        index=[last_sampled_id+1]
    )
    trcrpm.incorporate(frame_new)
    tabulated_data = trcrpm.engine.states[0].data_array()
    expected_data = [
        [nan,  nan,  86.,  nan,  nan,  57.,  nan,  nan,  65.],
        [nan,  86.,  19.,  nan,  57.,  62.,  nan,  65.,  nan],
        [86.,  19.,  17.,  57.,  62.,  41.,  65.,  nan,  17.],
        [19.,  17.,  nan,  62.,  41.,   7.,  nan,  17.,  30.],
        [17.,  nan,  75.,  41.,   7.,   1.,  17.,  30.,  12.],
        [nan,  75.,  45.,   7.,   1.,  nan,  30.,  12.,  72.],
        [75.,  45.,  48.,   1.,  nan,  77.,  12.,  72.,   8.],
        [45.,  48.,  29.,  nan,  77.,  nan,  72.,   8.,  86.],
        [48.,  29.,  83.,  77.,  nan,  46.,   8.,  86.,  38.],
        [29.,  83.,  nan,  nan,  46.,  54.,  86.,  38.,  nan],
        [nan,   3.,  11.,  54.,   3.,  12.,  nan,   3.,  nan], # row updated.
        [83.,  nan,   3.,  46.,  54.,   3.,  38.,  nan,   3.], # new row.
    ]
    assert np.allclose(tabulated_data, expected_data, equal_nan=True)


def test_sampid_to_rowid():
    rng = np.random.RandomState(2)
    trcrpm = TRCRP_Mixture(chains=1, lag=0, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)
    for i in xrange(len(FRAME)):
        assert trcrpm._sampid_to_rowid(i) == i
        assert trcrpm._sampid_to_rowid(i) == i


def test_simulate_dimensions():
    rng = np.random.RandomState(1)
    trcrpm = TRCRP_Mixture(chains=4, lag=3, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    def check_dims_correct(sampids, variables, nsamples, simulator):
        samples = simulator(sampids, variables, nsamples, multiprocess=0)
        # Number of chains is 4.
        assert len(samples) == nsamples * 4
        for sample in samples:
            assert len(sample) == len(sampids)
            for subsample in sample:
                assert len(subsample) == len(variables)

    for simulator in [trcrpm.simulate, trcrpm.simulate_ancestral]:
        check_dims_correct([1], ['a'], 4, simulator)
        check_dims_correct([1], ['a', 'b'], 7, simulator)
        check_dims_correct([1,2], ['a'], 1, simulator)
        check_dims_correct([1,2,3], ['c', 'b'], 10, simulator)
        check_dims_correct([9, 10, 11, 12], ['c', 'b'], 10, simulator)


def test_get_cgpm_constraints():
    rng = np.random.RandomState(2)
    trcrpm = TRCRP_Mixture(chains=1, lag=3, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    # For reference, lag=3 implies the following tabulation
    # column a          column b        column c
    # 0 1 2 3           4 5 6 7         8 9 10 11

    # Incorporated sampids should have no constraints.
    for sampid in trcrpm.dataset.index:
        assert trcrpm._get_cgpm_constraints(sampid) is None

    # Fresh sampid 10 should have appropriate constraints.
    constraints_10 = trcrpm._get_cgpm_constraints(10)
    assert constraints_10 == {
        # column a      column b        column c
        0:29, 1:83,     5:46, 6:54,     8:86 , 9:38
    }
    # Incorporating data into sampid 9 should update constraints for 10.
    frame_new = pd.DataFrame(
        [[-12, nan, -12]],
        columns=['a', 'b', 'c'],
        index=[9]
    )
    trcrpm.incorporate(frame_new)
    constraints_10 = trcrpm._get_cgpm_constraints(10)
    assert constraints_10 == {
        # column a            column b           column c
        0:29, 1:83, 2:-12,    5:46, 6:54,        8:86 , 9:38, 10:-12
    }

    # Fresh sampid 12 should have appropriate constraints.
    constraints_12 = trcrpm._get_cgpm_constraints(12)
    assert constraints_12 == {
        # column a          column b            column c
        0:-12,              4:54,               8:-12
    }
    # Incorporating data into sampid 11 should update constraints for 12.
    frame_new = pd.DataFrame(
        [[-44, -48, -47]],
        columns=['a', 'b', 'c'],
        index=[11]
    )
    trcrpm.incorporate(frame_new)
    constraints_12 = trcrpm._get_cgpm_constraints(12)
    assert constraints_12 == {
        # column a          column b            column c
        0:-12, 2:-44,       4:54, 6:-48,        8:-12, 10:-47
    }


def test_get_temporal_regimes_crash():
    rng = np.random.RandomState(1)
    trcrpm = TRCRP_Mixture(chains=4, lag=3, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)
    for variable in trcrpm.variables:
        regimes_all = trcrpm.get_temporal_regimes(variable)
        assert np.shape(regimes_all) == (trcrpm.chains, len(trcrpm.dataset))
        regimes_some = trcrpm.get_temporal_regimes(variable, sampids=[0,1,2])
        assert np.shape(regimes_some) == (trcrpm.chains, 3)
