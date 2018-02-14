# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pandas as pd
import pytest

from trcrpm import Hierarchical_TRCRP_Mixture
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

def test_trcrp_mixture_all_dependent():
    rng = np.random.RandomState(2)
    trcrpm = TRCRP_Mixture(chains=3, lag=3, variables=FRAME.columns, rng=rng)
    with pytest.raises(ValueError):
        # No data incorporated yet.
        trcrpm.resample_all(steps=10)
    trcrpm.incorporate(FRAME)
    trcrpm.resample_all(steps=10)
    for state in trcrpm.engine.states:
        assert len(state.views) == 1
    trcrpm.resample_hyperparameters(steps=5)
    regimes_a = trcrpm.get_temporal_regimes('a')
    regimes_b = trcrpm.get_temporal_regimes('b')
    regimes_c = trcrpm.get_temporal_regimes('c')
    assert np.all(regimes_a == regimes_b)
    assert np.all(regimes_a == regimes_c)

def test_trcrp_hierarchical_mixture_crashes():
    rng = np.random.RandomState(2)
    trcrpm = Hierarchical_TRCRP_Mixture(
        chains=3, lag=3, variables=FRAME.columns, rng=rng)
    with pytest.raises(ValueError):
        # No data incorporated yet.
        trcrpm.resample_all(steps=10)
    trcrpm.incorporate(FRAME)
    trcrpm.resample_all(steps=10)
    trcrpm.resample_hyperparameters(steps=5)
    regimes_a = trcrpm.get_temporal_regimes('a')
    regimes_b = trcrpm.get_temporal_regimes('b')
    regimes_c = trcrpm.get_temporal_regimes('c')
    assert not (
        np.all(regimes_a == regimes_b) and np.all(regimes_a == regimes_c))
