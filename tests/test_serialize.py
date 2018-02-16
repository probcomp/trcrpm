# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pandas as pd
import importlib

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


def test_serialize_trcrp_mixture():
    rng = np.random.RandomState(1)
    trcrpm = TRCRP_Mixture(chains=4, lag=3, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    metadata = trcrpm.to_metadata()
    modulename, attributename = metadata['factory']
    module = importlib.import_module(modulename)
    builder = getattr(module, attributename)
    trcrpm2 = builder.from_metadata(metadata, seed=1)
    assert isinstance(trcrpm2, TRCRP_Mixture)

def test_serialize_hierarchical_trcrp_mixture():
    rng = np.random.RandomState(1)
    trcrpm = Hierarchical_TRCRP_Mixture(
        chains=4, lag=3, variables=FRAME.columns, rng=rng)
    trcrpm.incorporate(FRAME)

    metadata = trcrpm.to_metadata()
    modulename, attributename = metadata['factory']
    module = importlib.import_module(modulename)
    builder = getattr(module, attributename)
    trcrpm2 = builder.from_metadata(metadata, seed=1)
    assert isinstance(trcrpm2, Hierarchical_TRCRP_Mixture)

