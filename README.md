# Temporally-Reweighted Chinese Restaurant Process Mixture Models

[![Build Status](https://travis-ci.org/probcomp/trcrpm.svg?branch=master)](https://travis-ci.org/probcomp/trcrpm)
[![Anaconda Version Badge](https://anaconda.org/probcomp/trcrpm/badges/version.svg)](https://anaconda.org/probcomp/trcrpm)
[![Anaconda Platforms Badge](https://anaconda.org/probcomp/trcrpm/badges/platforms.svg)](https://anaconda.org/probcomp/trcrpm)


A nonparametric Bayesian method for clustering, imputation, and forecasting
in multivariate time series data.

## Installing

There are various ways to install this package. The easiest way is to pull
the package from conda,

```bash
$ conda install -c probcomp trcrpm
```

For more information, see [INSTALLING.md](./INSTALLING.md)

## Getting started

For tutorials showing how to use the method, refer to the
[tutorials](./tutorials) directory.

<img src="./tutorials/resources/bimodal-posteriors.png" width="750" align="middle">

## Documentation

The [API reference](https://probcomp-1.csail.mit.edu/trcrpm/doc/api.html) is
available online. Use `make doc` to build the documentation locally (needs
[`sphinx`](http://www.sphinx-doc.org/en/stable/install.html) and
[`napoleon`](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)).

## References

Feras A. Saad and Vikash K. Mansinghka, [Temporally-Reweighted Chinese
  Restaurant Process Mixtures For Clustering, Imputing, and
  Forecasting Multivariate Time Series](http://proceedings.mlr.press/v84/saad18a.html).
  In AISTATS 2018: _Proceedings of the 20th International Conference on Artificial
  Intelligence and Statistics_, Proceedings of Machine Learning Research 84,
  Playa Blanca, Lanzarote, Canary Islands, 2018.

To cite this work, please use the following BibTeX reference.

```bibtex
@inproceedings{saad2018trcrpm,
author          = {Saad, Feras A. and Mansinghka, Vikash K.},
title           = {Temporally-reweighted {C}hinese restaurant process mixtures for clustering, imputing, and forecasting multivariate time series},
booktitle       = {AISTATS 2018: Proceedings of the 21st International Conference on Artificial Intelligence and Statistics},
series          = {Proceedings of Machine Learning Research},
volume          = 84,
pages           = {755--764},
publisher       = {PMLR},
address         = {Playa Blanca, Lanzarote, Canary Islands},
year            = {2018},
keywords        = {probabilistic inference, multivariate time series, nonparametric Bayes, structure learning},
}

```

## License

Copyright (c) 2015-2018 MIT Probabilistic Computing Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
