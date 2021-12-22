# Installation Guide

There are various ways to install this package.

### Installing using conda

The package is available on the [probcomp Anaconda cloud channel](https://anaconda.org/probcomp/trcrpm).
The latest release can be installed into your conda environment using:

```bash
$ conda install -c probcomp trcrpm
```

To install the nightly "bleeding edge" version, which has not yet been tagged
into a new release, use:

```bash
$ conda install -c probcomp/label/edge trcrpm
```

If you have not used conda before and do not have an environment, these
steps show an end-to-end installation process:

```bash
$ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
$ bash miniconda.sh -b -p ${HOME}/miniconda
$ export PATH="${HOME}/miniconda/bin:${PATH}"
$ conda update --yes conda
$ conda create --name probcomp --channel probcomp/label/edge --yes python=2.7 trcrpm
$ source activate probcomp
```

### Installing into a virtualenv (for Linux users)

All 3rd party dependencies are available in the Ubuntu 16.04 LTS standard
repositories (apt), so the process should be straightforward. Replicate the
instructions in the [Dockerfile](./docker/ubuntu1604) by writing the commands
directly in your own Linux environment.

### Installing into a docker image

If all else fails, you can obtain obtain a shell with the package by building
the Dockerfile. The first step builds an image called `probcomp/trcrpm`, the
second step runs a container with name `trcrp` and gives you a shell, and the
third step activates the python virtualenv containing the software.

```bash
$ docker build -t probcomp/trcrpm -f docker/ubuntu1604 .
$ docker run -it --name trcrp probcomp/trcrpm /bin/bash
root@9cc66a75a0e2:/# source /venv/bin/activate
```
