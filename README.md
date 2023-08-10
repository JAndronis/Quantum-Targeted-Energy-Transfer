# Quantum Targeted Energy Transfer Optimization

This repository contains code that was used for producing the results found in the relevant paper "Quantum Targeted Energy Transfer through Machine Learning Tools" (https://doi.org/10.1103/PhysRevE.107.065301).

## Cite

If any code from this repository is used for further scientific work on dimer or trimer configurations of non linear transfer, it should be cited using the citation file provided, using the `Cite this repository` side button.

## Installation

Our code can be used as long as these dependancies are installed. The specific packages are:

* [numpy](https://numpy.org/install/) >= 1.23
* [Tensorflow](https://www.tensorflow.org/install) >= 2.9
* python >= 3.8

We have also included an automated script to set up the dependencies in a virtual environment using the following commands:

```sh
git clone https://github.com/JAndronis/Quantum-Targeted-Energy-Transfer.git
cd Quantum-Targeted-Energy-Transfer
make install
```

It is important to note that a python version `>=3.8` with venv is needed for the installation. The `make` command will create a virtual environment directory called `.venv` which will contain all the necessary packages to run our code.

## Running QTET Application

We have implemented a recommended use-case of the package we developed into an application called `qtet.py` in the `src` directory. To run said application you need to do the following:

```sh
source .venv/bin/activate
qtet -p $data_path -c $constants_path -n $number_of_cpus --method $desired_method_of_optimization
```
