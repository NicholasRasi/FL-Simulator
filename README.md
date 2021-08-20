# Federated Learning Simulator
**FL-Simulator** is a highly configurable [Federated Learning](https://arxiv.org/abs/1602.05629) simulator designed for
Federated Learning experiments.

## FL characteristics
- the devices taking part to the FL network may be available only for a small amount of time. They may fail during the
  execution of a training job
- the computation capability made available by each device may be (very) different
- the available network speed on each device may be different and variable
- the training is performed with non-IID data available on devices
- the devices may be battery powered, or have a fixed amount of total available energy

## Goals
- the simulator aims to facilitate the development, test, comparison and analysis of custom algorithms (e.g. FedAvg)
- the execution requires only a single machine and not a real and expensive FL network
- the results do not depend on the hardware where the experiment was executed
- the experiment is easily reproducible
- optimizers (logic blocks defining the strategy applied at different stages of the FL protocol, e.g. client
  selection) can be plugged in the simulator
- the simulator allows to analyze, plot and visualize the resulting data

## Getting Started
### Setup
```
virtualenv env
pip install -r requirements.txt
```

### Start a Simulation
1. the simulation parameters are defined in the ```config.py``` file
2. the simulation can be started with ```python simulate.py```
3. the output of the simulation will be saved in the ```output``` folder by default

### Analyze the Simulation data
1. the *Analyzer* can be started with ```python analyze.py -f <list of files> -d -p```
2. the files are read from the ```output``` folder. The flags allow to: ```-d``` print the simulation data, ```-p```
   export the plots

## Documentation
More information about parameters, optimizers and the analyzer are available [here](./docs)
