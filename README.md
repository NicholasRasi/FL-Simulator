<p align="center">
  <img src="docs/imgs/logo_transparent.png" height="300">
</p>

**FELES** is a highly configurable FEderated LEarning Simulator designed for
Federated Learning experiments.


## Federated Learning
[Federated Learning](https://arxiv.org/abs/1602.05629) is a new machine learning approach that enables different devices
to collaboratively learn a shared prediction model while keeping all the training data on the device.

### FL Characteristics
FL has some unique characteristics:
- **non-IID data**: the training is performed with local data available on devices. Thus data are not
  [independent and identically distributed data (IID)](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
- **heterogeneous network**: the devices taking part to the network may be very different in terms of:
  - computation capability
  - network bandwidth and speed
  - energy constraints
- **variable network**: the property of the devices taking part to the network may change over time, e.g. the network speed
  available on a device may vary depending on its position
- **devices availability**: the devices taking part to the network may be available only for a small amount of time
- **failures**: the devices may fail during the execution of a training or evaluation job due to different reasons,
  e.g. unavailable network or available resources
- **high number of devices**: a high number of devices may be used to train a model

### FL Experiments Limits
Empirical evaluations of FL approaches may be difficult to be replicated and compared:
- FL network deployment may be complex and unfeasible
- experiment cost may be very high due to high number of devices
- the experiment may be difficult to be replicated
- lack of standardized benchmarks
- lack of standardized frameworks

## FELES Goals
**FELES** provides the following features:
- **orchestrator / worker architecture**: the experiments can be executed using only a single machine avoiding the need for
  an expensive FL network. Multiple workers can be used to scale the simulation
- **hardware independent**: the results do not depend on the hardware where the experiment was executed
- **reproducibility**: the experiments are easily reproducible with a configurable environment
- **development and comparison**: it facilitates the development, test, comparison and analysis of custom algorithms with
  popular ones (e.g. FedAvg, FedProx)
- **customizable**: it is easily customizable allowing to plug-in optimizers, logic blocks defining the strategy applied
  at different stages of the FL protocol, e.g. client selection)
- **results analyzer**: the simulator provide an analyzer for plotting and visualizing the resulting data

## Getting Started
### Setup
```
virtualenv env
pip install -r requirements.txt
```

### Start a Simulation
1. the simulation parameters are defined in the ```config.py``` file.
   A complete explanation of the available parameters is given in [Parameters](docs/parameters.md)
2. open a shell and start the orchestrator with ```python simulate_orchestrator.py```
2. open another shell and start the worker with ```python simulate_worker.py```
3. the output of the simulation will be saved in the ```output``` folder by default

### Analyze the Simulation data
The *Analyzer* can be started with ```python analyze.py -f <list of files> -d -p```

- the files after ```-f``` are read from the ```output``` folder 
- ```-d``` print the simulation data
- ```-p``` export the plots

## Documentation
More information about parameters, optimizers and the analyzer are available [here](./docs)
