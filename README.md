# Federated Learning Simulator
A [Federated Learning](https://arxiv.org/abs/1602.05629) simulator with device availability, failures, computation, network, local data and energy heterogeneity simulation.

The goal of this simulator is to execute an FL algorithm, such as FedAvg, in a real FL scenario, i.e., many
heterogeneous devices, with different computation, network, energy capabilities and local data availability.
The simulator aims to allow the developer to test and analyze custom optimizers.
Optimizers are logic blocks that can be plugged in the simulator and can control the
FL algorithm during its execution.

The simulator provides an **Analyzer** component that allows to plot and visualize the simulation data.

### Setup
```
virtualenv env
pip install -r requirements.txt
```

#### Simulator
Set the simulation parameters (described below) in the ```config.py``` file or with environment variables 
and start the simulation, e.g., ```export FL_OUT_FILE="base.json" && export FL_SEL_F="random" && python simulate.py```.
The output of the simulation will be saved in the ```output``` folder by default.

#### Analyzer
The **Analyzer** can be started with ```python analyze.py -f <list of files to analyze>```. The files are read from the 
```output``` folder and the resulting plots are saved to the ```graphs``` folder by default.

## FL phases
The model is trained during the **fit** phase and evaluated during the **eval** phase.
During the fit phase the model is trained locally, the resulting weights are aggregated and the global model is updated.
During the eval phase the model is evaluated locally, the resulting losses and accuracies are aggregated to compute how
well the model is performing.

