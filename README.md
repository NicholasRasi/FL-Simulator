# Federated Learning Simulator
A [Federated Learning](https://arxiv.org/abs/1602.05629) simulator with device availability, failures, computation,
network, local data and energy heterogeneity simulation.

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
Set the simulation parameters (described in the docs) with the ```config.py``` file and start the simulation with
```python simulate.py```.  The output of the simulation will be saved in the ```output``` folder by default.

#### Analyzer
The **Analyzer** can be started with ```python analyze.py -f <list of files to analyze> -d -p```. The files are
read from the ```output``` folder and the resulting plots along with a report file with all the graphs are saved into
the ```graphs``` folder by default. The flags allow to: ```-d```: print the simulation data, ```-p``` export the
plot.
