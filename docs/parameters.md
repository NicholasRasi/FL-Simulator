# Parameters
**FELES** reads the configuration from the ``config.yml`` file.

### Import Configuration
It is possible to import another configuration with the following
key-value pair: ``import: <other.yml file>``.
If the same parameter is defined multiple times,
the priority is given to the latest definition.

## Structure
The configuration file structure is organized as follows:

| name | type | desc |
|---|---|---|
| ``simulation`` | object | the simulation parameters, see [Simulation](#simulation) |
| ``algorithms`` | object | the algorithms parameters, see [Algorithms](#algorithms) |
| ``devices`` | object | the devices parameters, see [Devices](#devices) |
| ``computation`` | object | the computation parameters, see [Computation](#computation) |
| ``energy`` | object | the energy parameters, see [Energy](#energy) |
| ``network`` | object | the network parameters, see [Network](#network) |
| ``data`` | object | the data parameters, see [Data](#data) |

Example:
```yaml
simulation:
  ...
algorithms:
  ...
devices:
  ...
computation:
  ...
energy:
  ...
network:
  ...
data:
  ...
```

### Simulation
The simulation parameters defines how the simulation is performed and which are the algorithms used
to perform the FL phases.

| name | type | desc |
|---|---|---|
| ``repetitions`` | int | the number of repetitions for each run |
| ``output_folder`` | str | the folder where to save the simulation output data |
| ``output_file`` | str | the file where to save the simulation output data |
| ``model_name`` | str | the model used for the simulation. See [Datasets and Models](./datasets_models.md) for possible values |
| ``metric`` | str | metric to be evaluated by the model during training and testing, see [tf.keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) |
| ``num_rounds`` | int | the total number of rounds to execute of the FL algorithm |
| ``stop_conds`` | object | the execution will stop when the stopping conditions are met, see [Stopping Conditions](#stopping-conditions) |
| ``initializer`` | str | set the initialization policy. Available values: ``default`` |
| ``seed`` | int | the random seed, it can be fixed for reproducibility |
| ``tf_verbosity`` | int | set the TensorFlow verbosity. Available values are: ``0 = silent, 1 = progress bar, 2 = one line per epoch`` |
| ``num_processes`` | int | number of concurrent processes for the model computations |

Example:
```yaml
simulation:
  repetitions: 4
  output_folder: "output"
  output_file: "results.json"
  model_name: "mnist"
  metric: "accuracy"
  num_rounds: 15
  stop_conds:
    metric: 1
    loss: 0
  initializer: "default"
  seed: 123
  tf_verbosity: 0
  num_processes: 1
```

#### Stopping Conditions
The execution will stop when the stopping conditions are met.

| name | type | desc |
|---|---|---|
| ``metric`` | float | the simulation is stopped when the metric value is reached  |
| ``loss`` | float | the simulation is stopped when the loss value is reached  |


### Algorithms
These values express the algorithms that are used for the simulation
during the fit and evaluation phases.

| name | type | desc |
|---|---|---|
| ``federated_algorithm`` | str | the algorithm used for the local update aggregation. See [Federated Algorithms](./federated_algorithms.md) for possible values |
| ``fit`` | object | it describes how the fit phase is performed, see [Fit](#fit) |
| ``eval`` | object | it describes how the eval phase is performed, see [Eval](#eval) |
| ``optimizer`` | str | optimizer used by the local update. Available values: ``sgd``, ``adam`` and all the optimizers provided by TensorFlow |

Example:
```yaml
algorithms:
  federated_algorithm: "fedavg"
  fit:
    aggregation: "fedavg"
    selection: "best_time_expected"
    update: "static"
    data: "random"
    params:
      k: 0.5
      epochs: 2
      batch_size: 16
      num_examples: 100
  eval:
    selection: "random"
    update: "static"
    data: "random"
    params:
      k: 0.5
      epochs: 2
      batch_size: 16
      num_examples: 100
  optimizer: "sgd"
```

#### Fit
These parameters describe how to perform the fit phase (i.e. algorithms to be used and how they need to be configured)

| name | type | desc |
|---|---|---|
| ``aggregation`` | str | the algorithm used for the local update aggregation. See [Aggregation Strategies](./aggregation_strategies.md) for possible values |
| ``selection`` | str | the algorithm used for the clients selection, see [Selection Strategies](./selection_strategies.md) for possible values |
| ``update`` | str | the algorithm used for the global update optimizer, see [Update Strategies](./update_strategies.md) for possible values |
| ``data`` | str | the algorithm used for the local data optimizer, see [Data Strategies](./data_strategies.md) for possible values |
| ``params`` | object | parameters used by the algorithms, see [Params](#params) |

##### Params
These parameters define how the algorithms are executed and how updates are performed,
i.e., the amount of computation performed  by each device at each round.
For example, these parameters can be used to set the number of local iterations
(epochs * num_examples / batch_size) used by the standard FedAvg algorithm.

| name | type | desc |
|---|---|---|
| ``k`` | float | fraction of clients used for the computation, ``(0, 1]``  |
| ``epochs`` | int | number of epochs executed for each round  |
| ``batch_size`` | int | batch size used for each round  |
| ``num_examples`` | int | number of examples used for each round  |

#### Eval
The parameters are equal to the [Fit](#fit) phase.


### Devices
In FL devices can join and leave the network at every instant of time. The availability and failures are
modeled with a binomial distribution with probability ```p_available``` and ```p_fail``` respectively.

| name | type | desc |
|---|---|---|
| ``num`` | int | the total number of devices that can take part to the network concurrently  |
| ``p_available`` | float | the probability a device is available for a round, ``(0, 1]`` |
| ``p_fail`` | float | batch size used for each round ``(0, 1]`` |
| ``adversary_num`` | float | TODO |

Example:
```yaml
devices:
  num: 50
  p_available: 0.8
  p_fail: 0.1
  adversary_num: 0
```


### Computation
Devices in FL are heterogeneous (e.g. smartphones with different CPUs and memory).
The computation capabilities of each device is modeled through the number of iterations the device
is capable of running per second (IPS).
Using the ```default``` initializer, IPS is taken from a uniform distribution
between ``[ips_mean - ips_var, ips_mean + ips_var]``.  IPS is assumed to be fixed for each device for each round.
The ```ips_var``` defines the heterogeneity of the available devices.
 
| name | type | desc |
|---|---|---|
| ``ips_mean`` | int | mean number of computed iterations/second per device (among devices) [iter/s]  |
| ``ips_var`` | int | ips variance |

Example:
```yaml
computation:
  ips_mean: 100
  ips_var: 50
```

### Energy
In FL the available energy capacity is different among devices.
Using the ```default``` initializer, the energy, expressed in mWh, available at each device is taken from a
uniform distribution between ``[energy_mean - energy_var, energy_mean + energy_var]``.
It changes for every device for every round because the devices could be recharged or used for other activities.
The amount of energy consumed by computation or network are assumed to be equal for all the devices.

| name | type | desc |
|---|---|---|
| ``avail_mean`` | int | mean energy capacity available at each device [mWh]  |
| ``avail_var`` | int | energy variance |
| ``pow_comp_s`` | int | power consumption for 1 second of computation [mW/s] |
| ``pow_net_s`` | int | power consumption for 1 second of network used [mW/s] |

Example:
```yaml
energy:
  avail_mean: 25000
  avail_var: 20000
  pow_comp_s: 100
  pow_net_s: 200
```

### Network
Devices in FL are connected to different networks (e.g. WiFI, 3G, 4G, 5G), that provides different capabilities.
Using the ```default``` initializer, the network speed is taken from a uniform distribution
between ``[netspeed_mean - netspeed_var, netspeed_mean + netspeed_var]``.
It changes for every device for every round because devices can move at different location and be connected
to different networks.

| name | type | desc |
|---|---|---|
| ``speed_mean`` | int | mean network speed available at each device [params/s]  |
| ``speed_var`` | int | network speed variance |

Example:
```yaml
network:
  speed_mean: 100000
  speed_var: 70000
```

### Data
The size of the data locally available is generally different among devices. While some devices could have a large
dataset other may have a smaller one. Using the ```default``` initializer,, the local data size is modeled with
a uniform distribution  between ```[local_data_mean - local_data_var, local_data_mean + local_data_var]```.
The data set size is assumed to be fixed for each device for each round.

| name | type | desc |
|---|---|---|
| ``num_examples_mean`` | int | mean number of examples available at each device [examples]  |
| ``num_examples_var`` | int | number of examples variance |
| ``mislabelling_percentage`` | float | the percentage on mislabelled data, ``[0, 1)`` |
| ``non_iid_partitions`` | float | see [Non-IID Partitions Generation](#non-iid-partitions-generation) for details |

Example:
```yaml
data:
  num_examples_mean: 1500
  num_examples_var: 0
  non_iid_partitions: 0
  mislabelling_percentage: 0
```

#### Non-IID Partitions Generation
**non_iid_partitions**: if set to 0 the non-iid partitions are not created
(and examples will be extracted from the dataset randomly),
otherwise the partitions will be created as described in [Measuring the Effects of Non-Identical Data Distribution for
Federated Visual Classification](https://arxiv.org/pdf/1909.06335.pdf).
Small values (e.g., 0.1) will create high partitioning while
bigger values (e.g., 100) will act as random sampling. 
This parameter indirectly also controls the number of available examples,
so for some small values the **local_data_mean** and **local_data_var** will be discarded.
