## Simulation Parameters
The simulation parameters defines how the simulation is performed and which are the algorithms used
to perform the FL phases.

- **repetitions**:
    - type: int
    - desc: the number of repetitions for each run
- **output_folder**:
    - type: str
    - desc: folder where to save the simulation output data
- **output_file**:
    - type: str
    - desc: file where to save the simulation output data
- **model_name**:
    - type: str
    - desc: the model used for the simulation
    - available models are:
        - ```mnist```
        - ```fashion_mnist```
        - ```cifar10```
        - ```imdb_reviews```
- **num_rounds**:
    - type: int
    - desc: the total number of rounds to execute of the FL algorithm
- **stop_conds**:
  - **accuracy**:
    - type: float
    - desc: the accuracy of the model. When this value is reached the simulation
      is stopped
  - **loss**:
    - type: float
    - desc: the loss of the model. When this value is reached the simulation
      is stopped
- **initializer**
  - type: str
  - desc: set the initialization policy
  - available initializers are:
    - ```default```
- **seed**:
    - type: int
    - desc: random seed can be fixed for reproducibility
- **tf_verbosity**:
    - type: int
    - desc: set the TensorFlow verbosity
    - available values:
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch
- **num_processes**:
    - type: int
    - desc: number of concurrent processes for the model computations
    
### Algorithms
These values express the algorithms that are used for the simulation
during the fit and evaluation phases.

- **aggregation**:
    - type: str
    - desc: the algorithm used for the local update aggregation
    - available values:
        - ```fedavg```
- **selection**:
    - type: str
    - desc: the algorithm used for the clients selection
    - defined for: fit, eval
    - available values:
        - ```random```
        - ```best_ips```
        - ```best_net_speed```
        - ```best_energy```
- **update**:
    - type: str
    - desc: the algorithm used for the global update optimizer
    - defined for: fit, eval
    - available values:
        - ```static```
- **data**:
    - type: str
    - desc: the algorithm used for the local data optimizer
    - defined for: fit, eval    
    - available values:
        - ```random```
- **optimizer**:
    - type: str
    - desc: optimizer used by the local update
    - available values:
        - ```sgd```
        - ```adam```
        - all optimizers provided by TensorFlow
- **params**:
    - type: list
    - desc: a list of parameters used by the algorithms.
      See **Algorithm Parameters** for more information
      

### Algorithm Parameters
These parameters define how the algorithms are executed and how updates are performed,
i.e., the amount of computation performed  by each device at each round.
For example, these parameters can be used to set the number oflocal iterations
(epochs * num_examples / batch_size) used by the standard FedAvg algorithm.

- **k**:
    - type: float
    - desc: fraction of clients used for the computation
    - available values: (0, 1]
    - defined for: fit, eval
- **epochs**:
    - type: int
    - desc: number of epochs executed for each round
- **batch_size**:
    - type: int
    - desc: batch size used for each round
    - defined for: fit, eval
- **num_examples**:
    - type: int
    - desc: number of examples used for each round
    - defined for: fit, eval


### Devices Parameters
In FL devices can join and leave the network at every instant of time. ```num``` defines the total number of
devices that can take part to the network concurrently. A local update of the model on the device can fail,
e.g., the process is stopped by the OS. The availability and failures are modeled with a binomial distribution
with probability ```p_available``` and ```p_fail``` respectively.

- **num**:
    - type: int
    - desc: total number of devices
- **p_available**:
    - type: float
    - desc: probability a device is available for a round
    - available values: [0, 1]
- **p_fail**:
    - type: float
    - desc: probability a device fails during a round
    - available values: [0, 1]

### Computation
Devices in FL are heterogeneous (e.g. smartphones with different CPUs and memory).
The computation capabilities of each device is modeled through the number of iterations the device
is capable of running per second (IPS).
Using the ```default``` initializer, IPS is taken from a uniform distribution
between [ips_mean - ips_var, ips_mean + ips_var].  IPS is assumed to be fixed for each device for each round.
The ```ips_var``` defines the heterogeneity of the available devices.
 
- **ips_mean**:
    - type: int
    - desc: mean number of computed iterations/second per device (among devices) [iter/s]
- **ips_var**:
    - type: int
    - desc: ips variance

### Energy
Devices in FL are generally powered by batteries. The available energy capacity is different among devices.
Using the ```default``` initializer, the energy, expressed in mWh, available at each device is taken from a
uniform distribution between [energy_mean - energy_var, energy_mean + energy_var].
It changes for every device for every round because the devices could be recharged or used for other activities.

- **avail_mean**:
    - type: int
    - desc: mean energy capacity available at each device [mWh]
- **avail_var**:
    - type: int
    - desc: energy variance
    
The amount of energy consumed by computation or network is computed with the following parameters
(assumed to be equal for all the devices): 
- **pow_comp_s**:
    - type: int
    - desc: power consumption for 1 second of computation [mW/s]
- **pow_net_s**:
    - type: int
    - desc: power consumption for 1 second of network used [mW/s]

### Network
Devices in FL are connected to different networks (e.g. WiFI, 3G, 4G, 5G), that provides different capabilities.
Using the ```default``` initializer, the network speed is taken from a uniform distribution
between [netspeed_mean - netspeed_var, netspeed_mean + netspeed_var].
It changes for every device for every round because devices can move at different location and be connected
to different networks.

- **speed_mean**:
    - type: int
    - desc: mean network speed available at each device [params/s]
- **speed_var**:
    - type: int
    - desc: network speed variance

### Data
The size of the data locally available is generally different among devices. While some devices could have a large
dataset other may have a smaller one. Using the ```default``` initializer,, the local data size is modeled with
a uniform distribution  between [local_data_mean - local_data_var, local_data_mean + local_data_var].
The data set size is assumed to be fixed for each device for each round.

- **num_examples_mean**:
    - type: int
    - desc: mean number of examples available at each device [examples]
- **num_examples_var**:
    - type: int
    - desc: number of examples variance
- **non_iid_partitions**:
    - type: float
    - desc: if set to 0 the non-iid partitions are not created (and examples will be extracted from the dataset randomly),
    otherwise the partitions will be created as described in [Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification](https://arxiv.org/pdf/1909.06335.pdf). Small values (e.g., 0.1) will create high partitioning while
    bigger values (e.g., 100) will act as random sampling. This parameter indirectly also controls the number of available examples,
    so for some small values the **local_data_mean** and **local_data_var** will be discarded.
