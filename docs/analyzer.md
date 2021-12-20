# Analyzer
The analyzer allows to:
- plot graphs
- export a report file (Markdown) with all the graphs
- get information about the execution on the console and in a html file.

### Available graphs
When the simulation is repeated multiple times, the standard deviation is plotted with a vertical line, for every round.

The available graphs are:

| name | details | example |
|---|---|---|
| ``plot_accuracy(phase, color)`` | it shows the aggregated accuracy | ![](imgs/example_graphs/agg_accuracy_eval.png) |
| ``plot_loss(phase, color)`` | it shows the aggregated accuracy | ![](imgs/example_graphs/agg_loss_eval.png) |
| ``plot_computation_time(phase, color)`` | it shows the computation time | ![](imgs/example_graphs/rt_computation_fit.png) |
| ``plot_communication_time(phase, color)`` | it shows the communication time | ![](imgs/example_graphs/rt_communication_fit.png)   |
| ``plot_total_time(phase, color)`` | it shows the total round time | ![](imgs/example_graphs/rt_total_fit.png)   |
| ``plot_resources_consumption(phase, color)`` | it shows the resource consumption | ![](imgs/example_graphs/consumption_resources_fit.png)   |
| ``plot_network_consumption(phase, color)`` | it shows the network consumption | ![](imgs/example_graphs/consumption_network_fit.png)   |
| ``plot_energy_consumption(phase, color)`` | it shows the energy consumption | ![](imgs/example_graphs/consumption_energy_fit.png) |
| ``plot_available_devices(color)`` | it shows the available devices | ![](imgs/example_graphs/devices_available.png) |
| ``plot_selected_devices(phase, color)`` | it shows the selected devices | ![](imgs/example_graphs/devices_selected_fit.png) |
| ``plot_available_failed_devices(phase, color)`` | it shows the available and failed devices | ![](imgs/example_graphs/devices_available_failed.png) |
| ``plot_selected_successful_devices(phase, color)`` | it shows the selected and successful devices | ![](imgs/example_graphs/devices_selected_successful_fit.png) |
| ``plot_epochs_config(phase, color)`` | it shows the configured epochs | ![](imgs/example_graphs/config_epochs_fedavg_0_fit.png) |
| ``plot_batch_size_config(phase, color)`` | it shows the configured batch size | ![](imgs/example_graphs/config_batch_size_fedavg_0_fit.png) |
| ``plot_num_examples_config(phase, color)`` | it shows the configured num examples | ![](imgs/example_graphs/config_num_examples_fedavg_0_fit.png) |
| ``plot_matrix_devices(color)`` | it shows the not available, available and selected devices | ![](imgs/example_graphs/matrix_devices_fedavg_0_fit.png) |
| ``plot_devices_bar_availability(phase, color)`` | it shows the %of availability of the devices | ![](imgs/example_graphs/devs_bar_availability_fedavg_0_fit.png) |
| ``plot_devices_bar_failures(phase, color)`` | it shows the % of failures of the devices | ![](imgs/example_graphs/devs_bar_failures_fedavg_0_fit.png) |
| ``plot_devices_bar_selected(phase, color)`` | it shows the selected % of the devices | ![](imgs/example_graphs/devs_bar_selected_fedavg_0_fit.png) |
| ``plot_devices_data_size(color)`` | it shows the local data size and density | ![](imgs/example_graphs/devs_data_size_fedavg_0.png) |
| ``plot_devices_ips(color)`` | it shows the IPS of the devices and density | ![](imgs/example_graphs/devs_ips_fedavg_0.png) |
| ``plot_devices_available_energy(color)`` | it shows the available energy of the devices and density | ![](imgs/example_graphs/devs_available_energy_fedavg_0.png) |
| ``plot_devices_network_speed(color)`` | it shows the network speed of the devices and density | ![](imgs/example_graphs/devs_network_speed_fedavg_0.png) |
| ``plot_devices_data_distribution()`` | it shows the data distribution of classes on the devices | ![](imgs/example_graphs/devs_local_data_fedavg_0.png) |
  

### Available print functions

- ```print_availability()```

- ```print_failures()```
  
- ```print_ips()```
    
- ```print_energy()```
    
- ```print_net_speed()```
    
- ```print_local_data_size()```
    
- ```print_model_params()```
    
- ```print_selection(phase)```
     
- ```print_total_time(phase)```
     
- ```print_resources_consumption(phase)```
     
- ```print_energy_consumption(phase)```
     
- ```print_network_consumption(phase)```
     
- ```print_metric(phase, round)```
     
- ```print_loss(phase, round)```