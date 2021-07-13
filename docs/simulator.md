# Simulator
The simulator is in charge of:
- run the simulation loop until a stopping condition is met
- initialize the simulation Status
- export the simulation data

## Status
The status aggregates all the information about the simulation.
It provides complete information about the training, such as
the selected devices, the metrics of the model, the status of
the devices.

## Simulation loop
```
for given number of repetitions:
    init new simulation status
    init federated algorithm with the status
    run the fed. alg. until a stopping condition is met
    save run data
export simulation data
```