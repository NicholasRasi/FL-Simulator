# Architecture
**FELES** is implements the orchestrator / workers architecture.

The functionalities are split into 2 different actors:

### Orchestrator
The orchestrator is in charge of:
- initialize the simulation *Orchestrator Status*
- start the Orchestrator API for the communication with workers
- start the Orchestrator which runs the simulation loop until a stopping condition is met.
  The stopping condition is defined in the configuration file as:
  ```num_rounds``` or ```acc``` or ```loss```. Once the simulation is finished, it saves and export the simulation data to a file

### Worker
The worker is in charge of:
- initialize the simulation *Worker Status*
- start the Worker API for the communication with the orchestrator
- start the Worker which consumes the fit and evaluation jobs received by the orchestrator and communicates the results


## Status
Since in a real world scenario **Worker** and **Orchestrator** do not share the same information, each role
is provided with a different *Status*.

### Orchestrator Status
The orchestrator status aggregates all the information about the simulation which are relevant for the orchestrator, 
such as the selected devices, the metrics of the model, the status of the devices (availability and failures).

### Worker Status
The worker status aggregates all the information about the simulation which are relevant for the worker, such as 
the federated algorithm, optimizer and metric to be used. Most of this information is sent by the orchestrator during
the initialization phase.


## Execution Loop

#### Orchestrator

```
for given number of repetitions:
    init new simulation status
    run the fed. alg. until a stopping condition is met
    save run data
export simulation data
```

#### Worker

```
while True:
    if orchestrator job queus is not empty:
      ask for next job
      handle next job
```


## Parallelization
Parallelization is realized through an orchestrator-workers architecture in which multiple workers 
can be run in parallel in order to speed up training simulation.

## Execution Flow

- init orchestrator
- init worker(s)
- worker sends to the orchestrator a request to participate
- orchestrator replies with all the necessary initialization information (such as the federated algorithm to be used)
- loop
  - orchestrator pushes a job in a queue every time a new computation needs to be performed
  - an available worker sends a message to the orchestrator to get the next job 
  - if the job queue is not empty, the orchestrator will reply with the next job information
  - as soon as a worker has completed a job, it will send to the orchestrator a message containing the results 
  of the job
  - orchestrator adds the received result in a queue of completed jobs
  - as soon as all the results are received, orchestrator performs aggregation