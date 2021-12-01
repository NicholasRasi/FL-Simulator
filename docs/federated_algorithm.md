## FL phases
The model is trained during the **fit** phase and evaluated during the **eval** phase.
During the fit phase the model is trained locally, the resulting weights are aggregated and the global model is updated.
During the eval phase the model is evaluated locally, the resulting losses and accuracies are aggregated to compute how
well the model is performing.

## Federated Algorithm
The simulator provides a number of federated algorithms. The basic federated algorithm is FedAvg but each federated algorithm 
implements a personalized version for the worker and the orchestrator by overriding functions. 
Each federated worker can override functions:

- #### handle_fit_job

- #### handle_eval_job

Each federated orchestrator can override function:

- #### model_fit

- #### model_eval

- ####select_devs

- ####put_client_job_fit

- #### put_client_job_eval

- #### get_fit_results

- #### get_eval_results

## Available federated algorithms

The algorithms implemented are the followings:

### FedAvg
- Reference: https://arxiv.org/abs/1602.05629

### FedProx
- Reference: https://arxiv.org/abs/1812.06127

### FedNova
- Reference: https://arxiv.org/abs/2007.07481

### SCAFFOLD
- Reference: https://arxiv.org/abs/1910.06378

### FedDyn
- Reference: https://arxiv.org/abs/2111.04263
