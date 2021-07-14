## FL phases
The model is trained during the **fit** phase and evaluated during the **eval** phase.
During the fit phase the model is trained locally, the resulting weights are aggregated and the global model is updated.
During the eval phase the model is evaluated locally, the resulting losses and accuracies are aggregated to compute how
well the model is performing.

## Federated Algorithm
The federated algorithm should implement two functions, called by the simulator during the
execution of the loop.

#### model_fit

#### model_eval