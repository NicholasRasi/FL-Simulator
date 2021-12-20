# Update Strategies
The update strategy defines how work is performed locally, on the device.

**FELES** provides different update strategies in order to realize the federated algorithms:

| name | details |
|---|---|
| ``static`` | [Static](#static-optimizer) |
| ``bestrt`` | [Best RT Optimizer](#best-rt-optimizer) |
| ``uniform`` | [Uniform Optimizer](#uniform-optimizer) |
| ``equal_computation_time`` | [Equal Computation Time Optimizer](#equal-computation-time-optimizer) |

### Custom Data Strategy
The **Global Update Optimizer** computes ```epochs``` (E), ```batch_size``` (B) and ```num_examples``` (N) so the
amount of computation performed locally by each devices. The configuration can be different for every elected device.

The **Global Update Optimizer** must implement the ***optimize*** method that given a round and a device index
return the update configuration, as a dict:
```
def optimize(self, r: int, dev_index: int, phase: str) -> dict:
```

Example of returned configuration:
```
{"epochs": self.epochs, "batch_size": self.batch_size, "num_examples": self.num_examples}
```

---

## Static Optimizer
The **Static Optimizer** statically set E, B and N from the given parameters.

## Best RT Optimizer
The **Best RT Optimizer** sets B and N from the given parameters and E is assigned proportionally with respect to the 
maximum IPS value. The device with the highest IPS (which has the lowest computation) is assigned with the highest number 
of epochs which is the parameter in the configuration file.

## Uniform Optimizer
The **Uniform Optimizer** generates E, B, N with a uniform distribution in the range specified.

## Equal Computation Time Optimizer
The **Equal Computation Time Optimizer** generates E, B, N depending on a fixed amount of desired computation
time and the specific ips value for the device. The assumption is that ips is known for each device.
