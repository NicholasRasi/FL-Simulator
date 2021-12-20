# Data Strategies
The data strategy defines how local data is used.

**FELES** provides a default local data strategy:

| name | details |
|---|---|
| ``random`` | [Random](#random-optimizer) |

### Custom Data Strategy
The **Local Data Optimizer** selects the ```num_examples``` examples used for the local update

The **Local Data Optimizer** must implement the ***optimize*** method that given the number of current round,
the device index, the number of examples from the local update configuration and the available dataset, it returns
a subset data from the local available data, of size min(num_examples, available_examples):

```
def optimize(self, r: int, dev_index: int, num_examples: int, data) -> tuple
```

---

## Random Optimizer
The **Random Optimizer** randomly selects the available examples.
