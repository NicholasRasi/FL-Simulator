from functools import reduce
import numpy as np
from typing import List, Tuple
from .aggregation_strategy import NDArrayList, AggregationStrategy


class Aggregate(AggregationStrategy):

    @staticmethod
    def aggregate(results: List[Tuple[int, NDArrayList]]) -> NDArrayList:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for num_examples, _ in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for num_examples, weights in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrayList = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    @staticmethod
    def average(results: List[Tuple[NDArrayList]]) -> NDArrayList:
        averaged_weights: NDArrayList = [
            reduce(np.add, layer_updates) / len(results)
            for layer_updates in zip(*results)
        ]
        return averaged_weights

    @staticmethod
    def weighted_loss_avg(samples_losses: List[Tuple[int, float]]) -> float:
        """Aggregate evaluation results obtained from multiple clients."""
        num_total_evaluation_examples = sum(
            [num_examples for num_examples, _ in samples_losses]
        )
        weighted_losses = [num_examples * loss for num_examples, loss in samples_losses]
        return sum(weighted_losses) / num_total_evaluation_examples

    @staticmethod
    def weighted_accuracies_avg(samples_accuracies: List[Tuple[int, float]]) -> float:
        """Aggregate evaluation results obtained from multiple clients."""
        num_total_evaluation_examples = sum(
            [num_examples for num_examples, _ in samples_accuracies]
        )

        weighted_accuracies = [num_examples * acc for num_examples, acc in samples_accuracies]
        return sum(weighted_accuracies) / num_total_evaluation_examples

    @staticmethod
    def federated_normalized_averaging(global_weights, results: List[Tuple[int, NDArrayList, int]]) -> NDArrayList:
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for num_examples, w, l in results])

        tau_eff = sum([(num_examples * local_iterations / num_examples_total) for
                                  num_examples, deltas, local_iterations in results])

        weighted_deltas = [
            [layer * tau_eff / local_it for layer in delta] for num_examples, delta, local_it in results
        ]

        local_weights = [np.subtract(global_weights, scaled_delta) for scaled_delta in weighted_deltas]

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * result[0] for layer in weights] for result, weights in zip(results, local_weights)
        ]

        # Compute average weights of each layer
        weights_prime: NDArrayList = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime




