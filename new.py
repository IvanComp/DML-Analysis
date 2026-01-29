"""
Custom Flower Aggregation Strategy
This module implements a new aggregation algorithm for Federated Learning.
The strategy can be imported and used by flower_baseline.py.
"""

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Union, Optional, Dict
from logging import WARNING
import numpy as np


class NewAggregationStrategy(fl.server.strategy.FedAvg):
    """
    Custom Aggregation Strategy for Federated Learning.
    
    This strategy extends FedAvg and implements a custom aggregation logic.
    You can modify the aggregate_fit method to implement your own algorithm.
    
    Current implementation: Custom weighted aggregation based on client performance.
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    ) -> None:
        """Initialize the custom strategy."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        # Custom attributes for your algorithm
        self.round_number = 0
        self.client_history = {}  # Track client performance over rounds
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Custom aggregation logic for model updates.
        
        This is where you implement your new aggregation algorithm.
        Current implementation: weighted aggregation based on training loss.
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed clients
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}
        
        self.round_number = server_round
        
        # --- CUSTOM AGGREGATION LOGIC STARTS HERE ---
        
        # Extract weights and metrics from results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        
        # Calculate custom weights based on client loss
        # Lower loss = higher weight (better performing clients get more influence)
        custom_weights = []
        for _, num_examples, metrics in weights_results:
            client_loss = metrics.get("loss", 1.0)
            
            # Custom weighting: inverse of loss, scaled by number of examples
            # You can modify this formula to implement your own algorithm
            weight = num_examples / (client_loss + 1e-6)
            custom_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(custom_weights)
        normalized_weights = [w / total_weight for w in custom_weights]
        
        # Aggregate parameters using custom weights
        aggregated_ndarrays = []
        for i in range(len(weights_results[0][0])):  # For each layer
            layer_updates = []
            for j, (client_weights, _, _) in enumerate(weights_results):
                weighted_layer = client_weights[i] * normalized_weights[j]
                layer_updates.append(weighted_layer)
            
            # Sum all weighted updates for this layer
            aggregated_layer = np.sum(layer_updates, axis=0)
            aggregated_ndarrays.append(aggregated_layer)
        
        # Convert back to Parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # --- CUSTOM AGGREGATION LOGIC ENDS HERE ---
        
        # Aggregate metrics (optional)
        metrics_aggregated = {}
        if results:
            losses = [fit_res.metrics.get("loss", 0.0) for _, fit_res in results]
            metrics_aggregated["avg_loss"] = float(np.mean(losses))
        
        return aggregated_parameters, metrics_aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        You can also customize this if needed for your algorithm.
        """
        # Use default FedAvg aggregation for evaluation
        return super().aggregate_evaluate(server_round, results, failures)


# Alternative: You can also implement a completely custom strategy from scratch
class NewCustomStrategy(fl.server.strategy.Strategy):
    """
    Completely custom strategy implementation (alternative approach).
    Use this if you need full control over the entire strategy logic.
    """
    
    def __init__(self, initial_parameters: Optional[Parameters] = None):
        self.initial_parameters = initial_parameters
    
    def initialize_parameters(self, client_manager):
        return self.initial_parameters
    
    def configure_fit(self, server_round, parameters, client_manager):
        # Implement client selection and configuration for training
        pass
    
    def aggregate_fit(self, server_round, results, failures):
        # Implement custom aggregation
        pass
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        # Implement client selection and configuration for evaluation
        pass
    
    def aggregate_evaluate(self, server_round, results, failures):
        # Implement evaluation aggregation
        pass
    
    def evaluate(self, server_round, parameters):
        # Optional: server-side evaluation
        return None


# Export the strategy to be used in flower_baseline.py
__all__ = ['NewAggregationStrategy', 'NewCustomStrategy']
