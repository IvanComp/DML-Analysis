"""
Custom Flower Aggregation Strategy: FedTest
This module implements the FedTest aggregation algorithm for Federated Learning.
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
import numpy as np


class FedTest(fl.server.strategy.FedAvg):
    """
    FedTest Aggregation Strategy.
    
    Weights client updates based on their 'Consensus Score', which measures how 
    well a client's update aligns with the collective trajectory (average update).
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
        consensus_alpha: float = 0.5, # Controls strength of consensus weighting
    ) -> None:
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
        self.consensus_alpha = consensus_alpha

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[FedTest] --- Round {server_round} Aggregation Started ---")
        print(f"[FedTest] Aggregating updates from {len(results)} clients...")

        # 1. Flatten all client updates to compute consensus
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Calculate standard FedAvg as base for consensus
        total_examples = sum([num_examples for _, num_examples in weights_results])
        base_weights = [num_examples / total_examples for _, num_examples in weights_results]
        
        print(f"[FedTest] Total examples: {total_examples}. Computing Global Trajectory (base average)...")

        # Compute the "Global Trajectory" (standard average update)
        avg_update = []
        for i in range(len(weights_results[0][0])):
            layer_updates = [client_weights[i] * base_weights[j] for j, (client_weights, _) in enumerate(weights_results)]
            avg_update.append(np.sum(layer_updates, axis=0))
            
        flat_avg = np.concatenate([arr.flatten() for arr in avg_update])
        
        # 2. Compute Consensus Scores using Cosine Similarity
        print(f"[FedTest] Calculating Consensus Scores (Cosine Similarity with trajectory)...")
        consensus_scores = []
        for j, (client_weights, _) in enumerate(weights_results):
            flat_client = np.concatenate([arr.flatten() for arr in client_weights])
            
            # Cosine similarity between client update and average update
            norm_c = np.linalg.norm(flat_client)
            norm_a = np.linalg.norm(flat_avg)
            
            if norm_c > 0 and norm_a > 0:
                similarity = np.dot(flat_client, flat_avg) / (norm_c * norm_a)
            else:
                similarity = 0.0
                
            consensus_scores.append(similarity)
            print(f"  - Client {j}: Similarity = {similarity:.4f}")
            
        # 3. Apply Softmax to Consensus Scores to get weights
        print(f"[FedTest] Applying Softmax normalization (alpha={self.consensus_alpha})...")
        exp_scores = np.exp(np.array(consensus_scores) / max(self.consensus_alpha, 1e-6))
        consensus_weights = exp_scores / np.sum(exp_scores)
        
        print("[FedTest] Final Consensus Weights:")
        for j, w in enumerate(consensus_weights):
            print(f"  - Client {j}: Weight = {w:.4f} (Base Weight was {base_weights[j]:.4f})")
        
        # 4. Final Aggregation using Consensus Weights
        aggregated_ndarrays = []
        for i in range(len(weights_results[0][0])):
            layer_updates = [client_weights[i] * consensus_weights[j] for j, (client_weights, _) in enumerate(weights_results)]
            aggregated_ndarrays.append(np.sum(layer_updates, axis=0))
            
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            losses = [fit_res.metrics.get("loss", 0.0) for _, fit_res in results]
            metrics_aggregated["avg_loss"] = float(np.mean(losses))
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))

        print(f"[FedTest] Round {server_round} Aggregation Complete. Avg Loss: {metrics_aggregated.get('avg_loss', 0.0):.4f}")
        return aggregated_parameters, metrics_aggregated


# Export FedTest
__all__ = ['FedTest']

