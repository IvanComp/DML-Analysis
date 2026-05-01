"""strategy_laa.py

Custom Flower aggregation strategy: FedLAA (Layer-wise Adaptive Aggregation)

Aggrega ogni layer separatamente con pesi ottimizzati per quel layer.
Ogni layer può avere una distribuzione di pesi diversa in base alla
varianza degli aggiornamenti per quel layer specifico.

Vincoli rispettati:
- No dati aggiuntivi lato server
- Solo pesi dei client (privacy preservata)
- Computazione aggiuntiva sul server
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def _layer_similarity(layer_updates: List[np.ndarray], mean_update: np.ndarray) -> np.ndarray:
    """Calcola la similarità di ogni client rispetto alla media per un singolo layer."""
    similarities = []
    mean_flat = mean_update.reshape(-1)
    mean_norm = np.linalg.norm(mean_flat)
    
    for update in layer_updates:
        update_flat = update.reshape(-1)
        update_norm = np.linalg.norm(update_flat)
        
        if mean_norm > 0 and update_norm > 0:
            sim = np.dot(update_flat, mean_flat) / (update_norm * mean_norm)
        else:
            sim = 0.0
        similarities.append(sim)
    
    return np.array(similarities, dtype=np.float64)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Applica softmax con temperatura."""
    x = x / max(temperature, 1e-12)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 0.0:
        return np.ones_like(x) / max(len(x), 1)
    return e / s


class FedLAA(fl.server.strategy.FedAvg):
    """
    Layer-wise Adaptive Aggregation (LAA) Strategy.
    
    Per ogni layer, calcola separatamente i pesi ottimali basandosi
    sulla similarità degli aggiornamenti di quel layer specifico.
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
        temperature: float = 0.5,
        layer_weight_blend: float = 0.5,  # Blend tra pesi layer-specific e globali
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
        self.temperature = float(temperature)
        self.layer_weight_blend = float(layer_weight_blend)
        self._prev_global: Optional[List[np.ndarray]] = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else None
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[FedLAA] === Round {server_round} ===")
        print(f"[FedLAA] Ricevuti aggiornamenti da {len(results)} client.")

        # 1. Estrai parametri e numero di esempi
        client_params: List[List[np.ndarray]] = []
        num_examples: List[int] = []
        
        for _, fit_res in results:
            client_params.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(int(fit_res.num_examples))

        n_clients = len(client_params)
        n_layers = len(client_params[0])
        
        # 2. Inizializza il modello globale precedente se necessario
        if self._prev_global is None:
            self._prev_global = [np.copy(a) for a in client_params[0]]

        # 3. Calcola i pesi base (numero di esempi)
        total_examples = float(sum(num_examples))
        base_weights = np.array([ne / total_examples for ne in num_examples], dtype=np.float64)
        
        # 4. Calcola gli aggiornamenti (delta) per ogni client
        deltas: List[List[np.ndarray]] = []
        for w_k in client_params:
            client_delta = [wk - wg for wk, wg in zip(w_k, self._prev_global)]
            deltas.append(client_delta)
        
        # 5. Per ogni layer, calcola i pesi adattivi
        aggregated_params: List[np.ndarray] = []
        layer_weights_log = []
        
        for layer_idx in range(n_layers):
            # Estrai gli aggiornamenti per questo layer da tutti i client
            layer_updates = [d[layer_idx] for d in deltas]
            
            # Calcola la media pesata per questo layer
            mean_update = np.zeros_like(layer_updates[0])
            for i, update in enumerate(layer_updates):
                mean_update += base_weights[i] * update
            
            # Calcola similarità di ogni client con la media per questo layer
            layer_similarities = _layer_similarity(layer_updates, mean_update)
            layer_similarities = (layer_similarities + 1.0) / 2.0  # Normalizza tra 0 e 1
            
            # Converti in pesi tramite softmax
            layer_adaptive_weights = _softmax(layer_similarities, temperature=self.temperature)
            
            # Blend con i pesi base
            blended_weights = (
                self.layer_weight_blend * layer_adaptive_weights + 
                (1 - self.layer_weight_blend) * base_weights
            )
            blended_weights = blended_weights / np.sum(blended_weights)
            
            layer_weights_log.append(blended_weights)
            
            # Aggrega questo layer
            layer_aggregated = np.zeros_like(client_params[0][layer_idx])
            for client_idx in range(n_clients):
                layer_aggregated += blended_weights[client_idx] * client_params[client_idx][layer_idx]
            
            aggregated_params.append(layer_aggregated)
        
        # Log dei pesi medi per client
        avg_weights = np.mean(layer_weights_log, axis=0)
        print("[FedLAA] Pesi medi di aggregazione (media su tutti i layer):")
        for i, w in enumerate(avg_weights):
            print(f"  - Client {i}: {w:.4f}")
        
        # 6. Aggiorna il modello globale precedente
        self._prev_global = [np.copy(a) for a in aggregated_params]
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        
        # Metriche
        weight_variance = np.mean([np.var(lw) for lw in layer_weights_log])
        metrics_aggregated: Dict[str, Scalar] = {
            "weight_variance": float(weight_variance),
        }
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))
        
        return aggregated_parameters, metrics_aggregated


__all__ = ["FedLAA"]
