"""strategy_ama.py

Custom Flower aggregation strategy: FedAMA (Adaptive Momentum Aggregation)

Mantiene un momentum lato server degli aggiornamenti passati.
Usa la correlazione tra aggiornamenti attuali e momentum per pesare
i client, stabilizzando la convergenza.

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


def _flatten_layers(arrs: List[np.ndarray]) -> np.ndarray:
    """Appiattisce una lista di array numpy in un singolo vettore."""
    return np.concatenate([a.reshape(-1) for a in arrs], axis=0)


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Calcola la similarità coseno tra due vettori."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Applica softmax con temperatura."""
    x = x / max(temperature, 1e-12)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 0.0:
        return np.ones_like(x) / max(len(x), 1)
    return e / s


class FedAMA(fl.server.strategy.FedAvg):
    """
    Adaptive Momentum Aggregation (AMA) Strategy.
    
    Mantiene un momentum lato server che rappresenta la "direzione storica"
    dell'apprendimento. Pesa i client in base a quanto i loro aggiornamenti
    sono allineati con questa direzione.
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
        momentum_beta: float = 0.9,  # Fattore di momentum (0.9 = memoria lunga)
        temperature: float = 0.5,     # Temperatura per softmax
        momentum_weight: float = 0.3, # Quanto conta il momentum nel peso finale
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
        self.momentum_beta = float(momentum_beta)
        self.temperature = float(temperature)
        self.momentum_weight = float(momentum_weight)
        
        self._prev_global: Optional[List[np.ndarray]] = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else None
        )
        self._momentum: Optional[np.ndarray] = None  # Vettore momentum appiattito

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[FedAMA] === Round {server_round} ===")
        print(f"[FedAMA] Ricevuti aggiornamenti da {len(results)} client.")

        # 1. Estrai parametri e numero di esempi
        client_params: List[List[np.ndarray]] = []
        num_examples: List[int] = []
        
        for _, fit_res in results:
            client_params.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(int(fit_res.num_examples))

        n_clients = len(client_params)
        
        # 2. Inizializza il modello globale precedente se necessario
        if self._prev_global is None:
            self._prev_global = [np.copy(a) for a in client_params[0]]

        # 3. Calcola i pesi base (numero di esempi)
        total_examples = float(sum(num_examples))
        base_weights = np.array([ne / total_examples for ne in num_examples], dtype=np.float64)
        
        # 4. Calcola gli aggiornamenti (delta) per ogni client
        deltas_flat: List[np.ndarray] = []
        for w_k in client_params:
            delta = _flatten_layers([wk - wg for wk, wg in zip(w_k, self._prev_global)])
            deltas_flat.append(delta)
        
        # 5. Calcola la media pesata degli aggiornamenti
        mean_delta = np.zeros_like(deltas_flat[0])
        for i, delta in enumerate(deltas_flat):
            mean_delta += base_weights[i] * delta
        
        # 6. Aggiorna il momentum
        if self._momentum is None:
            self._momentum = mean_delta.copy()
            print("[FedAMA] Inizializzato momentum con il primo aggiornamento medio.")
        else:
            self._momentum = (
                self.momentum_beta * self._momentum + 
                (1 - self.momentum_beta) * mean_delta
            )
            print(f"[FedAMA] Momentum aggiornato (beta={self.momentum_beta}).")
        
        # 7. Calcola la correlazione di ogni client con il momentum
        momentum_correlations = np.array([
            _cosine_similarity(delta, self._momentum) for delta in deltas_flat
        ], dtype=np.float64)
        
        # Normalizza tra 0 e 1
        momentum_correlations = (momentum_correlations + 1.0) / 2.0
        
        print("[FedAMA] Correlazione con il momentum storico:")
        for i, corr in enumerate(momentum_correlations):
            print(f"  - Client {i}: {corr:.4f}")
        
        # 8. Calcola la similarità con la media corrente
        current_similarities = np.array([
            _cosine_similarity(delta, mean_delta) for delta in deltas_flat
        ], dtype=np.float64)
        current_similarities = (current_similarities + 1.0) / 2.0
        
        # 9. Combina i due segnali
        combined_scores = (
            (1 - self.momentum_weight) * current_similarities +
            self.momentum_weight * momentum_correlations
        )
        
        # 10. Converti in pesi
        adaptive_weights = _softmax(combined_scores, temperature=self.temperature)
        
        # 11. Combina con pesi base
        final_weights = adaptive_weights * base_weights
        final_weights = final_weights / np.sum(final_weights)
        
        print("[FedAMA] Pesi di aggregazione finali:")
        for i, w in enumerate(final_weights):
            print(f"  - Client {i}: {w:.4f}")
        
        # 12. Aggrega i parametri
        aggregated_params: List[np.ndarray] = []
        for layer_idx in range(len(client_params[0])):
            layer_sum = np.zeros_like(client_params[0][layer_idx])
            for client_idx, params in enumerate(client_params):
                layer_sum += final_weights[client_idx] * params[layer_idx]
            aggregated_params.append(layer_sum)
        
        # 13. Aggiorna il modello globale precedente
        self._prev_global = [np.copy(a) for a in aggregated_params]
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        
        # Metriche
        momentum_norm = float(np.linalg.norm(self._momentum))
        metrics_aggregated: Dict[str, Scalar] = {
            "momentum_norm": momentum_norm,
            "avg_momentum_correlation": float(np.mean(momentum_correlations)),
        }
        
        return aggregated_parameters, metrics_aggregated


__all__ = ["FedAMA"]
