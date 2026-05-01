"""newAux.py

Custom Flower aggregation strategy: FedGSW (Gradient Similarity Weighting)

Pesa i client in base alla similarità coseno tra il loro aggiornamento
e la media degli aggiornamenti. Client più allineati con il consenso
ricevono un peso maggiore nell'aggregazione.

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
    """Applica softmax con temperatura per convertire score in pesi."""
    x = x / max(temperature, 1e-12)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    if s <= 0.0:
        return np.ones_like(x) / max(len(x), 1)
    return e / s


class FedGSW(fl.server.strategy.FedAvg):
    """
    Gradient Similarity Weighting (GSW) Strategy.
    
    Pesa i client proporzionalmente alla similarità coseno tra il loro
    aggiornamento e la media pesata degli aggiornamenti.
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
        temperature: float = 0.5,  # Controlla quanto i pesi sono "sharp"
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

        print(f"\n[FedGSW] === Round {server_round} ===")
        print(f"[FedGSW] Ricevuti aggiornamenti da {len(results)} client.")

        # 1. Estrai parametri e numero di esempi
        client_params: List[List[np.ndarray]] = []
        num_examples: List[int] = []
        
        for _, fit_res in results:
            client_params.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(int(fit_res.num_examples))

        # 2. Calcola i delta (aggiornamenti) rispetto al modello globale precedente
        if self._prev_global is None:
            self._prev_global = [np.copy(a) for a in client_params[0]]
        
        deltas: List[np.ndarray] = []
        for w_k in client_params:
            delta = _flatten_layers([wk - wg for wk, wg in zip(w_k, self._prev_global)])
            deltas.append(delta)
        
        # 3. Calcola la media pesata degli aggiornamenti (baseline FedAvg)
        total_examples = float(sum(num_examples))
        base_weights = np.array([ne / total_examples for ne in num_examples], dtype=np.float64)
        
        mean_delta = np.zeros_like(deltas[0])
        for i, delta in enumerate(deltas):
            mean_delta += base_weights[i] * delta
        
        print(f"[FedGSW] Calcolata media pesata degli aggiornamenti.")
        
        # 4. Calcola similarità coseno tra ogni client e la media
        similarities = np.array([
            _cosine_similarity(delta, mean_delta) for delta in deltas
        ], dtype=np.float64)
        
        # Normalizza le similarità tra 0 e 1 (possono essere negative)
        similarities = (similarities + 1.0) / 2.0
        
        print("[FedGSW] Similarità coseno con la media:")
        for i, sim in enumerate(similarities):
            print(f"  - Client {i}: {sim:.4f}")
        
        # 5. Converti similarità in pesi tramite softmax
        gsw_weights = _softmax(similarities, temperature=self.temperature)
        
        # 6. Combina con i pesi base (numero di esempi)
        final_weights = gsw_weights * base_weights
        final_weights = final_weights / np.sum(final_weights)
        
        print("[FedGSW] Pesi di aggregazione finali:")
        for i, w in enumerate(final_weights):
            print(f"  - Client {i}: {w:.4f}")
        
        # 7. Aggrega i parametri
        aggregated_params: List[np.ndarray] = []
        for layer_idx in range(len(client_params[0])):
            layer_sum = np.zeros_like(client_params[0][layer_idx])
            for client_idx, params in enumerate(client_params):
                layer_sum += final_weights[client_idx] * params[layer_idx]
            aggregated_params.append(layer_sum)
        
        # 8. Aggiorna il modello globale precedente
        self._prev_global = [np.copy(a) for a in aggregated_params]
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        
        # Metriche
        metrics_aggregated: Dict[str, Scalar] = {
            "avg_similarity": float(np.mean(similarities)),
        }
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))
        
        return aggregated_parameters, metrics_aggregated


# Alias per compatibilità con flower_baseline.py
FedTest = FedGSW

__all__ = ["FedGSW", "FedTest"]
