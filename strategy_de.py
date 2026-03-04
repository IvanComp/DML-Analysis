"""strategy_de.py

FedDE: Federated Differential Evolution Aggregation

Strategia di aggregazione che usa Differential Evolution per cercare
i pesi ottimali dei client ad ogni round. Più lenta ma potenzialmente
più accurata perché esplora combinazioni non-lineari.

MECCANISMO:
1. Definisce una funzione obiettivo basata sulla coerenza degli aggiornamenti
2. Usa DE per cercare i pesi ottimali nello spazio [0,1]^n_clients
3. I pesi ottimali vengono normalizzati e usati per l'aggregazione

FUNZIONE OBIETTIVO:
Minimizza la "divergenza residua" dopo l'aggregazione, definita come
la somma delle distanze tra ogni aggiornamento client e l'aggiornamento
aggregato pesato.

VINCOLI RISPETTATI:
- No dati aggiuntivi lato server
- Solo pesi dei client (privacy preservata)
- Computazione aggiuntiva sul server (DE richiede ~2-5 sec/round)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def _flatten_layers(arrs: List[np.ndarray]) -> np.ndarray:
    """Appiattisce una lista di array numpy in un singolo vettore."""
    return np.concatenate([a.reshape(-1) for a in arrs], axis=0)


class FedDE(fl.server.strategy.FedAvg):
    """
    Differential Evolution Aggregation Strategy.
    
    Usa DE per trovare i pesi ottimali che minimizzano la divergenza
    tra gli aggiornamenti dei client e l'aggiornamento aggregato.
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
        de_maxiter: int = 50,      # Iterazioni DE (più alto = più lento ma migliore)
        de_popsize: int = 10,       # Popolazione DE
        de_mutation: float = 0.7,   # Fattore di mutazione
        de_recombination: float = 0.7,  # Tasso di ricombinazione
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
        self.de_maxiter = int(de_maxiter)
        self.de_popsize = int(de_popsize)
        self.de_mutation = float(de_mutation)
        self.de_recombination = float(de_recombination)
        
        self._prev_global: Optional[List[np.ndarray]] = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else None
        )

    def _objective_function(
        self,
        weights: np.ndarray,
        deltas_flat: List[np.ndarray],
        num_examples: List[int],
    ) -> float:
        """
        Funzione obiettivo per DE.
        Minimizza la divergenza residua dopo aggregazione.
        """
        # Normalizza i pesi
        weights = np.abs(weights)
        weights = weights / (np.sum(weights) + 1e-12)
        
        # Calcola l'aggiornamento aggregato
        n_params = len(deltas_flat[0])
        aggregated = np.zeros(n_params, dtype=np.float64)
        for i, delta in enumerate(deltas_flat):
            aggregated += weights[i] * delta
        
        # Calcola la divergenza residua (somma delle distanze)
        divergence = 0.0
        for i, delta in enumerate(deltas_flat):
            # Pesa la divergenza per il numero di esempi
            dist = np.linalg.norm(delta - aggregated)
            divergence += num_examples[i] * dist
        
        # Aggiungi un termine di regolarizzazione per preferire pesi più uniformi
        # (evita soluzioni degeneri con pesi concentrati su un solo client)
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        max_entropy = np.log(len(weights))
        entropy_penalty = 0.1 * (max_entropy - entropy)
        
        return divergence + entropy_penalty

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[FedDE] === Round {server_round} ===")
        print(f"[FedDE] Ricevuti aggiornamenti da {len(results)} client.")

        # 1. Estrai parametri e metadati
        client_params: List[List[np.ndarray]] = []
        num_examples: List[int] = []
        
        for _, fit_res in results:
            client_params.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(int(fit_res.num_examples))

        n_clients = len(client_params)
        
        # 2. Inizializza il modello globale precedente se necessario
        if self._prev_global is None:
            self._prev_global = [np.copy(a) for a in client_params[0]]

        # 3. Calcola gli aggiornamenti (delta)
        deltas_flat: List[np.ndarray] = []
        for w_k in client_params:
            delta = _flatten_layers([wk - wg for wk, wg in zip(w_k, self._prev_global)])
            deltas_flat.append(delta)
        
        # 4. Usa Differential Evolution per trovare i pesi ottimali
        print(f"[FedDE] Ottimizzazione con DE (maxiter={self.de_maxiter}, popsize={self.de_popsize})...")
        
        bounds = [(0.01, 1.0) for _ in range(n_clients)]
        
        result = differential_evolution(
            func=self._objective_function,
            bounds=bounds,
            args=(deltas_flat, num_examples),
            maxiter=self.de_maxiter,
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_recombination,
            seed=server_round,  # Per riproducibilità
            disp=False,
            workers=1,  # Single thread per evitare overhead
        )
        
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        print(f"[FedDE] Ottimizzazione completata. Divergenza finale: {result.fun:.6f}")
        print("[FedDE] Pesi ottimali trovati:")
        for i, w in enumerate(optimal_weights):
            print(f"  - Client {i}: {w:.4f}")
        
        # 5. Aggrega i parametri con i pesi ottimali
        aggregated_params: List[np.ndarray] = []
        for layer_idx in range(len(client_params[0])):
            layer_sum = np.zeros_like(client_params[0][layer_idx])
            for client_idx, params in enumerate(client_params):
                layer_sum += optimal_weights[client_idx] * params[layer_idx]
            aggregated_params.append(layer_sum)
        
        # 6. Aggiorna il modello globale precedente
        self._prev_global = [np.copy(a) for a in aggregated_params]
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        
        # Metriche
        metrics_aggregated: Dict[str, Scalar] = {
            "de_divergence": float(result.fun),
            "de_iterations": int(result.nit),
        }
        
        return aggregated_parameters, metrics_aggregated


__all__ = ["FedDE"]
