"""strategy_sbfl.py

FedSBFL: Federated Spectrum-Based Fault Localization Aggregation

Strategia di aggregazione ispirata alle tecniche di Fault Localization
usate nel software testing. Tratta i client come "componenti" e usa
metriche come Tarantula, Ochiai e DStar per identificare quelli che
contribuiscono positivamente alla convergenza.

MECCANISMO:
1. Valuta l'impatto di ogni client sulla loss globale
2. Classifica ogni "test" (round/sub-evaluation) come pass/fail
3. Calcola lo score di "sospetto" (o affidabilità) per ogni client
4. Usa lo score per pesare l'aggregazione

METRICHE SBFL:
- Tarantula: failed(e)/total_failed / (failed(e)/total_failed + passed(e)/total_passed)
- Ochiai: failed(e) / sqrt(total_failed * (failed(e) + passed(e)))
- DStar: failed(e)^* / (passed(e) + not_failed(e))  [* = 2 di default]

In questo contesto:
- "passed" = il client ha contribuito a ridurre la divergenza
- "failed" = il client ha contribuito ad aumentare la divergenza

VINCOLI RISPETTATI:
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


class FedSBFL(fl.server.strategy.FedAvg):
    """
    Spectrum-Based Fault Localization Aggregation Strategy.
    
    Usa metriche di fault localization per identificare client
    che contribuiscono positivamente alla convergenza.
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
        sbfl_metric: str = "dstar",  # "tarantula", "ochiai", "dstar"
        dstar_exponent: float = 2.0,  # Esponente per DStar
        history_window: int = 10,     # Finestra storica per accumulare pass/fail
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
        self.sbfl_metric = sbfl_metric.lower()
        self.dstar_exponent = float(dstar_exponent)
        self.history_window = int(history_window)
        
        self._prev_global: Optional[List[np.ndarray]] = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else None
        )
        
        # Contatori pass/fail per ogni client (dizionario: client_id -> [passed, failed])
        self._client_spectrum: Dict[str, List[int]] = {}
        
        # Storia delle medie per calcolare il "target" ideale
        self._mean_history: List[np.ndarray] = []

    def _evaluate_client_contribution(
        self,
        client_delta: np.ndarray,
        mean_delta: np.ndarray,
        all_deltas: List[np.ndarray],
    ) -> bool:
        """
        Valuta se un client ha contribuito positivamente.
        
        Criterio: il client "passa" se il suo aggiornamento è più vicino
        alla media rispetto alla mediana delle distanze di tutti i client.
        """
        # Distanza del client dalla media
        client_dist = np.linalg.norm(client_delta - mean_delta)
        
        # Mediana delle distanze
        all_dists = [np.linalg.norm(d - mean_delta) for d in all_deltas]
        median_dist = np.median(all_dists)
        
        # "Passa" se è più vicino della mediana
        return client_dist <= median_dist

    def _compute_sbfl_score(self, passed: int, failed: int, total_passed: int, total_failed: int) -> float:
        """Calcola lo score SBFL basato sulla metrica selezionata."""
        
        # Evita divisioni per zero
        eps = 1e-12
        
        if self.sbfl_metric == "tarantula":
            # Tarantula = (failed/total_failed) / (failed/total_failed + passed/total_passed)
            fail_ratio = failed / max(total_failed, eps)
            pass_ratio = passed / max(total_passed, eps)
            return fail_ratio / max(fail_ratio + pass_ratio, eps)
        
        elif self.sbfl_metric == "ochiai":
            # Ochiai = failed / sqrt(total_failed * (failed + passed))
            return failed / max(np.sqrt(total_failed * (failed + passed)), eps)
        
        elif self.sbfl_metric == "dstar":
            # DStar = failed^* / (passed + (total_failed - failed))
            # Nota: in FL invertiamo il concetto - vogliamo ALTA affidabilità
            # Quindi usiamo: passed^* / (failed + (total_passed - passed))
            return (passed ** self.dstar_exponent) / max(failed + (total_passed - passed), eps)
        
        else:
            # Default: rapporto semplice
            return passed / max(passed + failed, eps)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[FedSBFL] === Round {server_round} ===")
        print(f"[FedSBFL] Ricevuti aggiornamenti da {len(results)} client.")
        print(f"[FedSBFL] Metrica SBFL: {self.sbfl_metric.upper()}")

        # 1. Estrai parametri e metadati
        client_params: List[List[np.ndarray]] = []
        num_examples: List[int] = []
        client_ids: List[str] = []
        
        for proxy, fit_res in results:
            client_params.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(int(fit_res.num_examples))
            client_ids.append(str(proxy.cid))

        n_clients = len(client_params)
        
        # 2. Inizializza il modello globale precedente se necessario
        if self._prev_global is None:
            self._prev_global = [np.copy(a) for a in client_params[0]]

        # 3. Calcola i pesi base (numero di esempi)
        total_examples = float(sum(num_examples))
        base_weights = np.array([ne / total_examples for ne in num_examples], dtype=np.float64)
        
        # 4. Calcola gli aggiornamenti (delta)
        deltas_flat: List[np.ndarray] = []
        for w_k in client_params:
            delta = _flatten_layers([wk - wg for wk, wg in zip(w_k, self._prev_global)])
            deltas_flat.append(delta)
        
        # 5. Calcola la media pesata degli aggiornamenti
        mean_delta = np.zeros_like(deltas_flat[0])
        for i, delta in enumerate(deltas_flat):
            mean_delta += base_weights[i] * delta
        
        # 6. Valuta ogni client e aggiorna lo spettro
        for i, cid in enumerate(client_ids):
            if cid not in self._client_spectrum:
                self._client_spectrum[cid] = [0, 0]  # [passed, failed]
            
            passed = self._evaluate_client_contribution(deltas_flat[i], mean_delta, deltas_flat)
            
            if passed:
                self._client_spectrum[cid][0] += 1
            else:
                self._client_spectrum[cid][1] += 1
        
        # 7. Calcola i totali
        total_passed = sum(s[0] for s in self._client_spectrum.values())
        total_failed = sum(s[1] for s in self._client_spectrum.values())
        
        # 8. Calcola gli score SBFL per ogni client
        sbfl_scores = []
        for cid in client_ids:
            passed, failed = self._client_spectrum[cid]
            score = self._compute_sbfl_score(passed, failed, total_passed, total_failed)
            sbfl_scores.append(score)
        
        sbfl_scores = np.array(sbfl_scores, dtype=np.float64)
        
        # Normalizza gli score
        sbfl_scores = sbfl_scores / max(np.sum(sbfl_scores), 1e-12)
        
        print("[FedSBFL] Score SBFL per client:")
        for i, (cid, score) in enumerate(zip(client_ids, sbfl_scores)):
            p, f = self._client_spectrum[cid]
            print(f"  - Client {i}: score={score:.4f} (passed={p}, failed={f})")
        
        # 9. Combina con i pesi base
        final_weights = sbfl_scores * base_weights
        final_weights = final_weights / np.sum(final_weights)
        
        print("[FedSBFL] Pesi di aggregazione finali:")
        for i, w in enumerate(final_weights):
            print(f"  - Client {i}: {w:.4f}")
        
        # 10. Aggrega i parametri
        aggregated_params: List[np.ndarray] = []
        for layer_idx in range(len(client_params[0])):
            layer_sum = np.zeros_like(client_params[0][layer_idx])
            for client_idx, params in enumerate(client_params):
                layer_sum += final_weights[client_idx] * params[layer_idx]
            aggregated_params.append(layer_sum)
        
        # 11. Aggiorna il modello globale precedente
        self._prev_global = [np.copy(a) for a in aggregated_params]
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        
        # Metriche
        metrics_aggregated: Dict[str, Scalar] = {
            "total_passed": int(total_passed),
            "total_failed": int(total_failed),
        }
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))
        
        return aggregated_parameters, metrics_aggregated


__all__ = ["FedSBFL"]
