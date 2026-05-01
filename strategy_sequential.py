"""Sequential Flower strategies for non-federated training flows."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


def _cid_sort_key(cid: str) -> Tuple[int, Union[int, str]]:
    """Sort numeric client IDs numerically and the rest lexicographically."""
    try:
        return (0, int(cid))
    except ValueError:
        return (1, cid)


class SequentialRoundRobin(fl.server.strategy.FedAvg):
    """Train exactly one client per round, rotating deterministically."""

    def __init__(self, *args, evaluate_every_n_rounds: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_every_n_rounds = max(1, int(evaluate_every_n_rounds))

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)
        clients_by_id = client_manager.all()
        if not clients_by_id:
            return []

        ordered_cids = sorted(clients_by_id.keys(), key=_cid_sort_key)
        selected_cid = ordered_cids[(server_round - 1) % len(ordered_cids)]
        return [(clients_by_id[selected_cid], fit_ins)]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        if server_round % self.evaluate_every_n_rounds != 0:
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        _, fit_res = results[0]

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif fit_res.metrics:
            metrics_aggregated = {
                key: value
                for key, value in fit_res.metrics.items()
                if isinstance(value, (bool, bytes, float, int, str))
            }

        return fit_res.parameters, metrics_aggregated


__all__ = ["SequentialRoundRobin"]
