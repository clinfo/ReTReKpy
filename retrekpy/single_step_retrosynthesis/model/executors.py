""" The ``retrekpy.single_step_retrosynthesis.model`` package ``executors`` module. """

import logging
import numpy
import torch

from typing import Union

from rdkit.Chem import Mol, MolToSmiles

from torch_geometric.data import Data

from kmol.core.config import Config
from kmol.core.helpers import Namespace
from kmol.core.observers import EventManager
from kmol.core.utils import progress_bar
from kmol.data.resources import Batch, LoadedContent
from kmol.model.executors import Predictor, Trainer

from ..data.featurizers import CustomizedGraphFeaturizer


class CustomizedTrainer(Trainer):
    """ The customized 'Trainer' class. """

    def _to_device(
            self,
            batch: Batch
    ) -> None:
        """ The customized '_to_device' method of the class. """

        batch.outputs = batch.outputs.type(torch.LongTensor).to(self._device)

        for key, values in batch.inputs.items():
            try:
                if isinstance(values, torch.Tensor) or issubclass(type(values), Data):
                    batch.inputs[key] = values.to(self._device)

                elif isinstance(values, dict):
                    batch.inputs[key] = self.dict_to_device(values)

                elif isinstance(values, list):
                    if isinstance(values[0], torch.Tensor):
                        batch.inputs[key] = [a.to(self._device) for a in values]

                    else:
                        batch.inputs[key] = [a for a in values]

                else:
                    batch.inputs[key] = values

            except (AttributeError, ValueError) as e:
                logging.debug(e)

                pass

    def _training_step(
            self,
            batch: Batch,
            epoch: int
    ) -> None:
        """ The customized '_training_step' method of the class. """

        self._to_device(batch)

        self.optimizer.zero_grad()

        outputs = self.network(batch.inputs)

        payload = Namespace(
            features=batch,
            logits=outputs,
            extras=[],
            epoch=epoch,
            config=self.config
        )

        EventManager.dispatch_event(
            event_name="before_criterion",
            payload=payload
        )

        loss = self.criterion(payload.logits, payload.features.outputs.squeeze(), *payload.extras)

        loss.backward()

        self.optimizer.step()

        if self.config.is_stepwise_scheduler:
            self.scheduler.step()

        payload = Namespace(
            outputs=outputs
        )

        EventManager.dispatch_event(
            event_name="before_tracker_update",
            payload=payload
        )

        outputs = payload.outputs

        self._update_trackers(loss.item(), batch.outputs, outputs)

    @torch.no_grad()
    def _validation(
            self,
            val_loader: LoadedContent
    ) -> Namespace:
        """ The customized '_validation' method of the class. """

        if val_loader is None:
            return Namespace()

        self.network.eval()

        metrics = list()

        with progress_bar() as progress:
            for batch in progress.track(
                val_loader.dataset,
                description="Validating..."
            ):
                self._to_device(batch)

                payload = Namespace(
                    logits=self.network(batch.inputs),
                    logits_var=None,
                    softmax_score=None
                )

                EventManager.dispatch_event(
                    event_name="after_val_inference",
                    payload=payload
                )

                metrics.append(
                    CustomizedTrainer.calculate_top_k_accuracy(
                        logits=payload.logits,
                        ground_truth=batch.outputs,
                        k=int(self.config.target_metric.split("_")[1])
                    )
                )

            averages = Namespace(**{
                self.config.target_metric: [numpy.mean(numpy.array(metrics))]
            })

        return averages

    @staticmethod
    def calculate_top_k_accuracy(
            logits: torch.Tensor,
            ground_truth: torch.Tensor,
            k: int
    ) -> float:
        """ The 'calculate_top_k_accuracy' method of the class. """

        _, indices_of_the_top_k_predictions = torch.topk(
            input=torch.softmax(logits, dim=1),
            k=k,
            dim=1
        )

        return (indices_of_the_top_k_predictions == ground_truth).any(dim=1).float().mean().item()


class CustomizedEvaluator(Predictor):
    """ The customized 'Evaluator' class. """

    def evaluate(
            self,
            data_loader: LoadedContent
    ) -> Namespace:
        """ The 'evaluate' method of the class. """

        metrics = []

        with progress_bar() as progress:
            for batch in progress.track(
                data_loader.dataset,
                description="Evaluating..."
            ):
                metrics.append(
                    CustomizedTrainer.calculate_top_k_accuracy(
                        logits=self.run(batch).logits,
                        ground_truth=batch.outputs,
                        k=int(self.config.target_metric.split("_")[1])
                    )
                )

        return Namespace(
            **{self.config.target_metric: [numpy.mean(numpy.array(metrics))]}
        )


class CustomizedPredictor(Predictor):
    """ The customized 'Predictor' class. """

    def __init__(
            self,
            config: Config
    ) -> None:
        """ The customized constructor method of the class. """

        super().__init__(config)

        config.featurizers[0].pop("type")

        self._featurizer = CustomizedGraphFeaturizer(
            **config.featurizers[0]
        )

    def predict(
            self,
            compound: Union[str, Mol]
    ) -> torch.Tensor:
        """ The 'predict' method of the class. """

        data = {
            "graph": self._featurizer.process(
                data=compound if isinstance(compound, str) else MolToSmiles(compound)
            )
        }

        with torch.inference_mode():
            return torch.softmax(self.network(data), dim=1).squeeze().numpy()
