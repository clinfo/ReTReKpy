""" The ``retrekpy.ease_of_synthesis`` package ``executors`` module. """

import numpy
import torch
import logging

from json import dump, loads
from optuna import create_study, Study, Trial
from pathlib import Path
from re import findall
from tqdm import tqdm
from typing import Any, Dict, Tuple, Union

from torch.utils.data import DataLoader

from kmol.data.resources import Batch
from kmol.core.config import Config
from kmol.core.helpers import Namespace
from kmol.core.observers import EventManager
from kmol.model.executors import Trainer as kTrainer


class Trainer(kTrainer):
    """ The 'Trainer' class. """

    def _to_device(
            self,
            batch: Batch
    ) -> None:
        """ The '_to_device' method class. """

        batch.outputs = batch.outputs.to(self._device)

        for key, values in batch.inputs.items():
            try:
                batch.inputs[key] = values.to(self._device)

            except (AttributeError, ValueError):
                pass

    def _training_step(
            self,
            batch: Batch,
            epoch: int
    ) -> None:
        """ The '_training_step' method class. """

        self.optimizer.zero_grad()

        outputs = self.network(batch.inputs)

        loss = self.criterion(outputs, batch.outputs.clone())

        loss.backward()

        self.optimizer.step()

        if self.config.is_stepwise_scheduler:
            self.scheduler.step()

        self._update_trackers(loss.item(), batch.outputs, outputs)

    def _train_epoch(
            self,
            train_loader: DataLoader,
            epoch: int
    ) -> None:
        """ The '_train_epoch' method class. """

        self.network.train()

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), leave=False)

        for iteration, batch in pbar:
            self._to_device(batch)
            self._training_step(batch, epoch)

            if iteration % self.config.log_frequency == 0:
                pbar.set_description(f"Epoch {epoch} | Train Loss: {self._loss_tracker.get():.5f}")

        if not self.config.is_stepwise_scheduler:
            self.scheduler.step()

    @torch.no_grad()
    def _validation(
            self,
            val_loader,
            return_breakdown: bool = False
    ) -> Union[Namespace, Tuple[Namespace, Namespace]]:
        """ The '_validation' method class. """

        ground_truth = []
        logits = []

        self.network.eval()

        for batch in tqdm(val_loader, leave=False):
            self._to_device(batch)

            ground_truth.append(batch.outputs)
            logits.append(self.network(batch.inputs))

        metrics = self._metric_computer.compute_metrics(ground_truth, logits)
        averages = self._metric_computer.compute_statistics(metrics, (numpy.mean,))

        if return_breakdown:
            return averages, metrics

        return averages

    def _check_best(
            self,
            epoch,
            val_metrics,
            best_metric
    ) -> Tuple[float, bool]:
        """ The '_check_best' method class. """

        target_metric = getattr(val_metrics, self.config.target_metric, [0])[0]

        new_best = target_metric > best_metric

        if new_best:
            best_metric = target_metric
            self.save(epoch)

        return best_metric, new_best

    def run(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader
    ) -> None:
        """ The 'run' method class. """

        self._device = self.config.get_device()

        if "pos_weight" in self.config.criterion:
            self.config.criterion["pos_weight"] = torch.FloatTensor([self.config.criterion["pos_weight"]])

        if self.config.scheduler["type"] == "torch.optim.lr_scheduler.OneCycleLR":
            self.config.scheduler["max_lr"] = 10*self.config.optimizer["lr"]
            self.config.scheduler["epochs"] = self.config.epochs

        self._setup(training_examples=len(train_loader.dataset))

        initial_payload = Namespace(trainer=self, data_loader=train_loader)
        EventManager.dispatch_event(event_name="before_train_start", payload=initial_payload)
        best_metric = -numpy.inf

        for epoch in range(self._start_epoch + 1, self.config.epochs + 1):
            self._train_epoch(train_loader, epoch)

            val_metrics = self._validation(val_loader)
            best_metric, new_best = self._check_best(epoch, val_metrics, best_metric)

            self.log(epoch, val_metrics, new_best)
            self._reset_trackers()

        EventManager.dispatch_event(event_name="after_train_end", payload=initial_payload)

        self.best_metric = best_metric

    def log(
            self,
            epoch: int,
            val_metrics: Namespace,
            new_best: bool
    ) -> None:
        """ The 'log' method class. """

        message = "epoch: {} - Train loss: {:.4f} - time elapsed: {}".format(
            epoch,
            self._loss_tracker.get(),
            str(self._timer),
        )

        for name, tracker in self._metric_trackers.items():
            message += " - Train {}: {:.4f}".format(name, tracker.get())

        for name, value in vars(val_metrics).items():
            message += " - Val {}: {:.4f}".format(name, value[0])

        message += " (New best)" if new_best else ""

        logging.info(message)

        with (Path(self.config.output_path) / "logs.txt").open("a") as f:
            f.write(message+"\n")

    def save(
            self,
            epoch: int
    ) -> None:
        """ The 'save' method class. """

        info = {
            "epoch": epoch,
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

        model_path = Path(self.config.output_path) / "best_ckpt.pt"

        logging.debug("Saving checkpoint: {}".format(model_path))

        payload = Namespace(info=info)
        EventManager.dispatch_event(event_name="before_checkpoint_save", payload=payload)

        torch.save(info, model_path)


class Evaluator(Trainer):
    """ The 'Evaluator' class. """

    def run(
            self,
            data_loader: DataLoader
    ) -> None:
        """ The 'run' method class. """

        self._device = self.config.get_device()
        self._load_checkpoint()

        metrics, breakdown = self._validation(data_loader, return_breakdown=True)

        print(f"\nNumber of samples: {len(data_loader.dataset)}")
        logging.info(f"Number of samples: {len(data_loader.dataset)}")

        print("\nBreakdown:")

        for name, value in vars(metrics).items():
            print("{}: {:.4f}".format(name, value[0]))
            logging.info("{}: {:.4f}".format(name, value[0]))

        print("\nBreakdown per task:")
        logging.info("Breakdown per class:")

        for name, value in vars(breakdown).items():
            print("{}: {}".format(name, [round(e, 4) for e in value]))
            logging.info("{}: {}".format(name, [round(e, 4) for e in value]))

        with open(Path(self.config.output_path) / "metrics.json", "w") as f:
            dump(vars(metrics), f)


class Predictor(Trainer):
    """ The 'Predictor' class. """

    @torch.no_grad()
    def run(
            self,
            data_loader: DataLoader
    ) -> torch.Tensor:
        """ The 'run' method class. """

        self._device = self.config.get_device()
        self._load_checkpoint()

        logits = []

        self.network.eval()

        for batch in tqdm(data_loader, leave=False):
            self._to_device(batch)

            logits.append(self.network(batch.inputs))

        return torch.cat(logits).cpu()


class BayesianOptimizer:
    """ The 'BayesianOptimizer' class. """

    def __init__(
            self,
            config_path: str
    ) -> None:
        """ The constructor method class. """

        with open(config_path) as read_buffer:
            self._template = read_buffer.read()
            self._template = self._template.replace(" ", "").replace("\n", "")

        self.config = Config.from_file(config_path, "bo")

    def _suggest_configuration(
            self,
            trial: Trial
    ) -> Dict[str, Any]:
        """ The '_suggest_configuration' method class. """

        replacements = {}

        for id_, key in enumerate(findall(r"{{{.*?}}}", self._template), start=1):
            name, placeholder = key[3:-3].split("=")

            if "|" in placeholder:
                replacements[key] = trial.suggest_categorical(name, placeholder.split("|"))
                continue

            key = "\"{}\"".format(key)

            low, high, step = placeholder.split("-")

            if "." in placeholder:
                replacements[key] = trial.suggest_float(name=name, low=float(low), high=float(high), step=float(step))

            else:
                replacements[key] = trial.suggest_int(name=name, low=int(low), high=int(high), step=int(step))

        template = self._template

        for key, value in replacements.items():
            template = template.replace(key, str(value))

        return loads(template)

    def __run_trial(
            self,
            trial: Trial
    ) -> float:
        """ The '__run_trial' method class. """

        try:
            settings = self._suggest_configuration(trial)

            logging.info(f"Trial parameters: {trial.params}")

            config = Config(**settings)

            trainer = Trainer(config)

            trainer.run(self.train_loader, self.val_loader)

            results = trainer.best_metric

            return results

        except Exception as e:
            logging.error("[Trial Failed] {}".format(e))

            return -numpy.inf

    def optimize(
            self
    ) -> Study:
        """ The 'optimize' method class. """

        study = create_study(direction="maximize")

        study.optimize(self.__run_trial, n_trials=self.config.optuna_trials, callbacks=[
            lambda study, trial: study.trials_dataframe().to_csv(
                Path(self.config.output_path) / "optuna_trials.csv")
        ])

        logging.info("---------------------------- [BEST VALUE] ----------------------------")
        logging.info(study.best_value)
        logging.info("---------------------------- [BEST TRIAL] ---------------------------- ")
        logging.info(study.best_trial)
        logging.info("---------------------------- [BEST PARAMS] ----------------------------")
        logging.info(study.best_params)

        return study

    def run(
            self,
            train_loader,
            val_loader
    ) -> None:
        """ The 'run' method class. """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimize()
