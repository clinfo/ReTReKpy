""" The ``scripts.in_scope_filter`` directory ``train`` script. """

import numpy
import torch

from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from sklearn.metrics import roc_auc_score, accuracy_score

from torch.utils.data import DataLoader, Subset

from retrekpy.in_scope_filter import InScopeFilterNetwork, Meter, ReactionDataset


def prepare_batch(
        batch: Tuple,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ The 'prepare_batch' function. """

    reaction, product, y = batch

    reaction = reaction.to(device)
    product = product.to(device)
    y = y.to(device)

    return reaction, product, y


def train_epoch(
        model: InScopeFilterNetwork,
        optimizer: torch.optim.AdamW,
        loss_fn: torch.nn.BCEWithLogitsLoss,
        loader: DataLoader,
        epoch: int,
        acc_meter: Meter,
        auc_meter: Meter,
        device: torch.device
) -> None:
    """ The 'train_epoch' function. """

    pbar = tqdm(loader)

    model.train()

    auc = 0

    for i, batch in enumerate(pbar):
        reaction, product, y = prepare_batch(
            batch=batch,
            device=device
        )

        out = model(reaction, product, logits=True)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            labels = y.cpu().numpy()
            preds = torch.sigmoid(out).detach().cpu().numpy()

            if not (labels == labels[0]).all():
                auc = roc_auc_score(labels, preds)

            bin_preds = (preds > 0.5).astype(int)

            acc = accuracy_score(labels, bin_preds)

            acc_meter.update(acc)
            auc_meter.update(auc)

        pbar.set_description(
            f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {acc_meter.value:.4f} | AUC: {auc_meter.value:.4f}"
        )


def validation_epoch(
        model: InScopeFilterNetwork,
        loader: DataLoader,
        device: torch.device
) -> Tuple[float, float]:
    """ The 'validation_epoch' function. """

    pbar = tqdm(loader, leave=False)

    model.eval()

    all_outs = list()
    all_labels = list()

    with torch.no_grad():
        for batch in pbar:
            reaction, product, y = prepare_batch(batch, device)

            out = model(reaction, product)

            all_outs.append(out.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    labels = numpy.concatenate(all_labels, axis=0)
    preds = numpy.concatenate(all_outs, axis=0)

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds > 0.5).astype(int))

    print(f"Validation metrics: Acc: {acc:.4f} | AUC: {auc:.4f}")

    return auc, acc


def get_script_arguments(
) -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-c",
        "--csv_path",
        type=str,
        help="Path to csv dataset."
    )

    argument_parser.add_argument(
        "-rc",
        "--reaction_column",
        type=str,
        default="reaction_smiles",
        help="Reaction column name."
    )

    argument_parser.add_argument(
        "-pc",
        "--product_column",
        type=str,
        default="main_product",
        help="Product column name."
    )

    argument_parser.add_argument(
        "-lc",
        "--label_column",
        type=str,
        default="label",
        help="Label column name."
    )

    argument_parser.add_argument(
        "-pw",
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive weight for BCE loss."
    )

    argument_parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate."
    )

    argument_parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay."
    )

    argument_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs."
    )

    argument_parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="Number of epochs."
    )

    argument_parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloaders."
    )

    argument_parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
        help="Use CUDA."
    )

    argument_parser.add_argument(
        "-sp",
        "--save_path",
        type=str,
        default="results",
        help="Save folder path."
    )

    return argument_parser.parse_args()


if __name__ == "__main__":
    script_arguments = get_script_arguments()

    ds = ReactionDataset(
        csv_path=script_arguments.csv_path,
        reaction_column=script_arguments.reaction_column,
        product_column=script_arguments.product_column,
        label_column=script_arguments.label_column
    )

    indices = numpy.arange(len(ds))
    numpy.random.shuffle(indices)

    train_ds = Subset(
        ds,
        indices=indices[:int(0.8*len(ds))]
    )

    val_ds = Subset(
        ds,
        indices=indices[int(0.8*len(ds)):]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=script_arguments.batch_size,
        shuffle=True,
        num_workers=script_arguments.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=script_arguments.batch_size,
        shuffle=True,
        num_workers=script_arguments.num_workers
    )

    in_scope_filter_model = InScopeFilterNetwork()
    model_device = torch.device("cuda") if script_arguments.use_cuda else torch.device("cpu")

    model_loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([script_arguments.pos_weight]).to(model_device)
    )

    save_path = Path(script_arguments.save_path) / "best_ckpt.pt"

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_optimizer = torch.optim.AdamW(
        params=in_scope_filter_model.parameters(),
        lr=script_arguments.learning_rate,
        weight_decay=script_arguments.weight_decay
    )

    in_scope_filter_model.to(model_device)

    acc_metric_meter, auc_metric_meter = Meter(), Meter()

    best_auc_metric = -numpy.inf

    for epoch_index in range(1, script_arguments.epochs + 1):
        train_epoch(
            model=in_scope_filter_model,
            optimizer=model_optimizer,
            loss_fn=model_loss_fn,
            loader=train_loader,
            epoch=epoch_index,
            acc_meter=acc_metric_meter,
            auc_meter=auc_metric_meter,
            device=model_device
        )

        auc_metric, acc_metric = validation_epoch(
            model=in_scope_filter_model,
            loader=val_loader,
            device=model_device
        )

        if auc_metric > best_auc_metric:
            print("New best checkpoint!")

            torch.save(in_scope_filter_model.state_dict(), save_path)

            best_auc_metric = auc_metric
