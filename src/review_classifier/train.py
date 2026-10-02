"""Train the project's RoBERTa-base classifier and save it ready to serve.

    python -m review_classifier.train --task sentiment --output checkpoints/roberta-sentiment

This ports ``train_roberta`` from the cross-variety-sentiment-sarcasm notebook
(stage Q2.1, all varieties pooled). The recipe is unchanged:

* ``roberta-base``, binary classification, trained on all three varieties
* raw text with no cleaning, truncated at 256 tokens (sentiment) or 384 (sarcasm),
  padded per batch
* cross-entropy weighted by ``balanced`` class weights from the training labels
* 4 epochs, learning rate 2e-5, batch 16 / 32, 10% warmup, weight decay 0.01,
  fp16 on GPU
* the checkpoint with the best validation macro-F1 is kept; the test split is
  evaluated once, after that choice

What changes is that the result is kept and can be served exactly:

* the selected checkpoint is saved instead of deleted
* labels get real names instead of ``class_0`` / ``class_1``
* ``serving_config.json`` records the task, truncation length and raw-text
  preprocessing, so the service reproduces training automatically
* ``training_manifest.json`` records the exact base-model and dataset versions,
  recipe, environment, and validation and test metrics
* ``--push-to-hub`` uploads everything and prints the commit to deploy

Needs the ``train`` extra and, in practice, a GPU (a free Colab T4 is enough).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import classification_metrics
from .model_source import hash_directory, resolve_model_source
from .serving_config import RAW_TEXT, CheckpointServingConfig, write_serving_config

DEFAULT_DATASET = "surrey-nlp/BESSTIE-CW-26"
# Canonical id of roberta-base. The legacy alias "roberta-base" redirects for most
# requests but not for Xet storage downloads, which then fail with 404.
DEFAULT_BASE_MODEL = "FacebookAI/roberta-base"
MANIFEST_FILE = "training_manifest.json"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    column: str
    max_length: int
    labels: tuple[str, str]  # index = class id in the dataset


# Label meanings follow the dataset: 0/1 = negative/positive, not sarcastic/sarcastic.
TASKS = {
    "sentiment": TaskSpec("sentiment", "Sentiment", 256, ("negative", "positive")),
    "sarcasm": TaskSpec("sarcasm", "Sarcasm", 384, ("not_sarcastic", "sarcastic")),
}


@dataclass(frozen=True)
class Recipe:
    epochs: int = 4
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42


def balanced_class_weights(labels: list[int], num_classes: int = 2) -> list[float]:
    """Same values as sklearn's compute_class_weight('balanced'): n / (k * count)."""
    counts = [labels.count(c) for c in range(num_classes)]
    if 0 in counts:
        raise ValueError(f"Every class needs training examples; counts were {counts}")
    return [len(labels) / (num_classes * count) for count in counts]


# ---- data -------------------------------------------------------------------------


def load_splits(dataset: str, revision: str | None):
    """Load the dataset and return it with an exact version identifier."""
    from datasets import load_dataset, load_from_disk

    local = Path(dataset).expanduser()
    if local.is_dir():
        return load_from_disk(str(local)), hash_directory(local)
    if dataset.startswith(("/", "./", "../", "~")) or dataset.count("/") > 1:
        raise FileNotFoundError(f"Dataset directory {local} does not exist.")

    from huggingface_hub import HfApi

    # Resolve the branch to a commit first, then load exactly that commit.
    commit = HfApi().dataset_info(dataset, revision=revision).sha
    return load_dataset(dataset, revision=commit), commit


def check_splits(splits, spec: TaskSpec) -> None:
    for split in SPLITS:
        if split not in splits:
            raise ValueError(f"Dataset has no {split!r} split; found {list(splits)}")
        columns = splits[split].column_names
        for column in ("text", spec.column):
            if column not in columns:
                raise ValueError(f"Split {split!r} has no {column!r} column; found {columns}")
        values = {int(v) for v in splits[split][spec.column]}
        if not values <= {0, 1}:
            raise ValueError(f"{spec.column} in {split!r} must be 0/1; found {sorted(values)}")


# ---- training ---------------------------------------------------------------------


def _training_arguments(output_dir: Path, recipe: Recipe, use_fp16: bool):
    import inspect

    from transformers import TrainingArguments

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=recipe.epochs,
        learning_rate=recipe.learning_rate,
        per_device_train_batch_size=recipe.train_batch_size,
        per_device_eval_batch_size=recipe.eval_batch_size,
        weight_decay=recipe.weight_decay,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=recipe.seed,
        data_seed=recipe.seed,
        report_to="none",
    )
    # Transformers 5.6 (the project's pin) has warmup_ratio; later versions take
    # the ratio as a fractional warmup_steps.
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["warmup_ratio"] = recipe.warmup_ratio
    else:
        kwargs["warmup_steps"] = recipe.warmup_ratio
    return TrainingArguments(**kwargs)


def _metrics_for_trainer(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    m = classification_metrics([int(x) for x in labels], [int(x) for x in preds])
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "class_1_f1": m["class_1_f1"]}


def _summary(metrics: dict) -> dict:
    keys = ("n", "accuracy", "macro_f1", "macro_precision", "macro_recall", "class_0_f1")
    out = {k: metrics[k] for k in keys}
    out["class_1_f1"] = metrics["class_1_f1"]
    out["confusion_matrix"] = metrics["confusion_matrix"]
    return out


def train(
    *,
    task: str,
    output: Path,
    dataset: str = DEFAULT_DATASET,
    dataset_revision: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    base_revision: str | None = None,
    max_length: int | None = None,
    recipe: Recipe | None = None,
    overwrite: bool = False,
    log=print,
) -> dict:
    """Train, evaluate and save a checkpoint. Returns the training manifest."""
    import torch
    import transformers
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        set_seed,
    )

    spec = TASKS[task]
    recipe = recipe or Recipe()
    max_length = max_length or spec.max_length
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    runs_dir = output.parent / f".{output.name}-runs"

    log(f"Loading dataset {dataset} ...")
    splits, dataset_version = load_splits(dataset, dataset_revision)
    check_splits(splits, spec)
    sizes = {split: len(splits[split]) for split in SPLITS}
    log(f"  version {dataset_version}; sizes {sizes}")

    log(f"Resolving base model {base_model} ...")
    base = resolve_model_source(base_model, base_revision)
    log(f"  version {base.version}")

    tokenizer = AutoTokenizer.from_pretrained(base.path)

    def to_features(split_name: str):
        split = splits[split_name]
        split = split.select_columns(["text", spec.column]).rename_column(spec.column, "labels")
        split = split.map(lambda b: {"labels": [int(v) for v in b["labels"]]}, batched=True)
        return split.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_length, padding=False),
            batched=True,
            remove_columns=["text"],
        )

    features = {split: to_features(split) for split in SPLITS}
    weights = balanced_class_weights(list(features["train"]["labels"]))
    log(f"  class weights {[round(w, 4) for w in weights]}")

    set_seed(recipe.seed)
    id2label = dict(enumerate(spec.labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        base.path,
        num_labels=2,
        id2label=id2label,
        label2id={label: i for i, label in id2label.items()},
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels").long()
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=logits.device))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    use_cuda = torch.cuda.is_available()
    trainer = WeightedTrainer(
        model=model,
        args=_training_arguments(runs_dir, recipe, use_fp16=use_cuda),
        train_dataset=features["train"],
        eval_dataset=features["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_metrics_for_trainer,
    )

    log(f"Training {task} on {'GPU ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")
    started = time.time()
    trainer.train()
    train_seconds = time.time() - started

    # The best checkpoint is loaded at the end of training; evaluate it on
    # validation (the selection set) and then, once, on test.
    def predict(split_name: str) -> tuple[list[int], list[int]]:
        result = trainer.predict(features[split_name])
        truth = result.label_ids.astype(int).tolist()
        return truth, result.predictions.argmax(axis=-1).tolist()

    validation = classification_metrics(*predict("validation"))
    truth, preds = predict("test")
    test = classification_metrics(truth, preds)
    by_variety = {}
    if "variety" in splits["test"].column_names:
        varieties = splits["test"]["variety"]
        for variety in sorted(set(varieties)):
            idx = [i for i, v in enumerate(varieties) if v == variety]
            by_variety[variety] = _summary(
                classification_metrics([truth[i] for i in idx], [preds[i] for i in idx])
            )
    log(f"Validation macro-F1 {validation['macro_f1']:.4f}; test macro-F1 {test['macro_f1']:.4f}")

    # Save the selected model with the truncation length it was trained with.
    trainer.model.save_pretrained(output)
    tokenizer.model_max_length = max_length
    tokenizer.save_pretrained(output)
    write_serving_config(
        output,
        CheckpointServingConfig(
            task=task, max_seq_length=max_length, preprocessing=RAW_TEXT.as_dict()
        ),
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task,
        "labels": list(spec.labels),
        "base_model": {"id": base_model, "version": base.version},
        "dataset": {"id": dataset, "version": dataset_version, "sizes": sizes},
        "recipe": {
            "source": "notebooks/complete_experiments.ipynb, train_roberta (stage Q2.1, pooled)",
            "max_length": max_length,
            "preprocessing": "none (raw text to tokenizer)",
            "loss": "cross-entropy with balanced class weights",
            "class_weights": weights,
            "selection": "best validation macro_f1 over epochs",
            "fp16": use_cuda,
            **asdict(recipe),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        },
        "train_seconds": round(train_seconds, 1),
        "validation_metrics": _summary(validation),
        "test_metrics": _summary(test),
        "test_metrics_by_variety": by_variety,
    }
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(_model_card(manifest), encoding="utf-8")
    shutil.rmtree(runs_dir, ignore_errors=True)
    log(f"Saved to {output}")
    return manifest


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['n']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['class_1_f1']:.4f} |"
    )


def _model_card(manifest: dict) -> str:
    test, val = manifest["test_metrics"], manifest["validation_metrics"]
    rows = "\n".join(
        f"| {variety} | {m['n']} | {m['macro_f1']:.4f} | {m['class_1_f1']:.4f} |"
        for variety, m in manifest["test_metrics_by_variety"].items()
    )
    by_variety = (
        f"\n| Variety | n | Macro-F1 | Class-1 F1 |\n| --- | ---: | ---: | ---: |\n{rows}\n"
        if rows
        else ""
    )
    recipe = manifest["recipe"]
    return f"""---
library_name: transformers
base_model: {manifest["base_model"]["id"]}
datasets:
- {manifest["dataset"]["id"]}
---

# RoBERTa-base {manifest["task"]} classifier (cross-variety English)

Fine-tuned from `{manifest["base_model"]["id"]}` on `{manifest["dataset"]["id"]}`, all
varieties (en-UK, en-AU, en-IN) pooled. Labels: {", ".join(manifest["labels"])}.

| Split | n | Accuracy | Macro-F1 | Class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
{_row("Validation", val)}
{_row("Test", test)}
{by_variety}
Recipe: {recipe["epochs"]} epochs, lr {recipe["learning_rate"]}, batch {recipe["train_batch_size"]},
seed {recipe["seed"]}, class-weighted cross-entropy, checkpoint chosen on validation macro-F1.
Input is raw text truncated at {recipe["max_length"]} tokens. Exact versions and the full recipe
are in `training_manifest.json`; `serving_config.json` tells the serving code how to reproduce
the training preprocessing.
"""


def push_to_hub(output: Path, repo_id: str, private: bool) -> str:
    """Upload the checkpoint and return the commit hash to deploy."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    manifest = json.loads((Path(output) / MANIFEST_FILE).read_text(encoding="utf-8"))
    commit = api.upload_folder(
        folder_path=str(output),
        repo_id=repo_id,
        commit_message=(
            f"{manifest['task']} classifier, seed {manifest['recipe']['seed']}, "
            f"test macro-F1 {manifest['test_metrics']['macro_f1']:.4f}"
        ),
    )
    return commit.oid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=Recipe.seed)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hub id or save_to_disk dir")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision")
    parser.add_argument("--max-length", type=int, help="Default: 256 sentiment, 384 sarcasm")
    parser.add_argument("--epochs", type=int, default=Recipe.epochs)
    parser.add_argument("--learning-rate", type=float, default=Recipe.learning_rate)
    parser.add_argument("--train-batch-size", type=int, default=Recipe.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Recipe.eval_batch_size)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--push-to-hub", metavar="REPO_ID", help="e.g. user/besstie-roberta-sentiment"
    )
    parser.add_argument("--public", action="store_true", help="Make the Hub repo public")
    args = parser.parse_args(argv)

    recipe = Recipe(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    train(
        task=args.task,
        output=args.output,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        base_model=args.base_model,
        base_revision=args.base_revision,
        max_length=args.max_length,
        recipe=recipe,
        overwrite=args.overwrite,
    )
    if args.push_to_hub:
        commit = push_to_hub(args.output, args.push_to_hub, private=not args.public)
        print(f"Uploaded to {args.push_to_hub} at commit {commit}")
        print(f"Serve it with: MODEL_ID={args.push_to_hub} MODEL_REVISION={commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
