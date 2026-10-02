# Review Classifier Service

A FastAPI inference service for sentiment and sarcasm classification of English reviews,
built on the models from my NLP group project,
[cross-variety-sentiment-sarcasm](https://github.com/mohammadabdalaziz241/cross-variety-sentiment-sarcasm)
(classical baselines, RoBERTa fine-tuning and Gemma-2-2B LoRA adapters, evaluated on the
BESSTIE-CW-26 dataset across British, Australian and Indian English).

The research already exists. This repository turns the strongest efficient model, pooled
RoBERTa-base, into a service: a reproducible training script that keeps the model, a serving
path proven to reproduce the training metrics, input validation, consistent errors, exact
model versioning and tests. Later milestones add containers, a database, cloud deployment,
CI/CD and measured performance.

> **Status:** milestones 1 and 1b done: a tested local service serving the project's own
> retrained RoBERTa models, with served test metrics verified against training
> ([results](#results)). Nothing here is deployed yet. See the [roadmap](#roadmap).

> **Deployment:** the RoBERTa models from this study are retrained, published on the
> Hugging Face Hub, and served by a tested FastAPI service in
> [review-classifier-service](https://github.com/mohammadabdalaziz241/review-classifier-service),
> which verifies that served predictions reproduce the test metrics.

## Results

Pooled RoBERTa-base retrained with the project notebook's recipe (seed 42, one Colab T4,
about 3 minutes per model), then served by this service and evaluated through its HTTP API on
the full BESSTIE-CW-26 test split (2,183 texts):

| Model | Test macro-F1 (training) | Test macro-F1 (served) | Original notebook, seed 42 | Class-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| Sentiment | 0.8982 | 0.8987 | 0.9028 | 0.8952 |
| Sarcasm | 0.6984 | 0.6984 | 0.7001 | 0.5086 |

- **Serving reproduces training.** Sarcasm predictions are identical. For sentiment, one of
  2,183 predictions differs (fp16 evaluation during training vs fp32 serving), well inside
  the ±0.005 tolerance.
- **The published models were verified too.** The same check passed on a service loading each
  model from the Hugging Face Hub at its pinned commit, as a deployment will.
- **The retrained models are close to the notebook's** (−0.005 and −0.002 macro-F1), within
  the run-to-run variation the notebook reported across seeds.
- **Indian English remains hardest**, as the project found: sentiment macro-F1 is 0.847 on
  en-IN vs 0.904 en-AU and 0.953 en-UK, and sarcastic-class F1 is 0.22 on en-IN.

Full manifests (exact dataset and base-model commits, recipe, environment, per-variety
metrics) and evaluation reports are in [`results/`](results/).

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/mohammadabdalaziz241/review-classifier-service.git
cd review-classifier-service
python -m venv .venv && source .venv/bin/activate

# Option A: run the API with the keyword baseline (no torch, no download)
pip install -e ".[dev]"
MODEL_BACKEND=dummy python -m review_classifier

# Option B: run a real transformers model on CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,hf]"
python -m review_classifier            # downloads the default model on first start
```

Then open <http://127.0.0.1:8000/docs> for the interactive OpenAPI docs.

The `dummy` backend is a deterministic keyword rule that exists so the API and its tests run
without a model. It is not a sentiment model, and the service logs a warning when it is used.

**The default `hf` model is not one of this project's models.** It is Cardiff NLP's public
Twitter sentiment classifier (negative / neutral / positive), there so the service works for
anyone who clones the repository. To serve the project's own models, pin them exactly:

```bash
# Option C: the project's retrained models (private repos: needs HF_TOKEN with read access)
export HF_TOKEN=<your token>
MODEL_ID=Mohammadeeu20/besstie-roberta-sentiment \
MODEL_REVISION=43da49621dc65cdd7cd11f77c8c2e7eb13d06cbb python -m review_classifier

MODEL_ID=Mohammadeeu20/besstie-roberta-sarcasm \
MODEL_REVISION=dedc10bff5817a4a84f0500150b85b99a4225ecb python -m review_classifier
```

## Train the project's model

The project notebook evaluated RoBERTa but deleted every checkpoint after testing, so there
is no saved model to serve. `review_classifier.train` reruns the notebook's `train_roberta`
recipe (stage Q2.1, all varieties pooled) unchanged and keeps the result:

| | Notebook | This script |
| --- | --- | --- |
| Model, data | `roberta-base`, BESSTIE-CW-26, varieties pooled | same (`FacebookAI/roberta-base`, its current Hub name) |
| Input | raw text, truncated at 256 (sentiment) / 384 (sarcasm) tokens | same |
| Loss | cross-entropy, balanced class weights | same |
| Schedule | 4 epochs, lr 2e-5, batch 16/32, 10% warmup, wd 0.01, fp16 | same |
| Selection | best validation macro-F1; test evaluated once | same |
| Checkpoint | deleted | saved, with `serving_config.json` and `training_manifest.json` |
| Labels | `class_0` / `class_1` | `negative`/`positive`, `not_sarcastic`/`sarcastic` |

It needs a GPU; a free Colab T4 is enough. The easiest route is
[`notebooks/train_on_colab.ipynb`](notebooks/train_on_colab.ipynb): upload it to
[Colab](https://colab.research.google.com), run the cells in order, and it trains, serves the
new checkpoint, [verifies it](#verify-the-served-model), uploads it to the Hub only if the
check passes, verifies the uploaded copy, and downloads a small results zip.

From a shell on any GPU machine instead:

```bash
pip install -e ".[hf,train]"
hf auth login             # a token with write access, to download the dataset and upload the model
python -m review_classifier.train --task sentiment --seed 42 \
    --output checkpoints/roberta-sentiment \
    --push-to-hub <hf-user>/besstie-roberta-sentiment
```

It prints the validation and test metrics, then the exact command to serve the upload:

```
Uploaded to <hf-user>/besstie-roberta-sentiment at commit 3f9c…
Serve it with: MODEL_ID=<hf-user>/besstie-roberta-sentiment MODEL_REVISION=3f9c…
```

The Hub repo is private unless you pass `--public`. Use `--task sarcasm` for the sarcasm
model. For reference, the notebook's pooled RoBERTa test macro-F1 was 0.9028 (sentiment)
and 0.7001 (sarcasm) with seed 42. A retrained model should land close to these, though not
exactly, because GPU training is not bit-for-bit deterministic and library versions differ.

Every checkpoint carries:

- `training_manifest.json`: the base model and dataset at exact commits, the full recipe
  and class weights, the environment, and validation, test and per-variety test metrics
- `serving_config.json`: the serving contract (below)
- `README.md`: a model card with the results

## Verify the served model

Training metrics only mean something for the service if the service reproduces them.
`review_classifier.evaluate` sends the test split through the running API, exactly as a
client would, and compares the result with the manifest:

```bash
MODEL_ID=<hf-user>/besstie-roberta-sentiment MODEL_REVISION=<commit> python -m review_classifier &
python -m review_classifier.evaluate \
    --manifest checkpoints/roberta-sentiment/training_manifest.json --report eval.json
```

```
Model    <hf-user>/besstie-roberta-sentiment @ 3f9c…
Preproc  none (raw text)
         0 of N texts changed by preprocessing
Served   n=… accuracy=… macro_f1=…
Training accuracy=… macro_f1=…
Result   MATCH (tolerance ±0.005)  macro_f1 +0.0000  accuracy +0.0000
```

It evaluates the same dataset commit the model was trained on, checks that the service's
labels are in training order, counts texts the service's preprocessing would change, and
exits non-zero on a mismatch. A gap points to a difference in preprocessing, tokenisation,
truncation or label order. If it reports texts over `MAX_TEXT_CHARS`, restart the service
with a higher limit; the model truncates by tokens anyway.

## API

| Method | Path                | Purpose                                                   |
| ------ | ------------------- | --------------------------------------------------------- |
| GET    | `/health`           | Liveness: the process is serving HTTP                     |
| GET    | `/ready`            | Readiness: the model is loaded (`503` until it is)        |
| GET    | `/v1/model`         | Model id, exact version, labels, serving contract, limits |
| POST   | `/v1/predict`       | Classify one text                                         |
| POST   | `/v1/predict/batch` | Classify up to `MAX_BATCH_SIZE` texts, order preserved    |

```bash
curl -s localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Proper brilliant, would buy again"}'
```

```json
{
  "request_id": "5f0c9a1e2b7d4c3a9e8f6d1b2a3c4d5e",
  "model": {"id": "<hf-user>/besstie-roberta-sentiment", "version": "<commit hash>"},
  "prediction": {
    "label": "positive",
    "score": 0.97,
    "scores": {"negative": 0.03, "positive": 0.97}
  },
  "inference_ms": 41.2
}
```

The scores above are illustrative. Every response records which model **and which exact
version** produced it, so a prediction can always be traced back to the weights that made it.

### Model versions

`version` always identifies the loaded files, never a branch name or a placeholder:

| Model source     | Reported `version`                         | How it is guaranteed |
| ---------------- | ------------------------------------------ | -------------------- |
| Hugging Face Hub | the resolved 40-character commit hash      | The model is downloaded into a snapshot folder named after that commit and loaded from that folder, never by name |
| Local directory  | `sha256:` hash of every file's path and contents | Any change to weights, config or tokenizer files changes the hash |

`GET /v1/model` also returns `requested_revision`, what was asked for (for example `main`),
so you can see that `main` resolved to a specific commit at startup.

### The serving contract

A checkpoint's `serving_config.json` states how it must be served:

```json
{
  "schema_version": 1,
  "task": "sentiment",
  "max_seq_length": 256,
  "preprocessing": {
    "normalize_unicode": false,
    "normalize_whitespace": false,
    "replace_urls": false,
    "replace_mentions": false
  }
}
```

The service applies it automatically, so serving matches training without settings being
copied by hand. Each value resolves as **environment variable > checkpoint > built-in
default**. An environment variable that contradicts the checkpoint still wins, for
experiments, but is logged as a warning because predictions will no longer be comparable
with the training metrics. `/v1/model` and the startup log show every effective value and
where it came from:

```
serving task=sentiment max_seq_length=256 preprocessing=normalize_unicode=false(checkpoint) …
```

A checkpoint without the file (such as the default model) gets the built-in defaults and a
warning to confirm they match its training. A malformed file stops the service at startup.

`task` is only a description. It never changes what the model predicts: the labels come from
the model's own `id2label`.

### Errors

All errors, including validation failures, unknown routes and unexpected exceptions, share
one envelope with a stable `code`:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request body is invalid.",
    "details": [
      {"loc": ["body", "texts", 1], "msg": "Text is empty after normalisation.", "type": "text_empty"}
    ]
  },
  "request_id": "client-req-42"
}
```

| Code                 | Status | When                                                    |
| -------------------- | ------ | ------------------------------------------------------- |
| `invalid_request`    | 422    | Malformed JSON, wrong types, unknown fields, blank or over-long text |
| `batch_too_large`    | 422    | More texts than `MAX_BATCH_SIZE`                        |
| `model_not_ready`    | 503    | Request arrived before the model finished loading       |
| `not_found`          | 404    | Unknown route                                           |
| `method_not_allowed` | 405    | Wrong HTTP method                                       |
| `internal_error`     | 500    | Unexpected failure; details are logged, never returned  |

A batch with several bad items reports all of them, by index, in one response.

### Request IDs

Send `X-Request-ID` to correlate a call with your own logs; otherwise one is generated. It
is returned in the response header and body, and included in the server's log line for the
request. IDs that are not 1–128 characters of `[A-Za-z0-9._-]` are replaced, so client
input cannot inject content into logs. `X-Process-Time-Ms` reports server-side time.

## Configuration

All settings are environment variables, validated at startup.

| Variable               | Default                                             | Notes |
| ---------------------- | --------------------------------------------------- | ----- |
| `MODEL_BACKEND`        | `hf`                                                | `hf` or `dummy` |
| `MODEL_ID`             | `cardiffnlp/twitter-roberta-base-sentiment-latest`  | Hub id or local directory |
| `MODEL_REVISION`       | unset (`main`)                                      | Branch, tag or commit; pin a commit so every deployment loads the same weights |
| `HF_TOKEN`             | unset                                               | Needed for private Hub repositories |
| `HF_HUB_OFFLINE`       | unset                                               | `1` loads from the local Hugging Face cache only; pin `MODEL_REVISION` to a commit |
| `DEVICE`               | `auto`                                              | `auto`, `cpu`, `cuda` or `mps` |
| `MAX_TEXT_CHARS`       | `2000`                                              | Per-text limit, checked before tokenisation |
| `MAX_BATCH_SIZE`       | `32`                                                | Texts per batch request |
| `INFERENCE_BATCH_SIZE` | `16`                                                | Texts per forward pass |
| `HOST` / `PORT`        | `127.0.0.1` / `8000`                                |  |
| `LOG_LEVEL`            | `info`                                              |  |

These override the [serving contract](#the-serving-contract); leave them unset to use the
checkpoint's values:

| Variable                          | Fallback when the checkpoint is silent | Effect when on |
| --------------------------------- | -------------------------------------- | -------------- |
| `MODEL_TASK`                      | `unspecified`                          | Descriptive label |
| `MAX_SEQ_LENGTH`                  | the tokenizer's limit (512 if it has none) | Truncation length in tokens |
| `PREPROCESS_NORMALIZE_UNICODE`    | `true`                                 | Unicode NFKC normalisation |
| `PREPROCESS_NORMALIZE_WHITESPACE` | `true`                                 | Collapse whitespace runs, trim ends |
| `PREPROCESS_REPLACE_URLS`         | `true`                                 | URLs become `http` |
| `PREPROCESS_REPLACE_MENTIONS`     | `true`                                 | `@name` becomes `@user` |

Control characters are always removed. With every step off, ordinary text reaches the
tokenizer unchanged, which is what the project's RoBERTa models were trained on.

## Design decisions

- **Train for serving.** The training script is the notebook's recipe, not a new one, so
  the served model is the model the project evaluated. What it adds is everything serving
  needs: the weights, real label names, exact data and model versions, and the contract.
- **Prove the serving path, don't assume it.** The evaluation runs the test split through
  the HTTP API and compares with the training metrics, so a preprocessing, tokenisation or
  label-order mismatch shows up as a number instead of silently costing accuracy.
- **The checkpoint carries its own contract.** Preprocessing and truncation travel with the
  weights, so they cannot drift apart between training and deployment.
- **Fail fast.** Invalid configuration, a malformed contract or a model that will not load
  stops the process at startup. A container that cannot serve should fail its deployment,
  not answer every request with an error.
- **Versions identify weights, not names.** A branch can move after deployment, so the
  service resolves it to a commit once and loads exactly that snapshot. Local models are
  content-hashed.
- **The HTTP layer depends on an interface, not on torch.** Routes use a small `Predictor`
  protocol. Tests inject fakes, and an ONNX or quantised backend can be added without
  changing the API.
- **Limits are enforced before tokenisation**, so oversized input is rejected cheaply.
- **Inference runs off the event loop.** Inference routes are sync functions, which FastAPI
  runs in a worker thread, so `/health` stays responsive during a slow forward pass.
- **Review text is never logged**, because reviews can contain personal data. Logs contain
  request IDs, paths, status codes and timings only.

## Tests

```bash
pytest -m "not hf"     # API, validation, preprocessing, config, metrics, evaluation — no torch
pytest -m hf           # real transformers code paths; needs the [hf,train] extras
ruff check . && ruff format --check .
```

No test needs network access. The `hf` tests build a tiny BERT classifier and a synthetic
dataset with BESSTIE-CW-26's columns, then run the real **train → serve → evaluate** loop on
CPU in seconds. They check that the served test predictions reproduce the training metrics
exactly (same confusion matrix), that the checkpoint's contract is applied, that an
overriding environment variable is visible in the evaluation, and that real model loading,
batching, padding, truncation, version hashing and startup failures behave correctly. The
metrics are checked against scikit-learn, which the notebook used.

CI (GitHub Actions) runs lint, the fast suite on Python 3.10–3.12, and the model tests on
CPU-only torch.

## Project layout

```
src/review_classifier/
  api.py             routes, validation, error envelope, request IDs, access log
  config.py          environment-based settings with validation
  serving_config.py  the checkpoint serving contract and how settings resolve
  model_source.py    resolves a model to files on disk and an exact version
  predictors.py      Predictor protocol, transformers backend, dummy backend
  preprocessing.py   text preprocessing
  schemas.py         request and response models (the API contract)
  metrics.py         macro-F1 and friends, shared by training and evaluation
  train.py           `python -m review_classifier.train`: the notebook recipe, kept
  evaluate.py        `python -m review_classifier.evaluate`: served vs training metrics
  __main__.py        `python -m review_classifier`: start the service
notebooks/
  train_on_colab.ipynb  train, verify and publish on a free Colab GPU
results/             manifests and evaluation reports of the published models
tests/               unit and integration tests
```

## Roadmap

- [x] **1. Local service** — FastAPI endpoints, validation, error handling, tests, CI
- [x] **1b. Own model** — pooled RoBERTa sentiment and sarcasm models retrained, published
      at pinned commits, and served metrics verified against training
- [ ] **2. Packaging** — Dockerfile, Docker Compose for local development
- [ ] **3. Persistence** — PostgreSQL for inference metadata and optional labelled feedback
- [ ] **4. Cloud** — AWS deployment with storage, container hosting, IAM and logging
- [ ] **5. Operations** — release pipeline, metrics, and a reproducible benchmark of
      latency, throughput, memory and error rate, with the conditions stated
- [ ] Later: the per-variety Gemma-2-2B LoRA sarcasm adapters with adapter switching,
      carried over from the Gradio app (needs a CUDA GPU)
