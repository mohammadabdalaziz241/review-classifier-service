# Local model directories

Put a checkpoint directory here to bake it into the Docker image instead of
downloading one from the Hugging Face Hub, then build with
`MODEL_ID=/opt/models/<directory>`:

```
models/
  my-checkpoint/
    config.json
    model.safetensors
    tokenizer.json
    ...
```

The service reports a local model's version as the SHA-256 of its files.
Everything here except this README is ignored by git.
