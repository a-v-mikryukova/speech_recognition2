import wandb


class WanDBLogger:
    def __init__(self, config) -> None:
        self.config = config
        self.run = wandb.init(
            project="asr",
            config=config,
        )

    def log_metrics(self, metrics, step) -> None:
        wandb.log(metrics, step=step)

    def watch_model(self, model) -> None:
        wandb.watch(model, log="all", log_freq=100)

    def log_checkpoint(self, path) -> None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)
