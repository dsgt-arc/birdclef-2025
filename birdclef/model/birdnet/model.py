import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score


class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        learning_rate: float = 0.002,
        **kwargs,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.learning_rate = 0.002

        self.save_hyperparameters()
        self.loss = nn.BCEWithLogitsLoss()

        # normalize the input features, then do a linear regression
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, num_labels),
        )

        self.f1_score = MultilabelF1Score(num_labels=num_labels, average="macro")
        self.auroc_score = MultilabelAUROC(num_labels=num_labels, average="weighted")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        # sigmoid the label and apply a threshold
        logit = batch["label"]
        logits_pred = self(batch["features"])

        loss = self.loss(logits_pred, logit)
        f1_score = self.f1_score(logits_pred, logit)
        auroc_score = self.auroc_score(logits_pred, logit.to(torch.int))

        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(f"{step_name}_f1", f1_score, on_step=False, on_epoch=True)
        self.log(f"{step_name}_auroc", auroc_score, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        batch["prediction"] = torch.sigmoid(self(batch["embedding"]))
        return batch
