from tqdm.auto import tqdm
from gensim.models.callbacks import CallbackAny2Vec


class TqdmCallback(CallbackAny2Vec):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_count = 0
        self.pbar = None

    def on_epoch_begin(self, model):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_epochs, desc="Epochs")
        self.epoch_count += 1
        self.pbar.set_description(
            f"Training Epoch {self.epoch_count}/{self.total_epochs}"
        )

    def on_epoch_end(self, model):
        if self.pbar is not None:
            self.pbar.update(1)
        current_loss = model.get_latest_training_loss()
        if current_loss is not None:
            self.pbar.set_postfix_str(f"Loss: {current_loss:.4f}", refresh=True)

    def on_train_end(self, model):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
