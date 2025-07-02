from opensoundscape import CNN
from opensoundscape.ml.cnn_architectures import resnet18
import torch
from bioacoustics_model_zoo.utils import register_bmz_model


@register_bmz_model
class RanaSierraeCNNOffline(CNN):
    def __init__(self, model_path):
        """RanaSierraeCNN for offline Kaggle use"""

        # initialize resnet with random weights, since we will load pre-trained weights
        arch = resnet18(num_classes=2, weights=None)
        super().__init__(
            architecture=arch,
            classes=["rana_sierrae", "negative"],
            sample_duration=2.0,
            single_target=True,
            channels=3,
        )

        # modify preprocessing of the CNN:
        # bandpass spectrograms to 300-2000 Hz
        self.preprocessor.pipeline.bandpass.set(min_f=300, max_f=2000)

        # use legacy interpolation mode
        self.preprocessor.pipeline.to_tensor.set(use_skimage=True)

        # modify augmentation routine parameters
        self.preprocessor.pipeline.frequency_mask.set(max_masks=5, max_width=0.1)
        self.preprocessor.pipeline.time_mask.set(max_masks=5, max_width=0.1)
        self.preprocessor.pipeline.add_noise.set(std=0.01)

        # decrease the learning rate from the default value
        self.optimizer_params["lr"] = 0.002

        ## Load pre-trained weights ##
        self.network.load_state_dict(
            torch.load(model_path)
        )