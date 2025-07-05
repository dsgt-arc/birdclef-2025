from opensoundscape import SpectrogramClassifier
from bioacoustics_model_zoo.bmz_birdset.bmz_birdset_convnext import ConvNextBirdsetPreprocessor, ConvNextForImageClassificationLogits
import bioacoustics_model_zoo as bmz

@bmz.register_bmz_model
class BirdSetConvNeXTOffline(SpectrogramClassifier):
    def __init__(self, model_path):
        """BirdSetConvNeXT for offline Kaggle use"""

        model = ConvNextForImageClassificationLogits.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
        )
        classes = [model.config.id2label[i] for i in range(model.num_labels)]

        super().__init__(model, classes=classes, sample_duration=5)

        self.preprocessor = ConvNextBirdsetPreprocessor()
        self.network.to(self.device)

        self.network.classifier_layer = "classifier"
        self.network.embedding_layer = "convnext.layernorm"
        self.network.cam_layer = "convnext.encoder.stages.2.layers.26"