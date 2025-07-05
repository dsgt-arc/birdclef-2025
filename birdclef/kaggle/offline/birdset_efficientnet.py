from opensoundscape import SpectrogramClassifier
from bioacoustics_model_zoo.bmz_birdset.bmz_birdset_efficientnetB1 import EfficientnetBirdsetPreprocessor, EfficientNetLogits
import bioacoustics_model_zoo as bmz

@bmz.register_bmz_model
class BirdSetEfficientNetB1Offline(SpectrogramClassifier):
    def __init__(self, model_path):
        """BirdSetEfficientNetB1 for offline Kaggle use"""

        model = EfficientNetLogits.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
        )
        classes = [model.config.id2label[i] for i in range(model.num_labels)]

        super().__init__(model, classes=classes, sample_duration=5)

        self.preprocessor = EfficientnetBirdsetPreprocessor()
        self.network.to(self.device)

        self.network.classifier_layer = "classifier"
        self.network.embedding_layer = "efficientnet.pooler"
        self.network.cam_layer = "efficientnet.encoder.blocks.1"