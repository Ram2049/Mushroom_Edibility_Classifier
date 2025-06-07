import tensorflow as tf
from pathlib import Path
from Mushroom_edibility_classifier.entity.config_entity import EvaluationConfig
from Mushroom_edibility_classifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _test_generator(self):
        datagen_kwargs = dict(rescale=1. / 255)
        flow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical',
            shuffle=False
        )

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        self.test_generator = test_datagen.flow_from_directory(
            directory=self.config.test_data,
            **flow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluate(self):
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
