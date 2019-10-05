import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


class ImdnModel:
    def __init__(self):
        self.train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
        (self.train_data, self.validation_data), self.test_data = tfds.load(name="imdb_reviews",split=(self.train_validation_split, tfds.Split.TEST),as_supervised=True)
        self.model=None
        self.train_examples_batch, self.train_labels_batch = next(iter(self.train_data.batch(10)))
        print(self.train_examples_batch)
    def execute(self):
        self.create_model()
        self.train_model()
        self.evaluate()
    def create_model(self):
        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
        hub_layer(self.train_examples_batch[:3])
        self.model = tf.keras.Sequential()
        self.model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.summary()

        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def train_model(self):
        history = self.model.fit(self.train_data.shuffle(10000).batch(512),
                                 epochs=20,
                                 validation_data=self.validation_data.batch(512),
                                 verbose=1)
        print('history:',history)
    def evaluate(self):
        results = self.model.evaluate(self.test_data.batch(512), verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print("%s: %.3f" % (name, value))

    @staticmethod
    def env_info():
        print("버전: ", tf.__version__)
        print("즉시 실행 모드: ", tf.executing_eagerly())
        print("허브 버전: ", hub.__version__)
        print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")
if __name__ == '__main__':
    ImdnModel.env_info()
    m = ImdnModel()
    m.execute()
