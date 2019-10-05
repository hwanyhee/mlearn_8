import os
import tensorflow as tf
from tensorflow import keras

class SaveLoad:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

        self.train_labels = self.train_labels[:1000]
        self.test_labels = self.test_labels[:1000]

        self.train_images = self.train_images[:1000].reshape(-1, 28 * 28) / 255.0
        self.test_images = self.test_images[:1000].reshape(-1, 28 * 28) / 255.0

        self.model=None
    def execute(self):
        self.create_model()
        self.train_model()
        self.eval_model()
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.summary()

    def train_model(self):
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # 체크포인트 콜백 만들기
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        self.model.fit(self.train_images, self.train_labels, epochs=10,
                  validation_data=(self.test_images, self.test_labels),
                  callbacks=[cp_callback])

    def eval_model(self):
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("훈련된 모델의 정확도: {:5.2f}%".format(100 * acc))
if __name__ == '__main__':
    s = SaveLoad()
    s.execute()