import os
import tensorflow as tf
from tensorflow import keras
import time

#https://www.tensorflow.org/tutorials/keras/save_and_load

class SaveLoad:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

        self.train_labels = self.train_labels[:1000]
        self.test_labels = self.test_labels[:1000]

        self.train_images = self.train_images[:1000].reshape(-1, 28 * 28) / 255.0
        self.test_images = self.test_images[:1000].reshape(-1, 28 * 28) / 255.0

        self.model=None

        self.checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.saved_model_path = "./saved_models/{}".format(int(time.time()))
    def execute(self):
        self.create_model()
        #self.train_model()
        self.train2_model()
        #self.eval_model()
        #self.load_weight()
        print('------------모델 저장------------')
        self.save_model()
        print('---------모델 복원후 평가하기------------')
        self.eval_model_after_load_model()
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

    def train2_model(self):


        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1, save_weights_only=True,
            # 다섯 번째 에포크마다 가중치를 저장합니다
            period=5)
        self.model.save_weights(self.checkpoint_path.format(epoch=0))
        self.model.fit(self.train_images, self.train_labels,
                  epochs=50, callbacks=[cp_callback],
                  validation_data=(self.test_images, self.test_labels),
                  verbose=0)

    def eval_model(self):
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("훈련된 모델의 정확도: {:5.2f}%".format(100 * acc))

    def load_weight(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(latest)
        self.model.load_weights(latest)
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

    def save_model(self):


        #tf.keras.experimental.export_saved_model(self.model, self.saved_model_path)
        self.model.save('./saved_models/my_model.h5')
        print(self.saved_model_path)
    def load_model(self):
        #new_model = tf.keras.experimental.load_from_saved_model(self.saved_model_path)
        #new_model.summary()
        new_model = keras.models.load_model('./saved_models/my_model.h5')
        new_model.summary()

        return new_model

    def eval_model_after_load_model(self):
        new_model = self.load_model()
        new_model.compile(optimizer=self.model.optimizer,  # 복원된 옵티마이저를 사용합니다.
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

        # 복원된 모델을 평가합니다
        loss, acc = new_model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    s = SaveLoad()
    s.execute()