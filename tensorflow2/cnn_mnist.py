import tensorflow as tf
from tensorflow import keras
import time
class CnnMnist:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

        self.train_images = self.train_images.reshape((60000, 28, 28, 1))
        self.test_images = self.test_images.reshape((10000, 28, 28, 1))

        # 픽셀 값을 0~1 사이로 정규화합니다.
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
        self.model=None
        self.saved_model_path = "./saved_models/{}".format(int(time.time()))
    def execute(self):
        self.create_model()
        self.train_model()
        #self.eval_model()
        self.save_model()
        self.eval_model_after_load_model()
    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        print(self.model.summary())
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    def train_model(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5)

    def eval_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print(test_loss,test_acc)

    def save_model(self):


        tf.keras.experimental.export_saved_model(self.model, self.saved_model_path)
        print(self.saved_model_path)
    def load_model(self):
        new_model = tf.keras.experimental.load_from_saved_model(self.saved_model_path)
        new_model.summary()

        return new_model

    def eval_model_after_load_model(self):
        new_model = self.load_model()
        new_model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # 복원된 모델을 평가합니다
        loss, acc = new_model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))

if __name__ == '__main__':
    c = CnnMnist()
    c.execute()