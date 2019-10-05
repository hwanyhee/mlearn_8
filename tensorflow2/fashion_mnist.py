import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FashinoModel:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def execute(self):
        fashion_mnist=keras.datasets.fashion_mnist
        (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

        print('------------------TRAIN SET SPEC-------------------------')
        print(train_images.shape)
        print(len(train_images.shape))
        print(train_labels.shape)
        print(len(train_labels.shape))
        print(train_images[0])
        print('------------------TEST SET SPEC-------------------------')
        print(test_images.shape)
        print(len(test_images.shape))
        print(test_labels.shape)
        print(len(test_labels.shape))
        print(test_images[0])
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[train_labels[i]])
        plt.show()
        #-------------모델 구성------------------
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        #----------훈련--------------------
        model.fit(train_images, train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        print('\n테스트 정확도:', test_acc)
        #---------예측하기---------------------
        predictions = model.predict(test_images)
        print('예측값 : %s' % predictions[0])
        print('가장 신뢰도가 높은 레이블: %s' % np.argmax(predictions[0]))
        #---- 태스트
        print('테스트: %s' % test_labels[0])

        # 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
        # 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions, test_labels)
        plt.show()
        # 테스트 세트에서 이미지 하나를 선택합니다
        img = test_images[0]
        print(img.shape)
        # 이미지 하나만 사용할 때도 배치에 추가합니다
        img = (np.expand_dims(img, 0))
        print(img.shape)
        predictions_single = model.predict(img)
        print(predictions_single)

        self.plot_value_array(0, predictions_single, test_labels)
        _ = plt.xticks(range(10), self.class_names, rotation=45)

        print(np.argmax(predictions_single[0]))

    def plot_image(self,i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)

    def plot_value_array(self,i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    f= FashinoModel()
    f.execute()