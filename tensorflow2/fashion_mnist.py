import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FashinoModel:
    def __init__(self):
        pass

    def execute(self):
        fashion_mnist=keras.datasets.fashion_mnist
        (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print('------------------TRAIN SET SPEC-------------------------')
        print(train_images.shape)
        print(len(train_images.shape))
        print(train_labels.shape)
        print(len(train_labels.shape))
        print('------------------TEST SET SPEC-------------------------')
        print(test_images.shape)
        print(len(test_images.shape))
        print(test_labels.shape)
        print(len(test_labels.shape))


if __name__ == '__main__':
    f= FashinoModel()
    f.execute()