import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow import keras
import time

from IPython import display
class Dcgan:
    def __init__(self):
        (self.train_images, self.train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        self.train_images = self.train_images.reshape(self.train_images.shape[0], 28, 28, 1).astype('float32')
        self.train_images = (self.train_images - 127.5) / 127.5  # 이미지를 [-1, 1]로 정규화합니다.
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        self.generator =None
        self.discriminator=None
        self.generated_image=None
        self.cross_entropy=None
        self.discriminator_optimizer=None
        self.generator_optimizer=None
        self.EPOCHS = 50
        self.noise_dim = 100
        self.num_examples_to_generate = 16

        # 이 시드를 시간이 지나도 재활용하겠습니다.
        # (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문입니다.)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
    def execute(self):
        self.make_generator_model()
        self.make_discriminator_model()
        self.loss()

        self.train(self.train_dataset, self.EPOCHS)
        self.generate_and_save_images()
        self.display_image(self.EPOCHS)
    '''
    생성자는 시드값 (seed; 랜덤한 잡음)으로부터 이미지를 생성하기 위해, tf.keras.layers.Conv2DTranspose (업샘플링) 층을 이용합니다. 
    처음 Dense층은 이 시드값을 인풋으로 받습니다. 그 다음 원하는 사이즈 28x28x1의 이미지가 나오도록 업샘플링을 여러번 합니다. 
    tanh를 사용하는 마지막 층을 제외한 나머지 각 층마다 활성함수로 tf.keras.layers.LeakyReLU을 사용하고 있음을 주목합시다
    
    '''
    def make_generator_model(self):
        self.generator = tf.keras.Sequential()
        self.generator.add(keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        self.generator.add(keras.layers.BatchNormalization())
        self.generator.add(keras.layers.LeakyReLU())

        self.generator.add(keras.layers.Reshape((7, 7, 256)))
        assert self.generator.output_shape == (None, 7, 7, 256)  # 주목: 배치사이즈로 None이 주어집니다.

        self.generator.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 7, 7, 128)
        self.generator.add(keras.layers.BatchNormalization())
        self.generator.add(keras.layers.LeakyReLU())

        self.generator.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 14, 14, 64)
        self.generator.add(keras.layers.BatchNormalization())
        self.generator.add(keras.layers.LeakyReLU())

        self.generator.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.generator.output_shape == (None, 28, 28, 1)

        #(아직 훈련이 되지않은)        생성자를        이용해        이미지를        생성해봅시다.
        noise = tf.random.normal([1, 100])
        self.generated_image = self.generator(noise, training=False)
        plt.imshow(self.generated_image[0, :, :, 0], cmap='gray')
    #감별자는 합성곱 신경망(Convolutional Neural Network, CNN) 기반의 이미지 분류기입니다.
    def make_discriminator_model(self):
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        self.discriminator.add(keras.layers.LeakyReLU())
        self.discriminator.add(keras.layers.Dropout(0.3))

        self.discriminator.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(keras.layers.LeakyReLU())
        self.discriminator.add(keras.layers.Dropout(0.3))

        self.discriminator.add(keras.layers.Flatten())
        self.discriminator.add(keras.layers.Dense(1))
        '''
        (아직까지 훈련이 되지 않은) 감별자를 사용하여, 생성된 이미지가 진짜인지 가짜인지 판별합니다. 
        모델은 진짜 이미지에는 양수의 값 (positive values)을, 가짜 이미지에는 음수의 값 (negative values)을 출력하도록 훈련되어집니다.
        '''
        decision = self.discriminator(self.generated_image)
        print(decision)

    #두    모델의    손실함수와    옵티마이저를    정의합니다.
    def loss(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    '''
    이 메서드는 감별자가 가짜 이미지에서 얼마나 진짜 이미지를 잘 판별하는지 수치화합니다. 진짜 이미지에 대한 감별자의 예측과 1로 이루어진 행렬을 비교하고, 가짜 (생성된) 이미지에 대한 감별자의 예측과 0으로 이루어진 행렬을 비교합니다.
    '''

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    '''
    생성자의 손실함수는 감별자를 얼마나 잘 속였는지에 대해 수치화를 합니다. 직관적으로 생성자가 원활히 수행되고 있다면, 감별자는 가짜 이미지를 진짜 (또는 1)로 분류를 할 것입니다. 여기서 우리는 생성된 이미지에 대한 감별자의 결정을 1로 이루어진 행렬과 비교를 할 것입니다.
    '''

    def generator_loss(self,fake_output):
        self.cross_entropy(tf.ones_like(fake_output), fake_output)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # `tf.function`이 어떻게 사용되는지 주목해 주세요.
    # 이 데코레이터는 함수를 "컴파일"합니다.
    @tf.function
    def train_step(self,images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self,dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # GIF를 위한 이미지를 바로 생성합니다.
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator,
                                     self.epoch + 1,
                                     self.seed)

            # 15 에포크가 지날 때마다 모델을 저장합니다.

            # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # 마지막 에포크가 끝난 후 생성합니다.
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator,
                                 epochs,
                                 self.seed)

    def generate_and_save_images(self,model, epoch, test_input):
        # `training`이 False로 맞춰진 것을 주목하세요.
        # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다.
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    # 에포크 숫자를 사용하여 하나의 이미지를 보여줍니다.
    def display_image(self,epoch_no):
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
if __name__ == '__main__':
    d=Dcgan()
    d.execute()