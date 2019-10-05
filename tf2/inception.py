import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

tf.app.flags.DEFINE_string('output_graph',
                           "./data/flower.pb",
                           "학습된 신경감이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels",
                           "./data/label.txt",
                           "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image",
                            True,
                            "이미지 추론 후 이미지를 보여줍니다.")
FLAGS = tf.app.flags.FLAGS

def main(_):
    pass

if __name__ == '__main__':
    pass