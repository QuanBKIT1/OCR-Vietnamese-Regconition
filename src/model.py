import tensorflow as tf
import keras
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, Dense, LSTM

class Squeeze(keras.layers.Layer):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis


    def call(self, inputs):
        return tf.squeeze(inputs, self.axis, name='squeeze_layer')

def construct_model(input_dim, output_dim, activation="leaky_relu"):

    inputs = Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = Lambda(lambda x: x / 255)(inputs)

    x = Conv2D(filters=64, kernel_size=(3,3),strides=1,activation=activation,padding='valid')(input)
    x = Conv2D(filters=64, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = Conv2D(filters=128, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,1))(x)
    x = Conv2D(filters=256, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,1))(x)
    x = Conv2D(filters=512, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3,3),strides=1,activation=activation,padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(3,1))(x)

    squeezed = Squeeze(axis=1)(x)
    # squeezed = tf.squeeze(x, axis=1, name='squeeze_layer')

    blstm1 = Bidirectional(LSTM(512, return_sequences=True))(squeezed)
    blstm2 = Bidirectional(LSTM(512, return_sequences=True))(blstm1)

    output = Dense(output_dim + 1, activation="softmax", name="output")(blstm2)

    model = keras.models.Model(inputs=inputs, outputs=output)
    return model