import tensorflow as tf
import numpy as np

from rate import SpikingReLU, SpikingSigmoid, SpikingTanh, Accumulate
from population import PopLSTM
import temporal

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import _caching_device


# Conversion function
def convert2rate(model, weights, x_test, y_test, err="CategoricalCrossentropy"):
    print("Converted model:\n" + "-"*32)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            print("Input Layer")
            inputs = tf.keras.Input(shape=(1, model.layers[0].input_shape[0][1]), batch_size=y_test.shape[0])
            x = inputs
        elif isinstance(layer, tf.keras.layers.Dense):
            x = tf.keras.layers.Dense(layer.output_shape[1])(x)
            # x = tf.keras.layers.RNN(DenseRNN(layer.output_shape[1]),
            #                         return_sequences=True,
            #                         return_state=False,
            #                         stateful=True)(x)
            if layer.activation.__name__ == 'linear':
                print("Dense Layer w/o activation")
                pass
            elif layer.activation.__name__ == 'relu':
                print("Dense Layer with SpikingReLU")
                x = tf.keras.layers.RNN(SpikingReLU(layer.output_shape[1]),
                                        return_sequences=True,
                                        return_state=False,
                                        stateful=True)(x)
            elif layer.activation.__name__ == 'sigmoid':
                print("Dense Layer with SpikingSigmoid")
                x = tf.keras.layers.RNN(SpikingSigmoid(layer.output_shape[1]),
                                        return_sequences=True,
                                        return_state=False,
                                        stateful=True)(x)
            elif layer.activation.__name__ == 'tanh':
                print("Dense Layer with SpikingTanh")
                x = tf.keras.layers.RNN(SpikingTanh(layer.output_shape[1]),
                                        return_sequences=True,
                                        return_state=False,
                                        stateful=True)(x)
            else:
                print('[Info] Activation type',
                      layer.activation.__name__,
                      'not implemented')
        elif isinstance(layer, tf.keras.layers.ReLU):
            print("SpikingReLU Layer")
            x = tf.keras.layers.RNN(SpikingReLU(layer.output_shape[1]),
                                    return_sequences=True,
                                    return_state=False,
                                    stateful=True)(x)
        elif isinstance(layer, tf.keras.layers.Softmax):
            print("Accumulate + Softmax Layer")
            print(layer.output_shape[1])
            x = tf.keras.layers.RNN(Accumulate(layer.output_shape[1]),
                                    return_sequences=True,
                                    return_state=False,
                                    stateful=True)(x)
            x = tf.keras.layers.Softmax()(x)
        else:
            print("[Info] Layer type ", layer, "not implemented")
    spiking = tf.keras.models.Model(inputs=inputs, outputs=x)
    print("-"*32 + "\n")

    if err == "CategoricalCrossentropy":
        print("CategoricalCrossentropy")
        spiking.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),#(from_logits=True),
            optimizer="adam",
            metrics=["categorical_accuracy"],)
    elif err == "SparseCategoricalCrossentropy":
        print("SparseCategoricalCrossentropy")
        spiking.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["sparse_categorical_accuracy"],)
    elif err == "MeanSquaredError":
        print("MeanSquaredError")
        spiking.compile(
            loss=tf.keras.losses.MeanSquaredError(),  # (from_logits=True),
            optimizer="adam",
            metrics=["mean_squared_error"],)

    spiking.set_weights(weights)
    return spiking


def get_normalized_weights(model, x_test, percentile=100):
    all_activations = np.zeros([1, ])
    for layer in model.layers:
        activation = tf.keras.Model(inputs=model.inputs,
                                    outputs=layer.output)(x_test).numpy()
        all_activations = np.concatenate((all_activations, activation.flatten()))

    max_activation = np.percentile(all_activations, percentile)

    weights = model.get_weights()
    if max_activation == 0:
        print("\n" + "-"*32 + "\nNo normalization\n" + "-"*32)
    else:
        print("\n" + "-"*32 + "\nNormalizing by", max_activation, "\n" + "-"*32)
        for i in range(len(weights)):
            weights[i] /= (max_activation)

    # Testing normalized weights
    """model.set_weights(weights)
    max_activation = 0
    for layer in model.layers:
        print(type(layer))
        if isinstance(layer, tf.keras.layers.ReLU):
            activation = tf.keras.Model(inputs=model.inputs, outputs=layer.output)(x_test).numpy()
            print("Local max", np.amax(activation))
            if np.amax(activation) > max_activation:
                max_activation = np.amax(activation)
            print("Max", max_activation)"""
    return weights


def evaluate_conversion(converted_model, x_test, y_test, testacc, timesteps=100):
    for i in range(1, int(timesteps+1)):
        _, acc = converted_model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=1)
        print(
            "Timesteps", str(i) + "/" + str(timesteps) + " -",
            "acc spiking (orig): %.2f%% (%.2f%%)" % (acc*100, testacc*100),
            "- conv loss: %+.2f%%" % ((-(1 - acc/testacc)*100)))


def convert2pop(max_review_length, top_words, units, pops, n):
    inputs = tf.keras.Input(shape=(max_review_length, top_words))
    x = tf.keras.layers.RNN(
        PopLSTM(units, n_i=pops[n], n_f=pops[n], n_c=None, n_o=pops[n], n_h=None),
        return_sequences=False,
        return_state=False)(inputs)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    snn = tf.keras.Model(inputs=inputs, outputs=x)
    snn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return snn
