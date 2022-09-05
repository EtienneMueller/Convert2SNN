import tensorflow as tf
import numpy as np
from convert2snn import convert2pop


# conda install numpy=1.19, otherwise: NotImplementedError: Cannot convert a symbolic Tensor (rnn/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
tf.random.set_seed(1234)

top_words = 2500
max_review_length = 500
#embedding_vecor_length = 32
units = 25
batch_size = 64
train_length = 1000  # reduce this if memory is not sufficient

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)

x_train, x_test = x_train[0:train_length], x_test[0:train_length]
y_train, y_test = y_train[0:train_length], y_test[0:train_length]
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length)

x_train = tf.one_hot(x_train, top_words, axis=-1)
x_test = tf.one_hot(x_test, top_words, axis=-1)

try:
    model = tf.keras.models.load_model('data/population')
except:
    print("\nNo trained model found. Training...")
    inputs = tf.keras.Input(shape=(max_review_length, top_words))
    # x = tf.keras.layers.RNN(tf.keras.layers.GRUCell(units, activation='tanh', recurrent_activation='sigmoid'), return_sequences=False, return_state=False)(inputs)  
    x = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='sigmoid'), return_sequences=False, return_state=False)(inputs)  # , activation='tanh', recurrent_activation='sigmoid'
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)  # validation_data=(x_test, y_test),
    print("Model trained")
    model.save('data/population')

_, acc = model.evaluate(x_train, y_train, batch_size=64)  # 93.43% (93.50%)
_, acc = model.evaluate(x_test, y_test, batch_size=64)  # 88.25% (88.20%)

# Conversion
seeds = 3
pops = [100, 1000]
for n in range(len(pops)):
    print("Current population size: ", pops[n])
    for seed in range(0, seeds):
        tf.random.set_seed(seed)
        snn = convert2pop(max_review_length, top_words, units, pops, n)
        snn.set_weights(model.get_weights())
        _, acc = snn.evaluate(x_test, y_test, batch_size=64)
