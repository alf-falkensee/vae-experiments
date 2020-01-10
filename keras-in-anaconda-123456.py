# This is a script largely derived from
# https://blog.keras.io/building-autoencoders-in-keras.html
#
# In addition this script defines a signal that would make sense to
# decode via a VAE. The signal is a sequence of integers e.g. 123456
# that is found in random integer noise; the aim of the network is
# then to encode the signal in its latent space
# test:
# >>> exec(open("./keras-in-anaconda-123456.py").read())

from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
import numpy as np


# this is the size of our encoded representations/network structure
encoding_dim = 10  # 10 -> compression of factor 2, assuming the input
                   # (seq_length) is 20 values
seq_length = 20
sample_size = 60000
test_sample_size = 10000 

signal_phase = np.random.randint(low=0, high=15)
print("signal phase to learn is: " + str(signal_phase))
x_seq_train = np.empty(shape=(sample_size, 20), dtype=int)
#x_seq_train = np.zeros(shape=(sample_size, 20), dtype=int)
for index in range(x_seq_train.shape[0]):
    x_seq_train[index] = np.random.randint(low=0, high=10, size=20)
    for sig_index in range(6):
        x_seq_train[index][sig_index + signal_phase] = sig_index + 1

x_seq_test = np.empty(shape=(test_sample_size, 20), dtype=int)
#x_seq_test = np.zeros(shape=(test_sample_size, 20), dtype=int)
for index in range(x_seq_test.shape[0]):
    x_seq_test[index] = np.random.randint(low=0, high=10, size=20)
    for sig_index in range(6):
        x_seq_test[index][sig_index + signal_phase] = sig_index + 1

x_seq_train = x_seq_train.astype('float32') / 10.
x_seq_test = x_seq_test.astype('float32') / 10.



# this is our input placeholder
input_seq = Input(shape=(seq_length,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_seq)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(seq_length, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_seq, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_seq, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='adadelta', loss='mse')

#(x_train, _), (x_test, _) = mnist.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    
print(x_seq_train.shape)
print(x_seq_test.shape)

autoencoder.fit(x_seq_train, x_seq_train,
                epochs=50,
                batch_size=256,
                shuffle=False,
                validation_data=(x_seq_test, x_seq_test))

# autoencoder.save_weights('simple_autoencoder_123456.h5')


# test snippets
encoded_seqs = encoder.predict(x_seq_test)
decoded_seqs = decoder.predict(encoded_seqs)


