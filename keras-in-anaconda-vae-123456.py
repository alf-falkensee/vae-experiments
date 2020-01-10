# This is a script largely derived from
# https://keras.io/examples/variational_autoencoder/ and
# https://blog.keras.io/building-autoencoders-in-keras.html
# 
# In addition this script defines a signal that is decoded via a
# VAE. The signal is a sequence of integers e.g. 123456 that is found
# in random integer noise; the aim of the network is then to encode
# the signal in a latent subspace of the input space
#
# test:
# >>> exec(open("./keras-in-anaconda-vae-123456.py").read())


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


from keras.datasets import mnist



# this is the size of our encoded representations/network structure
encoding_dim = 2  # 2 -> compression of factor 10, assuming the input
                   # (seq_length) is 20 values
seq_length = 20
#sample_size = 60000
#test_sample_size = 10000 
sample_size = 10000
test_sample_size = 3000 
epochs = 50
batch_size = 128


# define the dataset
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


# now some helper functions
def sampling(args):
  z_mean, z_log_var = args
  batch = K.shape(z_mean)[0]
  dim = K.int_shape(z_mean)[1]
  # by default, random_normal has mean = 0 and std = 1.0
  epsilon = K.random_normal(shape=(batch, dim))
  return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
               data,
               batch_size=128,
               model_name="vae_mnist"):
  """Plots labels and MNIST digits as a function of the 2D latent vector

  # Arguments
      models (tuple): encoder and decoder models
      data (tuple): test data and label
      batch_size (int): prediction batch size
      model_name (string): which model is using this function
  """

  encoder, decoder = models
  x_test, y_test = data
  os.makedirs(model_name, exist_ok=True)

  filename = os.path.join(model_name, "vae_mean.png")
  # display a 2D plot of the digit classes in the latent space
  z_mean, _, _ = encoder.predict(x_test,
                                 batch_size=batch_size)
  plt.figure(figsize=(12, 10))
  plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
  plt.colorbar()
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.savefig(filename)
  plt.show()

  filename = os.path.join(model_name, "digits_over_latent.png")
  # display a 30x30 2D manifold of digits
  n = 30
  digit_size = 28
  figure = np.zeros((digit_size * n, digit_size * n))
  # linearly spaced coordinates corresponding to the 2D plot
  # of digit classes in the latent space
  grid_x = np.linspace(-4, 4, n)
  grid_y = np.linspace(-4, 4, n)[::-1]

  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(digit_size, digit_size)
          figure[i * digit_size: (i + 1) * digit_size,
                 j * digit_size: (j + 1) * digit_size] = digit

  plt.figure(figsize=(10, 10))
  start_range = digit_size // 2
  end_range = (n - 1) * digit_size + start_range + 1
  pixel_range = np.arange(start_range, end_range, digit_size)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)
  plt.xticks(pixel_range, sample_range_x)
  plt.yticks(pixel_range, sample_range_y)
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.imshow(figure, cmap='Greys_r')
  plt.savefig(filename)
  plt.show()


# this is our input placeholder
input_seq = Input(shape=(seq_length,), name='encoder_input')
# "encoded" is the encoded representation of the input; use
# reparameterization trick to push the sampling out as input
encoded_mean = Dense(encoding_dim, name='encoded_mean')(input_seq)
encoded_log_var = Dense(encoding_dim, name='encoded_log_var')(input_seq)
encoded = Lambda(sampling, output_shape=(encoding_dim,), name='encoded')([encoded_mean, encoded_log_var])

# this model maps an input to its encoded representation
encoder = Model(input_seq, [encoded_mean, encoded_log_var, encoded], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_123456_encoder.png', show_shapes=True)

# "decoded" is the lossy reconstruction of the input; build a decoder model
latent_inputs = Input(shape=(encoding_dim,), name='encoded_sampling')
decoded = Dense(seq_length, activation='sigmoid')(latent_inputs)

# create the decoder model
decoder = Model(latent_inputs, decoded, name='decoder')
decoder.summary()
plot_model(encoder, to_file='vae_123456_decoder.png', show_shapes=True)

# create the VAE model; maps an input to its reconstruction
decoded = decoder(encoder(input_seq)[2])
vae = Model(input_seq, decoded, name='vae-123456')

models = (encoder, decoder)
#data = (x_seq_test, y_test)

reconstruction_loss = binary_crossentropy(input_seq, decoded)
reconstruction_loss *= encoding_dim
kl_loss = 1 + encoded_log_var - K.square(encoded_mean) - K.exp(encoded_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae,
           to_file='vae_123456.png',
           show_shapes=True)

print(x_seq_train.shape)
print(x_seq_test.shape)

vae.fit(x_seq_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_seq_test, None))



# test snippets
encoded_seqs_mean, _, _ = encoder.predict(x_seq_test)
decoded_seqs = decoder.predict(encoded_seqs_mean)
# >>> x_seq_test[0]
# array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0. , 0. , 0.1, 0.5, 0.3,
#       0.9, 0.9, 0.8, 0.7, 0. , 0.4, 0.2], dtype=float32)
# decoded_seqs[0]
# >>> decoded_seqs[0]
# array([0.11172634, 0.20015274, 0.30000058, 0.40000004, 0.5       ,
#        0.6       , 0.44813058, 0.44715348, 0.4528172 , 0.452056  ,
#        0.45378804, 0.4470731 , 0.45242873, 0.453215  , 0.44665492,
#        0.45157   , 0.4482117 , 0.45150116, 0.44963524, 0.44469997],
#       dtype=float32)




