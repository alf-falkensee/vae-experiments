In the current state of this project you'll find a simple autoencoder
example that performs a PCA equivalent found in
'keras-in-anaconda-123456.py' and a more elaborate vae found in
'keras-in-anaconda-vae-123456.py'. Both are almost directly inspired
from https://keras.io/examples/variational_autoencoder/ and
https://blog.keras.io/building-autoencoders-in-keras.html

The data that looks like:

`[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0. , 0. , 0.1, 0.5, 0.3, 0.9,
0.9, 0.8, 0.7, 0. , 0.4, 0.2]`

i.e. that has some signal of the form 123456 hidden in a sequence of
random numbers here of length 20. In that case the sequence 123456 is
mapped to the range [0,1] in order to facilitate the modeling. Thie
idea behind this sequence is to keep the data as simple as possible
whilst retaining enough complexity to try out all possible current xnn
approaches.

For testing preferably use an anaconda console and

`>>> exec(open("./keras-in-anaconda-123456.py").read())`

or

`>>> exec(open("./keras-in-anaconda-vae-123456.py").read())`


The next to be implemented/tried-out is a cnn combined with a
vae. Stay tuned.

