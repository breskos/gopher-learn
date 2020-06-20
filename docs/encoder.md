# AutoEncoder (experimental)

## Overview

Autoencoder means that the incoming data is automatically put into a data vector (processable for the neural net.)
Encoding of your data into processable feature vectors is very important.
This module helps to find a representation of your data without intervention.

## Encoder

The encoders work for different data types:

1. N-Grams (strings)
2. Status (strings)
3. Binary (binary)
4. Number (numbers)

## Representation

Out of the encoder activity the network generates a representation of the input space.
This representation can be persisted and loaded to continue working on the network.
This representation looks like this:

1. Number of feature vectors
2. Mapping of value to neuron values
