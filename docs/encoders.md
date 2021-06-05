# Encoders (experimental)

## Overview

**Attention:** Encoders are currently not able to serialize

Encoders means that the incoming data is automatically put into a data vector (processable for the neural net.)
Encoding of your data into processable feature vectors is very important, because strings are initially not suitable as feature vector.
This module helps to find a representation of your data set without your intervention.
Although the encoders can be controlled using by Config parameters in EncoderConfig.

## EncoderConfig

// TODO(abresk) here we describe all the encoder config values


## Workflow
This is the workflow. An Encoder can contain different models for encoding. In the workflow below, (namespace) means that you perform an action on a namespace within the encoder. For example, if you have a mixed input vector with strings and floats, you an put all floats together in one namespace as well as the string.

1. Collect data - The encoder needs the samples from the test or a similar set of data points to optimally fit and decide.
1. Create Encoder - create an encoder with the config (the Encoder itself can encode different inputs)
2. Scanner - (namespace) decides which Encoder to select if you choose encoders.Automatic, if not, the given Encoder will be applied
3. Transform - (namespace) After scanning the set and deciding the data is tranformed into the new vector space
4. After transformation is done, you are ready to go with your new vector representation
5. Using - Encode() (namespace) method of the encoder the encode your input.

## Encoders

The encoders work for different data types:
1. N-Grams (strings)
2. Splitted Dictionary (string)
3. Dictionary (strings)
4. FloatExact (numbers)
5. FloatReducer (numbers)

## Representation (experimental)

Out of the encoder activity the network generates a representation of the input space.
This representation can be persisted and loaded to continue working on the network.
This representation looks like this:

1. Number of feature vectors
2. Mapping of value to neuron values
