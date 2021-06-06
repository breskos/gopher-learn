# Encoders (experimental)

## Overview

**Attention:** Encoders are currently not able to serialize

Encoders means that the incoming data is automatically put into a data vector (processable for the neural net.)
Encoding of your data into processable feature vectors is very important, because strings are initially not suitable as feature vector.
This module helps to find a representation of your data set without your intervention.
Although the encoders can be controlled using by Config parameters in EncoderConfig.

## EncoderConfig

The encoder performs decisions during its runtime.
For that reason a DefaultConfig is applied.
It is possible to get the DefaultConfig, overwrite specific parameters and apply it again to the encoder.

```go
e := encoders.NewEncoder("test encoder")
cfg := encoders.DefaultConfig()
cfg.DictionaryMaxEntries = 300
e.Config = cfg
```
You can find all possible options for editting the encoder config in encoders/config.go.

## Example
Below you can find an example for the encoder.

```go
// generating the encoder, the encoder can hold different input types and dimensions
e := encoders.NewEncoder("test encoder")
cfg := encoders.DefaultConfig()
cfg.DictionaryMaxEntries = 300
e.Config = cfg
inputName := "language-classification"
set := encoders.NewInput(inputName, encoders.String)
for _, v := range data {
    // add your strings here
    set.AddString(someStringSample)
}
// scan takes the set and decides (if it is: encoders.Automatic) which encoding to apply
e.Scan(inputName, set, encoders.Automatic)
// transform brings the input into the choosen encoding
e.Transform(inputName, set)
// explain can be used to see what the encoder has done
e.Explain()
// using encode() and an Unified (can be string or float slice) you get the corresponding vector
vector := e.Encode(inputName, encoders.Unified{String: "Hello whats up with you?", Type: encoders.String})
```


## Workflow
This is the workflow. An Encoder can contain different models for encoding. In the workflow below, (namespace) means that you perform an action on a namespace within the encoder. For example, if you have a mixed input vector with strings and floats, you an put all floats together in one namespace as well as the string.

1. Collect data - The encoder needs the samples from the test or a similar set of data points to optimally fit and decide.
1. Create Encoder - create an encoder with the config (the Encoder itself can encode different inputs)
2. Scanner - (namespace) decides which Encoder to select if you choose encoders.Automatic, if not, the given Encoder will be applied
3. Transform - (namespace) After scanning the set and deciding the data is tranformed into the new vector space
4. After transformation is done, you are ready to go with your new vector representation
5. Using - Encode() (namespace) method of the encoder the encode your input.

## Encoders
If you have no specific idea which encoder to use you can also run using encoders.Automatic.
Using this the encoder will figure out by itself which encoding is applicable.

The encoders work for different data types:

1. N-Grams (strings), encoders.StringNGrams
2. Splitted Dictionary (string), encoders.StringSplitDictionary
3. Dictionary (strings), encoders.StringDictionary
4. FloatExact (numbers), encoders.FloatExact
5. FloatReducer (numbers), encoders.FloatReducer
6. Topic Modelling coming soon (strings) - not implemented yet


## Representation (experimental)

Out of the encoder activity the network generates a representation of the input space.
This representation can be persisted and loaded to continue working on the network.
This representation looks like this:

1. Number of feature vectors
2. Mapping of value to neuron values
