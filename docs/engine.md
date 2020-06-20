# Engine

## Overview

Engine triggers the training of the neural network and returns the winner network.
With engine you can use a training set to train the network.

## Why using an engine?

If you want to successfully train a neural network you need a lot of parameters doing the right things.
Engine was defined to bake your neural network based on your given data.
Instead of fine tuning parameters and split data sets on your own, you can use the engine for that.

See examples **sonar** and **wine-quality** for more insights on the engine.

## Rough pseudo code description

```go
// one epoch is defined as one forward pass and one backward pass of all the training examples
for number of #try (tries)
  for learningRate minus decay if decay is not 0
    for num of #epochs the network sees the training set
```
