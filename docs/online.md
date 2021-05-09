# Online Learning

## Overview

Online learning is a mechanism in gopher-learn that help to train neural networks on the fly.
With the Online Learning module one can add more data points step by step and the neural net is able to adjust.

## Iterative learning

The `Inject function` forwards new data points to the network.
So the `Iterate function` basically uses a sampled set of know data points and iterates the training process of the neural net.
After that the evaluation kicks in and looks whether the new point was sucessfully learned by the network.

## Usage

```golang
// we need to define the numer of input neurons we have
inputNeurons = 30
// we also need to define the number of hidden neurons we need (we are using one layer here)
hiddenNeurons = 100
// data from file that we want to stream in
// we create an empty set with correct number of inputs and classes
set := learn.NewSet(neural.Classification)
// we know that our data points will have 2 classes R and M
set.AddClass("R")
set.AddClass("M")

// our onlineSet is still empty but we need to defined it
o := online.NewOnline(neural.Classification, inputNeurons, []int{hiddenNeurons}, set)
// we set verbose = true because we want to see the progress
o.SetVerbose(true)
// in case we already started with a not empty data set we run
fMeasure := o.init()
// this will run the previously given data to init the network
// here the network tries to reach a specific overall fMeasure
fmt.Printf("init fMeasure: &f\n", fMeasure)
```

After this set up you can start injecting data points to the network.
Dont forget to call iterate at some points to force the learning process.
The `Inject()` function just runs **hot shot** which means that it tries to force the injection without retraining the network.
This is kind of a tradeoff between learning speed and repeating everything in the data set.

```golang
// here we get a new vector for the network
vector := []float64{1.0, 3.0, 10.5, 5.0, 4.0, 3.3, 5.2}
// now we apply this vector and generate a valid output vector for this class label
// using: GenerateOutputVector(classLabel) function of Set
sample := learn.NewClassificationSample(vector, set.GenerateOutputVector("R"), "R")
// create a sample with input vector and class label
// if a sample of this exists do not override it (force = false)
o.Inject(sample, false)
// after inserting a few points you have to iterate
// Iterate returns the current fMeasure which you can use to observe the quality of the network
fMeasure = o.Iterate()
```


## Config in Online
Online comes with a default config.
For classification tasks the default configuration should fit.
On the other hand for regression you should set Config that fits your data set.
Handling of the config is shown in the examples.