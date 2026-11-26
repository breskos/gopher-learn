# Examples for gopher-learn

![The gopher-learn logo](https://breskos.com/wp-content/uploads/2026/11/gopher.png)

## Important
All the data needed for the examples can be found at [gopher-learn-data](https://github.com/breskos/gopher-learn-data).

## Sonar

Basic example that uses the engine of gopher-learn to train a small neural network with basic parameters.
More information in sonar ReadMe file.
Data from: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29

## Wine quality

In this example we train a regressor (we don't have classes but real value to determine).
The quality of the wine is very important because wine is tasty (or should be).
So here we have some criteria we can use to determine good from bad wine.
Data from: http://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Dictionary (Encoder)

This example shows the encoder to a dictionary approach.
The data was collected by Alexander Bresk.

## NGram (Encoder)

This example shows how to encode strings via ngram encoder.
The data was collected by Alexander Bresk.

## Online Generator (Online)

This example shows the online learning part of gopher-learn.
It shows you how to apply this functionality.

## Online Sonar (Online)

Using the data from the Sonar example, this example shows you how to use it in online mode.

## Online service (production online learning)

Trains a model on the history of an event log, then runs it as a live service with delayed, wrong and corrupt feedback, corrections and a concept drift, and compares it with the frozen model. Also serves the model over HTTP. Needs no data files. See [online-service](online-service) and [docs/service.md](../docs/service.md).

## Embedded (float32 model)

Trains a motor fault detector on a computer, exports it as a 1 KB float32 model and runs it on an embedded device with a 1.5 to 1.9 MB program. See [embedded](embedded) and [docs/embedded.md](../docs/embedded.md).
