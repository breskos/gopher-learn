# Sonar

Train a classifier that discriminates sonar signals bounced off a metal cylinder (`M`) from those bounced off a roughly cylindrical rock (`R`).

Data: [Connectionist Bench (Sonar, Mines vs. Rocks)](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29), also available in [gopher-learn-data](https://github.com/breskos/gopher-learn-data/tree/master/sonar).

The example shows how to:

- load a CSV data set,
- train a network with 100 hidden neurons using the engine and `engine.Distance` as criterion,
- print per-class metrics and the confusion matrix,
- save and load the network.

```
go run . -data path/to/sonar/data.csv
```

Expect a validation accuracy of roughly 0.8 to 0.9, depending on the random split.
