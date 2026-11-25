# Online sonar

Uses the [sonar data set](../sonar) to demonstrate online learning: instead of training on the full data set at once, samples are injected one by one and the network adapts as the data streams in.

```
go run . -data path/to/sonar/data.csv
```
