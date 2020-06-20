# Sonar Data Set for Online Learning

Abstract: The task is to train a network to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock.

Connectionist Bench (Sonar, Mines vs. Rocks) Data Set

Found here: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29

## Example

The data set was also used for the simple sonar example where a basic Multi Layer Perceptron was built.
This example explains the use of the Gopher-Learn online mode.
In contrast to the simple example, where the data was initially given to the network, the data was given piece by piece to the network.
This can be seen as the stream approach.
Data is constantly flowing in and the network has to adapt.

Execute the example using the following command:

```
> go run main.go
```
