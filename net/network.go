package net

import (
	"fmt"
	"math/rand"
)

// Criterion is needed to decide if the Engine found a better working network.
// It functions as decider during the training.
type Criterion int

const (
	// Accuracy decides evaluation by accuracy
	Accuracy Criterion = iota
	// BalancedAccuracy decides evaluation by balanced accuracy
	BalancedAccuracy
	// FMeasure decides evaluation by f-measure
	FMeasure
	// Simple decides on simple wrong/correct ratio
	Simple
	// Distance decides evaluation by distance to ideal output
	Distance
)

// Network contains all the necessary information to use the neural network
type Network struct {
	Enters    []*Enter
	Layers    []*Layer
	Out       []float64 `json:"-"`
	OutLabels map[int]string
}

// NewNetwork creates a new neural network
func NewNetwork(in int, layers []int, labels map[int]string) *Network {
	n := &Network{
		Enters:    make([]*Enter, 0, in),
		Layers:    make([]*Layer, 0, len(layers)),
		OutLabels: labels,
	}
	n.init(in, layers, NewLogisticFunc(1))
	return n
}

// Initializes the Network with the given layers and activation function.
func (n *Network) init(in int, layers []int, aFunc ActivationFunction) {
	n.initLayers(layers)
	n.initEnters(in)
	n.ConnectLayers()
	n.ConnectEnters()
	n.SetActivationFunction(aFunc)
}

// Initializes the Layers with the given count of neurons as well as
// creating all the synapses necessary.
func (n *Network) initLayers(layers []int) {
	for _, count := range layers {
		layer := NewLayer(count)
		n.Layers = append(n.Layers, layer)
	}
}

// Intializes the Enters (size of feature vector) that enters the network.
func (n *Network) initEnters(in int) {
	for ; in > 0; in-- {
		e := NewEnter()
		n.Enters = append(n.Enters, e)
	}
}

// ConnectLayers connects all layers with corresponding neurons
func (n *Network) ConnectLayers() {
	for i := len(n.Layers) - 1; i > 0; i-- {
		n.Layers[i-1].ConnectTo(n.Layers[i])
	}
}

// ConnectEnters connects the input neurons with the first hidden layer
func (n *Network) ConnectEnters() {
	for _, e := range n.Enters {
		e.ConnectTo(n.Layers[0])
	}
}

// SetActivationFunction sets the activation function for the network
func (n *Network) SetActivationFunction(aFunc ActivationFunction) {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			n.SetActivationFunction(aFunc)
		}
	}
}

// Set the current feature vector for the network
func (n *Network) setEnters(v *[]float64) {
	values := *v
	if len(values) != len(n.Enters) {
		panic(fmt.Sprint("Enters count ( ", len(n.Enters), " ) != count of elements in SetEnters function argument ( ", len(values), " ) ."))
	}

	for i, e := range n.Enters {
		e.Input = values[i]
	}

}

// This function sends the current feature vector to the network
func (n *Network) sendEnters() {
	for _, e := range n.Enters {
		e.Signal()
	}
}

// Used during forward calculation through the network
func (n *Network) calculateLayers() {
	for _, l := range n.Layers {
		l.Calculate()
	}
}

// Generates the output from the neurons
func (n *Network) generateOut() {
	outL := n.Layers[len(n.Layers)-1]
	n.Out = make([]float64, len(outL.Neurons))

	for i, neuron := range outL.Neurons {
		n.Out[i] = neuron.Out
	}
}

// Calculate calculates the result of a input vector
func (n *Network) Calculate(enters []float64) []float64 {
	n.setEnters(&enters)
	n.sendEnters()
	n.calculateLayers()
	n.generateOut()

	return n.Out
}

// CalculateLabels output with all labels of output neurons
func (n *Network) CalculateLabels(enters []float64) map[string]float64 {
	results := make(map[string]float64)
	out := n.Calculate(enters)
	for index, label := range n.OutLabels {
		results[label] = out[index]
	}
	return results
}

// CalculateWinnerLabel calculates the output and just returns the label of the winning euron
func (n *Network) CalculateWinnerLabel(enters []float64) string {
	calculatedLabels := n.CalculateLabels(enters)
	winnerValue := 0.0
	winnerLabel := ""
	for label, value := range calculatedLabels {
		if value > winnerValue {
			winnerValue = value
			winnerLabel = label
		}
	}
	return winnerLabel
}

// RandomizeSynapses applies a random value to all synapses
func (n *Network) RandomizeSynapses() {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			for _, s := range n.InSynapses {
				s.Weight = 2 * (rand.Float64() - 0.5)
			}
		}
	}
}

// BuildNetwork builds a neural network from parameters given
func BuildNetwork(usage NetworkType, input int, hidden []int, labels map[int]string) *Network {
	hidden = append(hidden, len(labels))
	network := NewNetwork(input, hidden, labels)
	network.RandomizeSynapses()
	return network
}
