package net

// A layer is a structure that holds the neurons with their corresponding
// synapses. Here also the activation of the neurons is calculated.

// NewLayer creats a new layer with the number of neurons given.
func NewLayer(neurons int) *Layer {
	l := &Layer{}
	l.init(neurons)
	return l
}

// Layer holds the neurons with their synapse connections
type Layer struct {
	Neurons []*Neuron
}

// ConnectTo is used to connect the neurons of one layer with the neurons of
// another layer.
func (l *Layer) ConnectTo(layer *Layer) {
	for _, n := range l.Neurons {
		for _, toN := range layer.Neurons {
			n.SynapseTo(toN, 0)
		}
	}
}

// Initializes the new Layer with the given number of neurons
func (l *Layer) init(neurons int) {
	for ; neurons > 0; neurons-- {
		l.addNeuron()
	}
}

// Adds a new neuron to the layer
func (l *Layer) addNeuron() {
	n := NewNeuron()
	l.Neurons = append(l.Neurons, n)
}

// Calculate the a activation of the current layer
func (l *Layer) Calculate() {
	for _, n := range l.Neurons {
		n.Calculate()
	}
}
