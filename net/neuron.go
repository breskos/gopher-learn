package net

// Neuron holds the neuron structure with incoming synapses,
// outgoing synapses, the activation function of the neuron as well as
// the out value.
type Neuron struct {
	OutSynapses        []*Synapse
	InSynapses         []*Synapse         `json:"-"`
	ActivationFunction ActivationFunction `json:"-"`
	Out                float64            `json:"-"`
}

// NewNeuron creates a new empty neuron
func NewNeuron() *Neuron {
	return &Neuron{}
}

// SynapseTo creates a new synapse to a neuron
func (n *Neuron) SynapseTo(nTo *Neuron, weight float64) {
	NewSynapseFromTo(n, nTo, weight)
}

// SetActivationFunction sets the activation function for the neuron
func (n *Neuron) SetActivationFunction(aFunc ActivationFunction) {
	n.ActivationFunction = aFunc
}

// Calculate calculates the actual neuron activity
func (n *Neuron) Calculate() {
	var sum float64
	for _, s := range n.InSynapses {
		sum += s.Out
	}
	n.Out = n.ActivationFunction(sum)
	for _, s := range n.OutSynapses {
		s.Signal(n.Out)
	}
}
