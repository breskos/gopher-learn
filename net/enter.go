package net

// Enter represents the input vector that comes into the network
type Enter struct {
	OutSynapses []*Synapse
	Input       float64 `json:"-"`
}

// NewEnter creates a new Enter
func NewEnter() *Enter {
	return &Enter{}
}

// SynapseTo creates a new Synapse to a Neuron from the next layer
func (e *Enter) SynapseTo(nTo *Neuron, weight float64) {
	syn := NewSynapse(weight)

	e.OutSynapses = append(e.OutSynapses, syn)
	nTo.InSynapses = append(nTo.InSynapses, syn)
}

// SetInput sets the input the specific enter
func (e *Enter) SetInput(val float64) {
	e.Input = val
}

// ConnectTo connects the Enter to the next layer (to the neurons)
func (e *Enter) ConnectTo(layer *Layer) {
	for _, n := range layer.Neurons {
		e.SynapseTo(n, 0)
	}
}

// Signal passes the signal into the network
func (e *Enter) Signal() {
	for _, s := range e.OutSynapses {
		s.Signal(e.Input)
	}
}
