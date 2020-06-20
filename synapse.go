package neural

// Synapse holds the synapse structure
type Synapse struct {
	Weight float64
	In     float64 `json:"-"`
	Out    float64 `json:"-"`
}

// NewSynapse creates a new synapse
func NewSynapse(weight float64) *Synapse {
	return &Synapse{Weight: weight}
}

// NewSynapseFromTo creates a new synapse from neuron to neuron
func NewSynapseFromTo(from, to *Neuron, weight float64) *Synapse {
	syn := NewSynapse(weight)
	from.OutSynapses = append(from.OutSynapses, syn)
	to.InSynapses = append(to.InSynapses, syn)
	return syn
}

// Signal activates the Synapse with an input value
func (s *Synapse) Signal(value float64) {
	s.In = value
	s.Out = s.In * s.Weight
}
