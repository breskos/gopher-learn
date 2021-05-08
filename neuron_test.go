package neural

import (
	"testing"
)

func TestAttachNeurons(t *testing.T) {
	n := NewNeuron()
	n2 := NewNeuron()
	w := 0.5
	n.SynapseTo(n2, w)
	if n.OutSynapses[0].Weight != w {
		t.Errorf("out synapse has wrong weights")
	}
}

func TestInputsSynapses(t *testing.T) {
	n := NewNeuron()
	NewSynapseFromTo(NewNeuron(), n, 0.1)
	NewSynapseFromTo(NewNeuron(), n, 0.1)
	NewSynapseFromTo(NewNeuron(), n, 0.1)
	if len(n.InSynapses) != 3 {
		t.Errorf("in synapse is not 3")
	}
}
