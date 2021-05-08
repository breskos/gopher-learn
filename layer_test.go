package neural

import (
	"testing"
)

func TestLayer(t *testing.T) {
	l := NewLayer(5)
	if len(l.Neurons) != 5 {
		t.Errorf("len of Neurons not 5")
	}
}

func TestConnectToLayer(t *testing.T) {
	count := 5
	l := NewLayer(count)
	l2 := NewLayer(count)

	l.ConnectTo(l2)

	for _, n := range l.Neurons {
		if len(n.OutSynapses) != count {
			t.Errorf("out synapses are not equal %d", count)
		}
	}
}
