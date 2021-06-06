package net

import (
	"testing"
)

// TestLayer creates a layer with 5 neurons and tests if it was successful
func TestLayer(t *testing.T) {
	l := NewLayer(5)
	if len(l.Neurons) != 5 {
		t.Errorf("len of Neurons not 5")
	}
}

// TestConnectToLayer creates two layers and tries to connect both of them
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
