package neural

import (
	"testing"
)

// TestLogisticFunc tests the activation function
func TestLogisticFunc(t *testing.T) {
	f := NewLogisticFunc(1)

	if f(0) != 0.5 {
		t.Errorf("f(0) not equal 0.5")
	}
	if 1-f(6) <= 0 {
		t.Errorf("1-f(6) not > 0")
	}
	if 1-f(6) >= 0.1 {
		t.Errorf("1-f(6) not < 0.1")
	}
}
