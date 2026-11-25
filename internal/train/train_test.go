package train

import "testing"

func TestNewRand(t *testing.T) {
	a, b := NewRand(42), NewRand(42)
	for range 10 {
		if a.Uint64() != b.Uint64() {
			t.Fatal("same seed must produce the same sequence")
		}
	}
	if NewRand(1).Uint64() == NewRand(2).Uint64() {
		t.Fatal("different seeds should produce different sequences")
	}
	if NewRand(0) == nil {
		t.Fatal("seed 0 must return a random generator")
	}
}
