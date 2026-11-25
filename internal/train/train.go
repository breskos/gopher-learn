// Package train contains helpers shared by engine and online.
package train

import "math/rand/v2"

// NewRand returns a generator for the given seed, or a randomly seeded one for seed 0.
func NewRand(seed uint64) *rand.Rand {
	if seed == 0 {
		return rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
}
