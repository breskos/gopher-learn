//go:build race

package nn

// raceEnabled is true with the race detector, which makes sync.Pool drop
// items on purpose, so pooled code allocates.
const raceEnabled = true
