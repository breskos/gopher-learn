package engine

const (
	dTries               = 1
	dEpochs              = 100
	dTrainingSplit       = 0.7
	dLearningRate        = 0.4
	dDecay               = 0.005
	dRegressionThreshold = 0.05
)

// Config has all the learning configurations necessary to learn the netowrk in the engine
type Config struct {
	Tries               int
	Epochs              int
	TrainingSplit       float64
	LearningRate        float64
	Decay               float64
	RegressionThreshold float64
}

// DefaultConfig returns the default config for the engine learner
func DefaultConfig() *Config {
	return &Config{
		Tries:               dTries,
		Epochs:              dEpochs,
		TrainingSplit:       dTrainingSplit,
		LearningRate:        dLearningRate,
		Decay:               dDecay,
		RegressionThreshold: dRegressionThreshold,
	}
}
