package online

const (
	dFirstShots        = 5
	dHotShotBoost      = 0.5
	dTrainingSplit     = 0.7
	dMinimumDataPoints = 10
	dMinEpochs         = 10
	dMaxEpochs         = 30
	dMinLearningSpeed  = 0.2
	dMaxLearningSpeed  = 0.5
	dInitialFMeasure   = 0.70
	dMaxInitLoops      = 5
)

// Config has all the learning configurations necessary to learn the network online
type Config struct {
	FirstShots        int
	HotShotBoost      float64
	TrainingSplit     float64
	MinimumDataPoints int
	MinEpochs         int
	MaxEpochs         int
	MinLearningSpeed  float64
	MaxLearningSpeed  float64
	InitialFMeasure   float64
	MaxInitLoops      int
}

// DefaultConfig returns the default config for the online learner
func DefaultConfig() *Config {
	return &Config{
		FirstShots:        dFirstShots,
		HotShotBoost:      dHotShotBoost,
		TrainingSplit:     dTrainingSplit,
		MinimumDataPoints: dMinimumDataPoints,
		MinEpochs:         dMinEpochs,
		MaxEpochs:         dMaxEpochs,
		MinLearningSpeed:  dMinLearningSpeed,
		MaxLearningSpeed:  dMaxLearningSpeed,
		InitialFMeasure:   dInitialFMeasure,
		MaxInitLoops:      dMaxInitLoops,
	}
}
