package online

import (
	"errors"
	"fmt"
	"math/rand"
	"time"

	neural "github.com/breskos/gopher-learn"
	"github.com/breskos/gopher-learn/evaluation"
	learn "github.com/breskos/gopher-learn/learn"
)

const (
	errDataPointExists   = "data point exists (force not activated)"
	firstShots           = 5
	hotShotLearningSpeed = 0.8
	trainingSplit        = 0.7
	minimumDataPoints    = 10
	minEpochs            = 10
	maxEpochs            = 30
	minLearningSpeed     = 0.2
	maxLearningSpeed     = 0.5
	initialFMeasure      = 0.85
	maxInitLoops         = 5
)

// Online contains every necessary for starting the engine
type Online struct {
	NetworkInput        int
	NetworkLayer        []int
	NetworkOutput       int
	Data                *learn.Set
	Network             *neural.Network
	LastEvaluation      *evaluation.Evaluation
	Verbose             bool
	Usage               neural.NetworkType
	AddedPoints         int
	RegressionThreshold float64
}

// NewOnline creates a new Engine object
func NewOnline(usage neural.NetworkType, inputs int, hiddenLayer []int, data *learn.Set) *Online {
	var outputLength int
	if neural.Regression == usage {
		outputLength = 1
	} else {
		outputLength = len(data.ClassToLabel)
	}
	return &Online{
		NetworkInput:  inputs,
		NetworkOutput: outputLength,
		NetworkLayer:  hiddenLayer,
		Data:          data,
		Network:       neural.BuildNetwork(usage, inputs, hiddenLayer, data.ClassToLabel),
		Verbose:       false,
		Usage:         usage,
		AddedPoints:   0,
	}
}

// Init initializes the online learner with a short learning upfront
func (o *Online) Init() float64 {
	fMeasure := 0.0
	for i := 0; i < maxInitLoops; i++ {
		fMeasure = o.Iterate()
		if fMeasure < initialFMeasure {
			return fMeasure
		}
	}
	return fMeasure

}

// Inject tries to inject a new data point into the neural net
func (o *Online) Inject(sample *learn.Sample, force bool) error {
	exists := o.Data.SampleExists(sample)
	if exists && !force {
		return errors.New(errDataPointExists)
	}
	err := o.Data.AddSample(sample)
	if err != nil {
		return fmt.Errorf("cannot add example: %v", err)
	}
	o.hotShot(sample)
	return nil
}

// Applies a Sample with hotShot speed to the network
func (o *Online) hotShot(sample *learn.Sample) {
	for i := 0; i < firstShots; i++ {
		learn.Learn(o.Network, sample.Vector, sample.Output, hotShotLearningSpeed)
	}
}

// Iterate iterates over the data set and applies continous learning
func (o *Online) Iterate() float64 {
	if len(o.Data.Samples) < minimumDataPoints {
		return 0.0
	}
	rand.Seed(time.Now().UnixNano())
	training, testing := split(o.Usage, o.Data, trainingSplit)
	speed := minLearningSpeed + rand.Float64()*(maxLearningSpeed-minLearningSpeed)
	epochs := rand.Intn(maxEpochs-minEpochs+1) + minEpochs
	train(o.Network, training, speed, epochs)
	evaluation := evaluate(o.Usage, o.Network, testing, training, o.RegressionThreshold)
	if o.Verbose {
		evaluation.PrintConfusionMatrix()
		evaluation.PrintSummaries()
	}
	o.LastEvaluation = evaluation
	return evaluation.GetOverallFMeasure()
}

// SetVerbose sets the verbose version meaning debug and evaluation logs
func (o *Online) SetVerbose(verbose bool) {
	o.Verbose = verbose
}

func (o *Online) sampleExists(sample *learn.Sample) bool {
	if sample.VectorHash == "" {
		sample.UpdateHashes()
	}
	if o.Data.SampleExists(sample) {
		return true
	}
	return false
}

// Prints the current evaluation
func print(e *evaluation.Evaluation) {
	fmt.Printf("\n [Best] acc: %.2f  / bacc: %.2f / f1: %.2f / correct: %.2f / distance: %.2f\n", e.GetOverallAccuracy(), e.GetOverallBalancedAccuracy(), e.GetOverallFMeasure(), e.GetCorrectRatio(), e.GetDistance())
}
