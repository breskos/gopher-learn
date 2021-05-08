package engine

import (
	"fmt"
	"math/rand"

	neural "github.com/breskos/gopher-learn"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/persist"
)

const (
	runToken   = ","
	epochToken = "."
	tryToken   = "*"
)

// Engine contains every necessary for starting the engine
type Engine struct {
	NetworkInput        int
	NetworkLayer        []int
	NetworkOutput       int
	Data                *learn.Set
	WinnerNetwork       *neural.Network
	WinnerEvaluation    evaluation.Evaluation
	Verbose             bool
	Usage               neural.NetworkType
	RegressionThreshold float64
}

// NewEngine creates a new Engine object
func NewEngine(usage neural.NetworkType, hiddenLayer []int, data *learn.Set) *Engine {
	var outputLength int
	if neural.Regression == usage {
		outputLength = 1
	} else {
		outputLength = len(data.Samples[0].Output)
	}
	return &Engine{
		NetworkInput:        len(data.Samples[0].Vector),
		NetworkOutput:       outputLength,
		NetworkLayer:        hiddenLayer,
		Data:                data,
		WinnerNetwork:       neural.BuildNetwork(usage, len(data.Samples[0].Vector), hiddenLayer, data.ClassToLabel),
		WinnerEvaluation:    *evaluation.NewEvaluation(usage, data.GetClasses()),
		Verbose:             false,
		Usage:               usage,
		RegressionThreshold: 0.0,
	}
}

// SetVerbose set verbose mode default = false
func (e *Engine) SetVerbose(verbose bool) {
	e.Verbose = verbose
}

// SetRegressionThreshold sets the evaluation threshold for the regression
func (e *Engine) SetRegressionThreshold(threshold float64) {
	e.RegressionThreshold = threshold
}

// GetWinner returns the winner network from training
func (e *Engine) GetWinner() (*neural.Network, *evaluation.Evaluation) {
	return e.WinnerNetwork, &e.WinnerEvaluation
}

// Start takes the paramter to start the engine and run it
func (e *Engine) Start(criterion neural.Criterion, tries, epochs int, trainingSplit, startLearning, decay float64) {
	network := neural.BuildNetwork(e.Usage, e.NetworkInput, e.NetworkLayer, e.Data.ClassToLabel)
	training, validation := split(e.Usage, e.Data, trainingSplit)
	for try := 0; try < tries; try++ {
		learning := startLearning
		if e.Verbose {
			fmt.Printf("\n> start try %v. training / test: %v / %v (%v)\n", (try + 1), len(training.Samples), len(validation.Samples), trainingSplit)
		}
		for ; learning > 0.0; learning -= decay {
			train(network, training, learning, epochs, e.Verbose)
			evaluation := evaluate(e.Usage, network, validation, training, e.RegressionThreshold)
			if compare(e.Usage, criterion, &e.WinnerEvaluation, evaluation) {
				e.WinnerNetwork = copy(network)
				e.WinnerEvaluation = *evaluation
				if e.Verbose {
					print(&e.WinnerEvaluation)
				}
			}
		}
		if e.Verbose {
			fmt.Print(tryToken + "\n")
		}
	}
}

func print(e *evaluation.Evaluation) {
	fmt.Printf("\n [Best] acc: %.2f  / bacc: %.2f / f1: %.2f / correct: %.2f / distance: %.2f\n", e.GetOverallAccuracy(), e.GetOverallBalancedAccuracy(), e.GetOverallFMeasure(), e.GetCorrectRatio(), e.GetDistance())
}

func split(usage neural.NetworkType, set *learn.Set, ratio float64) (*learn.Set, *learn.Set) {
	multiplier := 100
	normalizedRatio := int(ratio * float64(multiplier))
	var training, evaluation learn.Set
	training.ClassToLabel = set.ClassToLabel
	evaluation.ClassToLabel = set.ClassToLabel
	for i := range set.Samples {
		if rand.Intn(multiplier) <= normalizedRatio {
			training.Samples = append(training.Samples, set.Samples[i])
		} else {
			evaluation.Samples = append(evaluation.Samples, set.Samples[i])
		}
	}
	return &training, &evaluation
}

func train(network *neural.Network, data *learn.Set, learning float64, epochs int, verbose bool) {
	for e := 0; e < epochs; e++ {
		for sample := range data.Samples {
			learn.Learn(network, data.Samples[sample].Vector, data.Samples[sample].Output, learning)
		}
		if verbose {
			fmt.Print(epochToken)
		}
	}
	if verbose {
		fmt.Print(runToken)
	}

}

func evaluate(usage neural.NetworkType, network *neural.Network, test *learn.Set, train *learn.Set, regressionThreshold float64) *evaluation.Evaluation {
	evaluation := evaluation.NewEvaluation(usage, train.GetClasses())
	evaluation.SetRegressionThreshold(regressionThreshold)
	for sample := range test.Samples {
		evaluation.AddDistance(network, test.Samples[sample].Vector, test.Samples[sample].Output)
		if neural.Classification == usage {
			winner := network.CalculateWinnerLabel(test.Samples[sample].Vector)
			evaluation.Add(test.Samples[sample].Label, winner)
		} else {
			prediction := network.Calculate(test.Samples[sample].Vector)
			evaluation.AddRegression(test.Samples[sample].Value, prediction[0])
		}
	}
	return evaluation
}

func compare(usage neural.NetworkType, criterion neural.Criterion, current *evaluation.Evaluation, try *evaluation.Evaluation) bool {
	if current.Correct+current.Wrong == 0 {
		return true
	}
	switch criterion {
	case neural.Accuracy:
		if current.GetOverallAccuracy() < try.GetOverallAccuracy() {
			return true
		}
	case neural.BalancedAccuracy:
		if current.GetOverallBalancedAccuracy() < try.GetOverallBalancedAccuracy() {
			return true
		}
	case neural.FMeasure:
		if current.GetOverallFMeasure() < try.GetOverallFMeasure() {
			return true
		}
	case neural.Simple:
		if current.GetCorrectRatio() < try.GetCorrectRatio() {
			return true
		}
	case neural.Distance:
		if current.GetDistance() > try.GetDistance() {
			return true
		}
	}
	return false
}

func copy(from *neural.Network) *neural.Network {
	return persist.FromDump(persist.ToDump(from))
}
