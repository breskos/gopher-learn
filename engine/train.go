package engine

import (
	"fmt"
	"math/rand"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	neural "github.com/breskos/gopher-learn/net"
	"github.com/breskos/gopher-learn/persist"
)

// Splits a given data set by a given ratio into training and evaluation
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

// Trains the neural network with the vgiven samples for a given epoch of time and and learning rate
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

// Evaluates the neural network using the current trained versions of it by a given criterion
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

// Compares two networks by a their evaluation and a given criterion
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

// Copies a neural network from another
func copy(from *neural.Network) *neural.Network {
	return persist.FromDump(persist.ToDump(from))
}
