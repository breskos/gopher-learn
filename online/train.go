package online

import (
	"math/rand"

	neural "github.com/breskos/gopher-learn"
	"github.com/breskos/gopher-learn/evaluation"
	learn "github.com/breskos/gopher-learn/learn"
)

// Trains the network with the given Sample set and learning rate
func train(network *neural.Network, data *learn.Set, learning float64, epochs int) {
	for e := 0; e < epochs; e++ {
		for sample := range data.Samples {
			learn.Learn(network, data.Samples[sample].Vector, data.Samples[sample].Output, learning)
		}
	}
}

// Splits the set into training and test set
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

// Evaluates the network and finds the winner network based on network criterion
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
