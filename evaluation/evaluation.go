package evaluation

import (
	"math"

	neural "github.com/breskos/gopher-learn"
)

// Evaluation contains all the structures necessary for the evaluation
type Evaluation struct {
	Confusion       map[string]map[string]int
	Correct         int
	Wrong           int
	OverallDistance float64
	Usage           neural.NetworkType
	Threshold       float64
}

// NewEvaluation creates a new evaluation object
func NewEvaluation(usage neural.NetworkType, classes []string) *Evaluation {
	evaluation := &Evaluation{
		Usage:     usage,
		Confusion: make(map[string]map[string]int),
	}
	for i := range classes {
		evaluation.Confusion[classes[i]] = make(map[string]int)
		for j := range classes {
			evaluation.Confusion[classes[i]][classes[j]] = 0
		}
	}
	return evaluation
}

// SetRegressionThreshold sets the threshold if you are trying to do Pos / Neg with a regressor
func (e *Evaluation) SetRegressionThreshold(threshold float64) {
	e.Threshold = threshold
}

// Add adds a new data point to the evaluation
func (e *Evaluation) Add(labeledClass, predictedClass string) {
	if _, ok := e.Confusion[labeledClass][predictedClass]; ok {
		e.Confusion[labeledClass][predictedClass]++
	} else {
		e.Confusion[labeledClass][predictedClass] = 1
	}
	if labeledClass == predictedClass {
		e.Correct++
	} else {
		e.Wrong++
	}
}

// AddRegression add a predicted regresssion value to tht set
func (e *Evaluation) AddRegression(label, predicted float64) {
	if math.Abs(label-predicted) >= e.Threshold {
		e.Wrong++
	} else {
		e.Correct++
	}
}
