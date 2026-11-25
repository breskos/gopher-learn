package evaluation

import (
	"math"
	"slices"
)

// MAE returns the mean absolute error of the regression predictions.
func (e *Evaluation) MAE() float64 {
	if e.Count == 0 {
		return 0
	}
	return e.SumAbsError / float64(e.Count)
}

// RMSE returns the root mean squared error of the regression predictions.
func (e *Evaluation) RMSE() float64 {
	if e.Count == 0 {
		return 0
	}
	return math.Sqrt(e.SumSqError / float64(e.Count))
}

// R2 returns the coefficient of determination of the regression predictions:
// 1 is perfect, 0 is as good as always predicting the mean, negative is worse.
// It returns 0 if the actual values do not vary.
func (e *Evaluation) R2() float64 {
	if e.Count == 0 {
		return 0
	}
	n := float64(e.Count)
	total := e.SumActualSq - e.SumActual*e.SumActual/n
	if total <= 0 {
		return 0
	}
	return 1 - e.SumSqError/total
}

// probabilities normalizes network outputs to a probability distribution.
// Softmax outputs pass unchanged; sigmoid outputs are divided by their sum.
func probabilities(scores []float64) []float64 {
	p := make([]float64, len(scores))
	var sum float64
	for i, s := range scores {
		p[i] = max(s, 0)
		sum += p[i]
	}
	if sum == 0 {
		for i := range p {
			p[i] = 1 / float64(len(p))
		}
		return p
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// LogLoss returns the mean cross-entropy of the predicted class
// probabilities (lower is better). Outputs that do not sum to one are
// normalized first. It returns 0 without recorded scores.
func (e *Evaluation) LogLoss() float64 {
	if len(e.scores) == 0 {
		return 0
	}
	var sum float64
	for i, s := range e.scores {
		p := probabilities(s)[e.classes[i]]
		sum -= math.Log(min(max(p, 1e-15), 1))
	}
	return sum / float64(len(e.scores))
}

// AUC returns the area under the ROC curve for one class against all
// others: the probability that a random sample of the class gets a higher
// score than a random sample of another class. 0.5 is chance, 1 is perfect.
// It returns 0 if the class is unknown or either group is empty.
func (e *Evaluation) AUC(label string) float64 {
	c := slices.Index(e.Labels, label)
	if c < 0 || len(e.scores) == 0 {
		return 0
	}
	type item struct {
		score    float64
		positive bool
	}
	items := make([]item, len(e.scores))
	pos := 0
	for i, s := range e.scores {
		items[i] = item{probabilities(s)[c], e.classes[i] == c}
		if items[i].positive {
			pos++
		}
	}
	neg := len(items) - pos
	if pos == 0 || neg == 0 {
		return 0
	}
	slices.SortFunc(items, func(a, b item) int {
		switch {
		case a.score < b.score:
			return -1
		case a.score > b.score:
			return 1
		}
		return 0
	})
	// Mann-Whitney U with average ranks for ties.
	var rankSum float64
	for i := 0; i < len(items); {
		j := i
		for j+1 < len(items) && items[j+1].score == items[i].score {
			j++
		}
		rank := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			if items[k].positive {
				rankSum += rank
			}
		}
		i = j + 1
	}
	return (rankSum - float64(pos*(pos+1))/2) / float64(pos*neg)
}

// MacroAUC returns the mean one-vs-rest AUC over all classes with samples.
func (e *Evaluation) MacroAUC() float64 {
	var sum float64
	n := 0
	for _, l := range e.Labels {
		if a := e.AUC(l); a > 0 {
			sum += a
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
