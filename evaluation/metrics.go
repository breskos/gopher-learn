package evaluation

// TruePositives returns the number of samples of class label predicted as label.
func (e *Evaluation) TruePositives(label string) int {
	return e.Confusion[label][label]
}

// FalsePositives returns the number of samples of other classes predicted as label.
func (e *Evaluation) FalsePositives(label string) int {
	s := 0
	for actual, row := range e.Confusion {
		if actual != label {
			s += row[label]
		}
	}
	return s
}

// FalseNegatives returns the number of samples of class label predicted as something else.
func (e *Evaluation) FalseNegatives(label string) int {
	s := 0
	for predicted, c := range e.Confusion[label] {
		if predicted != label {
			s += c
		}
	}
	return s
}

// TrueNegatives returns the number of samples of other classes not predicted as label.
func (e *Evaluation) TrueNegatives(label string) int {
	s := 0
	for actual, row := range e.Confusion {
		if actual == label {
			continue
		}
		for predicted, c := range row {
			if predicted != label {
				s += c
			}
		}
	}
	return s
}

// Positives returns TP + FN.
func (e *Evaluation) Positives(label string) int {
	return e.TruePositives(label) + e.FalseNegatives(label)
}

// Negatives returns FP + TN.
func (e *Evaluation) Negatives(label string) int {
	return e.FalsePositives(label) + e.TrueNegatives(label)
}

// Accuracy returns (TP + TN) / (P + N).
func (e *Evaluation) Accuracy(label string) float64 {
	return ratio(e.TruePositives(label)+e.TrueNegatives(label), e.Positives(label)+e.Negatives(label))
}

// Recall returns TP / P. Also known as sensitivity or true positive rate.
func (e *Evaluation) Recall(label string) float64 {
	return ratio(e.TruePositives(label), e.Positives(label))
}

// Specificity returns TN / N.
func (e *Evaluation) Specificity(label string) float64 {
	return ratio(e.TrueNegatives(label), e.Negatives(label))
}

// Precision returns TP / (TP + FP).
func (e *Evaluation) Precision(label string) float64 {
	return ratio(e.TruePositives(label), e.TruePositives(label)+e.FalsePositives(label))
}

// Fallout returns FP / N. Also known as false positive rate.
func (e *Evaluation) Fallout(label string) float64 {
	return ratio(e.FalsePositives(label), e.Negatives(label))
}

// FalseDiscoveryRate returns FP / (FP + TP).
func (e *Evaluation) FalseDiscoveryRate(label string) float64 {
	return ratio(e.FalsePositives(label), e.FalsePositives(label)+e.TruePositives(label))
}

// NegativePredictiveValue returns TN / (TN + FN).
func (e *Evaluation) NegativePredictiveValue(label string) float64 {
	return ratio(e.TrueNegatives(label), e.TrueNegatives(label)+e.FalseNegatives(label))
}

// FMeasure returns the F1 score 2TP / (2TP + FP + FN).
func (e *Evaluation) FMeasure(label string) float64 {
	tp := e.TruePositives(label)
	return ratio(2*tp, 2*tp+e.FalsePositives(label)+e.FalseNegatives(label))
}

// BalancedAccuracy returns (TP/P + TN/N) / 2.
func (e *Evaluation) BalancedAccuracy(label string) float64 {
	return (e.Recall(label) + e.Specificity(label)) / 2
}

// Informedness returns sensitivity + specificity - 1.
func (e *Evaluation) Informedness(label string) float64 {
	return e.Recall(label) + e.Specificity(label) - 1
}

// Markedness returns precision + negative predictive value - 1.
func (e *Evaluation) Markedness(label string) float64 {
	return e.Precision(label) + e.NegativePredictiveValue(label) - 1
}

// OverallAccuracy returns the mean accuracy over all classes.
func (e *Evaluation) OverallAccuracy() float64 { return e.mean(e.Accuracy) }

// OverallBalancedAccuracy returns the mean balanced accuracy over all classes.
func (e *Evaluation) OverallBalancedAccuracy() float64 { return e.mean(e.BalancedAccuracy) }

// OverallFMeasure returns the macro-averaged F1 score over all classes.
func (e *Evaluation) OverallFMeasure() float64 { return e.mean(e.FMeasure) }

func (e *Evaluation) mean(metric func(string) float64) float64 {
	if len(e.Confusion) == 0 {
		return 0
	}
	var sum float64
	for c := range e.Confusion {
		sum += metric(c)
	}
	return sum / float64(len(e.Confusion))
}
