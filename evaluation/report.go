package evaluation

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
)

// WriteConfusionMatrix writes the confusion matrix with actual classes as
// rows and predicted classes as columns.
func (e *Evaluation) WriteConfusionMatrix(w io.Writer) error {
	classes := e.Classes()
	var b strings.Builder
	b.WriteString("actual \\ predicted\t")
	for _, c := range classes {
		b.WriteString(c + "\t")
	}
	b.WriteString("\n")
	for _, actual := range classes {
		b.WriteString(actual + "\t")
		for _, predicted := range classes {
			fmt.Fprintf(&b, "%d\t", e.Confusion[actual][predicted])
		}
		b.WriteString("\n")
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	if _, err := io.WriteString(tw, b.String()); err != nil {
		return err
	}
	return tw.Flush()
}

// WriteSummary writes all metrics of one class.
func (e *Evaluation) WriteSummary(w io.Writer, label string) error {
	_, err := fmt.Fprintf(w, `summary for class %s
  TP: %d  TN: %d  FP: %d  FN: %d
  recall:              %.3f
  precision:           %.3f
  fallout:             %.3f
  false discovery:     %.3f
  neg. predictive:     %.3f
  accuracy:            %.3f
  F1:                  %.3f
  balanced accuracy:   %.3f
  informedness:        %.3f
  markedness:          %.3f
`,
		label,
		e.TruePositives(label), e.TrueNegatives(label), e.FalsePositives(label), e.FalseNegatives(label),
		e.Recall(label), e.Precision(label), e.Fallout(label), e.FalseDiscoveryRate(label),
		e.NegativePredictiveValue(label), e.Accuracy(label), e.FMeasure(label),
		e.BalancedAccuracy(label), e.Informedness(label), e.Markedness(label))
	return err
}

// WriteSummaries writes the summary of every class.
func (e *Evaluation) WriteSummaries(w io.Writer) error {
	for _, c := range e.Classes() {
		if err := e.WriteSummary(w, c); err != nil {
			return err
		}
	}
	return nil
}

// WriteRegressionSummary writes the results of a regression evaluation.
func (e *Evaluation) WriteRegressionSummary(w io.Writer) error {
	_, err := fmt.Fprintf(w, "correct: %d  wrong: %d  ratio: %.3f  distance: %.4f (threshold %v)\n",
		e.Correct, e.Wrong, e.CorrectRatio(), e.TotalDistance, e.Threshold)
	return err
}
