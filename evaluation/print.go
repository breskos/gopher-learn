package evaluation

import "fmt"

// PrintConfusionMatrix prints the confusion matrix of the evaluation
func (e *Evaluation) PrintConfusionMatrix() {
	fmt.Printf("\t|")
	for k := range e.Confusion {
		fmt.Printf("%v\t|", k)
	}
	fmt.Print("\n")
	for cl := range e.Confusion {
		fmt.Printf("%v\t|", cl)
		for c := range e.Confusion[cl] {
			fmt.Printf("%v\t|", e.Confusion[cl][c])
		}
		fmt.Printf("\n")
	}

}

// PrintSummaries prints the summaries of all classes
func (e *Evaluation) PrintSummaries() {
	for class := range e.Confusion {
		e.PrintSummary(class)
	}
}

// PrintRegressionSummary returns a summary of the evaluated regression
func (e *Evaluation) PrintRegressionSummary() {
	fmt.Println("summary")
	fmt.Printf("correct: %v\n", e.Correct)
	fmt.Printf("wrong: %v\n", e.Wrong)
	fmt.Printf("ratio: %v\n", float64(e.Correct)/float64(e.Correct+e.Wrong))
}

// PrintSummary returns a summary
func (e *Evaluation) PrintSummary(label string) {
	fmt.Printf("summary for class %v\n", label)
	fmt.Printf(" * TP: %v TN: %v FP: %v FN: %v\n", e.GetTruePositives(label), e.GetTrueNegatives(label), e.GetFalsePositives(label), e.GetFalseNegatives(label))
	fmt.Printf(" * Recall/Sensitivity: %.3f\n", e.GetRecall(label))
	fmt.Printf(" * Precision: %.3f\n", e.GetPrecision(label))
	fmt.Printf(" * Fallout/FalsePosRate: %.3f\n", e.GetFallout(label))
	fmt.Printf(" * False Discovey Rate: %.3f\n", e.GetFalseDiscoveryRate(label))
	fmt.Printf(" * Negative Prediction Rate: %.3f\n", e.GetNegativePredictionValue(label))
	fmt.Println("--")
	fmt.Printf(" * Accuracy: %.3f\n", e.GetAccuracy(label))
	fmt.Printf(" * F-Measure: %.3f\n", e.GetFMeasure(label))
	fmt.Printf(" * Balanced Accuracy: %.3f\n", e.GetBalancedAccuracy(label))
	fmt.Printf(" * Informedness: %.3f\n", e.GetInformedness(label))
	fmt.Printf(" * Markedness: %.3f\n", e.GetMarkedness(label))

}
