package evaluation_test

import (
	"fmt"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/nn"
)

func ExampleEvaluation() {
	e := evaluation.New(nn.Classification, []string{"cat", "dog"})
	for _, p := range [][2]string{
		{"cat", "cat"}, {"cat", "cat"}, {"cat", "dog"},
		{"dog", "dog"}, {"dog", "dog"}, {"dog", "dog"}, {"dog", "cat"},
	} {
		e.Add(p[0], p[1]) // actual, predicted
	}
	fmt.Printf("precision(cat): %.2f\n", e.Precision("cat"))
	fmt.Printf("recall(cat):    %.2f\n", e.Recall("cat"))
	fmt.Printf("F1 (macro):     %.2f\n", e.OverallFMeasure())
	fmt.Printf("correct ratio:  %.2f\n", e.CorrectRatio())
	// Output:
	// precision(cat): 0.67
	// recall(cat):    0.67
	// F1 (macro):     0.71
	// correct ratio:  0.71
}

func ExampleEvaluation_AddRegression() {
	e := evaluation.New(nn.Regression, nil)
	e.Threshold = 0.1
	e.AddRegression(0.50, 0.55) // within threshold
	e.AddRegression(0.80, 0.60) // too far off
	fmt.Println(e.Correct, e.Wrong)
	// Output:
	// 1 1
}
