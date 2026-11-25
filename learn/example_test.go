package learn_test

import (
	"fmt"
	"math/rand/v2"
	"strings"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func ExampleSet_ReadCSV() {
	csv := `5.1,3.5,setosa
7.0,3.2,versicolor
6.3,3.3,virginica
4.9,3.0,setosa
`
	set := learn.NewSet(nn.Classification)
	if err := set.ReadCSV(strings.NewReader(csv)); err != nil {
		panic(err)
	}
	fmt.Println("samples:", set.Len())
	fmt.Println("features:", set.InputSize())
	fmt.Println("classes:", set.Labels)
	fmt.Println("target of first sample:", set.Samples[0].Output)
	// Output:
	// samples: 4
	// features: 2
	// classes: [setosa versicolor virginica]
	// target of first sample: [1 0 0]
}

func ExampleSet_ReadSVM() {
	svm := `+1 1:0.5 3:1.0
-1 2:0.25
`
	set := learn.NewSet(nn.Classification)
	if err := set.ReadSVM(strings.NewReader(svm)); err != nil {
		panic(err)
	}
	for _, s := range set.Samples {
		fmt.Println(s.Label, s.Vector)
	}
	// Output:
	// +1 [0.5 0 1]
	// -1 [0 0.25 0]
}

func ExampleSet_Add() {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("spam")
	_ = set.AddClass("ham")

	if err := set.Add(learn.NewClassificationSample([]float64{0.9, 0.1}, "ham")); err != nil {
		panic(err)
	}
	err := set.Add(learn.NewClassificationSample([]float64{0.2, 0.3}, "unknown"))
	fmt.Println(set.Samples[0].Output)
	fmt.Println(err)
	// Output:
	// [0 1]
	// learn: unknown class: "unknown"
}

func ExampleSet_Split() {
	set := learn.NewSet(nn.Regression)
	for i := range 10 {
		_ = set.Add(learn.NewRegressionSample([]float64{float64(i)}, float64(i)/10))
	}
	train, test, err := set.Split(0.8, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		panic(err)
	}
	fmt.Println(train.Len(), test.Len())
	// Output:
	// 8 2
}
