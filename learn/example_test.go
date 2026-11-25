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

func ExampleSet_ReadCSV_header() {
	csv := `id,acidity,sugar,quality
1,7.4,1.9,0.5
2,7.8,2.6,0.6
`
	set := learn.NewSet(nn.Regression)
	err := set.ReadCSV(strings.NewReader(csv), learn.WithTarget("quality"), learn.WithIgnore("id"))
	if err != nil {
		panic(err)
	}
	fmt.Println("features:", set.Features)
	fmt.Println(set.Samples[1].Vector, "->", set.Samples[1].Value)
	// Output:
	// features: [acidity sugar]
	// [7.8 2.6] -> 0.6
}

func ExampleFitScaler() {
	set := learn.NewSet(nn.Regression)
	_ = set.Add(learn.NewRegressionSample([]float64{10, 0.1}, 0.2))
	_ = set.Add(learn.NewRegressionSample([]float64{30, 0.3}, 0.4))
	_ = set.Add(learn.NewRegressionSample([]float64{20, 0.2}, 0.6))

	// Fit on training data only, then apply the same scaler everywhere.
	scaler, err := learn.FitScaler(set, learn.MinMax)
	if err != nil {
		panic(err)
	}
	scaled, _ := scaler.TransformSet(set)
	for _, s := range scaled.Samples {
		fmt.Printf("[%.2f %.2f]\n", s.Vector[0], s.Vector[1])
	}
	v, _ := scaler.Transform([]float64{25, 0.25})
	fmt.Printf("new sample: [%.2f %.2f]\n", v[0], v[1])
	// Output:
	// [0.00 0.00]
	// [1.00 1.00]
	// [0.50 0.50]
	// new sample: [0.75 0.75]
}

func ExampleSet_KFold() {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("a")
	_ = set.AddClass("b")
	for i := range 12 {
		label := "a"
		if i%3 == 0 {
			label = "b"
		}
		_ = set.Add(learn.NewClassificationSample([]float64{float64(i)}, label))
	}
	folds, err := set.KFold(4, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		panic(err)
	}
	for i, f := range folds {
		d := f.Test.Distribution()
		fmt.Printf("fold %d: train %d, test %d (a=%d, b=%d)\n", i, f.Train.Len(), f.Test.Len(), d["a"], d["b"])
	}
	// Output:
	// fold 0: train 9, test 3 (a=2, b=1)
	// fold 1: train 9, test 3 (a=2, b=1)
	// fold 2: train 9, test 3 (a=2, b=1)
	// fold 3: train 9, test 3 (a=2, b=1)
}
