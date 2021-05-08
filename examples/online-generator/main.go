package main

import (
	"fmt"
	"math/rand"
	"time"

	neural "github.com/breskos/gopher-learn"
	learn "github.com/breskos/gopher-learn/learn"
	online "github.com/breskos/gopher-learn/online"
)

const (
	classLabelY    = "Y"
	classLabelN    = "N"
	numberOfInputs = 7
	hiddenNeurons  = 30
)

func main() {
	rand.Seed(time.Now().UnixNano())
	set := learn.NewSet(neural.Classification)
	set.AddClass(classLabelY) // class on index 0
	set.AddClass(classLabelN) // class on index 1
	classes := []string{classLabelY, classLabelN}
	o := online.NewOnline(neural.Classification, numberOfInputs, []int{hiddenNeurons}, set)
	// you can set Verbose to true to gain more insights
	o.SetVerbose(true)
	i := 0
	fmt.Printf("set: %v\n", set)
	for {
		class := rand.Intn(2)
		classLabel := classes[class]
		vector, target := createFeatureVector(classLabel)
		// target could also be replaced with: set.GenerateOutputVector(classLabel)
		sample := learn.NewClassificationSample(vector, target, classLabel)
		// sample := learn.NewClassificationSample(vector, target, classLabel)
		// here we inject a new sample from the generator
		// if the data points already exists in the set (we are not forcing to override it)
		o.Inject(sample, false)
		if i%20 == 0 {
			o.Iterate() // this function returns the F-Measure of the current state
		}
	}
}

func createFeatureVector(class string) ([]float64, []float64) {
	featuresY := []float64{1.0, 3.0, 10.5, 5.0, 4.0, 3.3, 5.2}
	featuresN := []float64{1.0, 8.7, 1.3, 3.3, 4.0, 10.1, 5.1}
	target := []float64{0.0, 0.0}
	var vector []float64
	if "Y" == class {
		for _, v := range featuresY {
			vector = append(vector, (v-1)+rand.Float64()*(v+1))

		}
		target[0] = 1.0
	} else {
		for _, v := range featuresN {
			vector = append(vector, (v-1)+rand.Float64()*(v+1))
		}
		target[1] = 1.0
	}
	return vector, target
}
