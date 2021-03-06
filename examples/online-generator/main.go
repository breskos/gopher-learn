package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/breskos/gopher-learn/learn"
	neural "github.com/breskos/gopher-learn/net"
	"github.com/breskos/gopher-learn/online"
)

const (
	classLabelY    = "Y"
	classLabelN    = "N"
	numberOfInputs = 7
	hiddenNeurons  = 30
	onlineFile     = "online_learner.json"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	set := learn.NewSet(neural.Classification)
	set.AddClass(classLabelY) // class on index 0
	set.AddClass(classLabelN) // class on index 1
	classes := []string{classLabelY, classLabelN}
	o := online.NewOnline(neural.Classification, numberOfInputs, []int{hiddenNeurons}, set)
	// o.SetConfig(&online.Config{}) you can also make use of the Config to fine tune the internals
	// you can set Verbose to true to gain more insights
	o.SetVerbose(true)
	fmt.Printf("set: %v\n", set)
	for i := 0; i < 2000; i++ {
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
	// The functions below allow you to save the state of the Online learner to and to read them from file
	// in order to continue with the work.
	// persist.OnlineToFile(onlineFile, o)
	// o, err := persist.OnlineFromFile(onlineFile)
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
