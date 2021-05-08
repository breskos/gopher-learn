package main

import (
	"fmt"

	neural "github.com/breskos/gopher-learn"
	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/persist"
)

const (
	dataFile            = "winequality-red.csv"
	networkFile         = "network.json"
	tries               = 1
	epochs              = 200
	trainingSplit       = 0.8
	learningRate        = 0.02
	decay               = 0.001
	hiddenNeurons       = 50
	regressionThreshold = 0.04 // helps evaluation to define between wrong or right
)

func main() {
	data := learn.NewSet(neural.Regression)
	ok, err := data.LoadFromCSV(dataFile)
	if !ok || nil != err {
		fmt.Printf("something went wrong -> %v", err)
	}
	e := engine.NewEngine(neural.Regression, []int{hiddenNeurons}, data)
	e.SetRegressionThreshold(regressionThreshold)
	e.SetVerbose(true)
	// here we ware choosing Distance because we want the regressor that produces the best examples
	e.Start(neural.Distance, tries, epochs, trainingSplit, learningRate, decay)
	network, evaluation := e.GetWinner()

	// regression evaluation
	evaluation.PrintRegressionSummary()

	err = persist.ToFile(networkFile, network)
	if err != nil {
		fmt.Printf("error while saving network: %v\n", err)
	}
	// persisted network
	network2, err := persist.FromFile(networkFile)
	if err != nil {
		fmt.Printf("error while loading network: %v\n", err)
	}

	// some examples
	w := network2.Calculate(data.Samples[0].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[0].Value, w)
	w = network2.Calculate(data.Samples[52].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[52].Value, w)
	w = network2.Calculate(data.Samples[180].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[189].Value, w)
}
