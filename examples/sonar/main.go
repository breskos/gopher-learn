package main

import (
	"fmt"

	neural "github.com/breskos/gopher-learn"
	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/persist"
)

const (
	dataFile      = "data.csv"
	networkFile   = "network.json"
	dataSetFile   = "set.json"
	tries         = 1
	epochs        = 100
	trainingSplit = 0.7
	learningRate  = 0.4
	decay         = 0.005
)

func main() {
	data := learn.NewSet(neural.Classification)
	ok, err := data.LoadFromCSV(dataFile)
	if !ok || nil != err {
		fmt.Printf("something went wrong -> %v", err)
	}
	e := engine.NewEngine(neural.Classification, []int{100}, data)
	e.SetVerbose(true)
	e.Start(neural.CriterionDistance, tries, epochs, trainingSplit, learningRate, decay)
	network, evaluation := e.GetWinner()

	evaluation.PrintSummary("R")
	fmt.Println()
	evaluation.PrintSummary("M")

	err = persist.SetToFile(dataSetFile, data)
	if err != nil {
		fmt.Printf("error while saving data set: %v\n", err)
	}
	err = persist.ToFile(networkFile, network)
	if err != nil {
		fmt.Printf("error while saving network: %v\n", err)
	}

	network2, err := persist.FromFile(networkFile)
	if err != nil {
		fmt.Printf("error while loading network: %v\n", err)
	}
	data2, err := persist.SetFromFile(dataSetFile)
	if err != nil {
		fmt.Printf("error while loading data set from file: %v\n", err)
	}
	fmt.Printf("data2: %v", data2)

	w := network2.CalculateWinnerLabel(data.Samples[0].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[0].Label, w)
	w = network2.CalculateWinnerLabel(data.Samples[70].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[70].Label, w)
	w = network2.CalculateWinnerLabel(data.Samples[120].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[120].Label, w)

	// print confusion matrix
	fmt.Println(" * Confusion Matrix *")
	evaluation.PrintConfusionMatrix()
}
