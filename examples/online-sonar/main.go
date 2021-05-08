package main

import (
	"fmt"

	neural "github.com/breskos/gopher-learn"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/online"
	"github.com/breskos/gopher-learn/persist"
)

const (
	dataFile      = "data.csv"
	networkFile   = "network.json"
	dataSetFile   = "set.json"
	hiddenNeurons = 30
)

func main() {
	// data from file that we want to stream in
	data := learn.NewSet(neural.Classification)
	ok, err := data.LoadFromCSV(dataFile)
	if !ok || nil != err {
		fmt.Printf("something went wrong -> %v", err)
	}
	// we create an empty set with correct number of inputs and classes
	onlineSet := learn.NewSet(neural.Classification)
	onlineSet.AddClass("R")
	onlineSet.AddClass("M")

	o := online.NewOnline(neural.Classification, len(data.Samples[0].Vector), []int{hiddenNeurons}, onlineSet)
	o.SetVerbose(true)

	l := len(data.Samples)
	for i := 0; i < l; i++ {
		o.Inject(data.Samples[i], false)
		if i%5 == 0 {
			fmt.Printf("\n\nAFTER INJECTING %d samples\n", i)
			o.Iterate() // this function also returns the F-Measure

		}
	}

	err = persist.SetToFile(dataSetFile, o.Data)
	if err != nil {
		fmt.Printf("error while saving data set: %v\n", err)
	}
	err = persist.ToFile(networkFile, o.Network)
	if err != nil {
		fmt.Printf("error while saving network: %v\n", err)
	}

	network2, err := persist.FromFile(networkFile)
	if err != nil {
		fmt.Printf("error while loading network: %v\n", err)
	}

	w := network2.CalculateWinnerLabel(data.Samples[0].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[0].Label, w)
	w = network2.CalculateWinnerLabel(data.Samples[70].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[70].Label, w)
	w = network2.CalculateWinnerLabel(data.Samples[120].Vector)
	fmt.Printf("%v -> %v\n", data.Samples[120].Label, w)
}
