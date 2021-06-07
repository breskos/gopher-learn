package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	neural "github.com/breskos/gopher-learn/net"
	"github.com/breskos/gopher-learn/persist"
)

const (
	dataFile      = "data.phrase"
	delimiter     = "#"
	tries         = 1
	epochs        = 100
	trainingSplit = 0.7
	learningRate  = 0.4
	decay         = 0.005
	modelName     = "answer-type"
)

/*
Different states to test dictionary

*/
func main() {
	data := getData()
	e := encoders.NewEncoder("dictionary")
	set := encoders.NewInput(modelName, encoders.String)
	for _, v := range data {
		splitted := strings.Split(v, delimiter)
		set.AddString(splitted[0])
	}
	e.Scan(modelName, set, encoders.Automatic)
	e.Transform(modelName, set)
	e.Explain()

	// this is just an example how to persist the encoders
	persist.EncoderToFile("encoder.json", e)
	e2, err := persist.EncoderFromFile("encoder.json")
	if err != nil {
		log.Fatalf("error persisting encoder: %v", err)
	}
	fmt.Println("after persisting")
	e2.Explain()

	// TODO(abresk) here I noticed that Sample, Set in learn are not optimized to be used in this manner
	// They were designed to load samples from file.
	// Get vectors for training
	output := make(map[string]int)
	output["FACT"] = 0
	output["PARAGRAPH"] = 1
	output["LIST"] = 2
	learningSet := learn.NewSet(neural.Classification)
	learningSet.AddClass("FACT")      // 0
	learningSet.AddClass("PARAGRAPH") // 1
	learningSet.AddClass("LIST")      // 2

	for _, v := range data {
		splitted := strings.Split(v, delimiter)
		input := encoders.Unified{String: splitted[0]}
		vector := e.Encode(modelName, input)
		outVec := []float64{0.0, 0.0, 0.0}
		outVec[output[splitted[1]]] = 1.0
		learningSet.AddSample(learn.NewClassificationSample(vector, outVec, splitted[1]))
	}

	// training the network
	en := engine.NewEngine(neural.Classification, []int{80}, learningSet)
	en.SetVerbose(true)
	en.SetConfig(&engine.Config{
		Tries:         tries,
		Epochs:        epochs,
		TrainingSplit: trainingSplit,
		LearningRate:  learningRate,
		Decay:         decay,
	})
	en.Start(neural.Distance)
	network, evaluation := en.GetWinner()
	evaluation.PrintSummary("FACT")
	fmt.Println()
	evaluation.PrintSummary("PARAGRAPH")
	fmt.Println()
	evaluation.PrintSummary("LIST")

	// testing with own example
	vector := e2.Encode(modelName, encoders.Unified{String: "Wieviel Saft ist drin?"})
	w := network.CalculateWinnerLabel(vector)
	fmt.Printf("%v -> %v\n", "FACT", w)

	vector = e2.Encode(modelName, encoders.Unified{String: "Welche Optionen gibt es um einen Org zu besiegen?"})
	w = network.CalculateWinnerLabel(vector)
	fmt.Printf("%v -> %v\n", "LIST", w)

	vector = e2.Encode(modelName, encoders.Unified{String: "Was ist ein Haus?"})
	w = network.CalculateWinnerLabel(vector)
	fmt.Printf("%v -> %v\n", "PARAGRAPH", w)

	vector = e2.Encode(modelName, encoders.Unified{String: "Woraus besteht ein Garten?"})
	w = network.CalculateWinnerLabel(vector)
	fmt.Printf("%v -> %v\n", "LIST", w)

}

func getData() []string {
	file, err := os.Open("data.phrase")

	if err != nil {
		log.Fatalf("failed to open")

	}
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	var text []string
	for scanner.Scan() {
		text = append(text, scanner.Text())
	}
	file.Close()
	return text
}
