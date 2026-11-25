// Command sonar trains a classifier on the UCI sonar data set.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"

	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

func main() {
	dataFile := flag.String("data", "data.csv", "CSV file with the sonar data")
	networkFile := flag.String("out", "network.json", "where to store the trained network")
	flag.Parse()

	data := learn.NewSet(nn.Classification)
	if err := data.LoadCSV(*dataFile); err != nil {
		log.Fatal(err)
	}

	cfg := engine.DefaultConfig()
	cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, nil))
	e, err := engine.New(data, []int{100}, cfg)
	if err != nil {
		log.Fatal(err)
	}
	if err := e.Run(context.Background(), engine.Distance); err != nil {
		log.Fatal(err)
	}
	network, eval := e.Winner()

	if err := eval.WriteSummaries(os.Stdout); err != nil {
		log.Fatal(err)
	}
	fmt.Println("confusion matrix:")
	if err := eval.WriteConfusionMatrix(os.Stdout); err != nil {
		log.Fatal(err)
	}

	if err := persist.SaveNetwork(*networkFile, network); err != nil {
		log.Fatal(err)
	}
	loaded, err := persist.LoadNetwork(*networkFile)
	if err != nil {
		log.Fatal(err)
	}
	for _, i := range []int{0, 70, 120} {
		if i >= data.Len() {
			continue
		}
		label, err := loaded.Classify(data.Samples[i].Vector)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s -> %s\n", data.Samples[i].Label, label)
	}
}
