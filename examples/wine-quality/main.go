// Command wine-quality trains a regressor on the UCI wine quality data set.
// The target column must be scaled to [0, 1].
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
	dataFile := flag.String("data", "winequality-red.csv", "CSV file with the wine data")
	networkFile := flag.String("out", "network.json", "where to store the trained network")
	flag.Parse()

	data := learn.NewSet(nn.Regression)
	if err := data.LoadCSV(*dataFile); err != nil {
		log.Fatal(err)
	}

	cfg := engine.DefaultConfig()
	cfg.Epochs = 200
	cfg.TrainingSplit = 0.8
	cfg.LearningRate = 0.02
	cfg.Decay = 0.001
	cfg.RegressionThreshold = 0.04
	cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, nil))

	e, err := engine.New(data, []int{50}, cfg)
	if err != nil {
		log.Fatal(err)
	}
	// Distance selects the regressor closest to the targets.
	if err := e.Run(context.Background(), engine.Distance); err != nil {
		log.Fatal(err)
	}
	network, eval := e.Winner()
	if err := eval.WriteRegressionSummary(os.Stdout); err != nil {
		log.Fatal(err)
	}

	if err := persist.SaveNetwork(*networkFile, network); err != nil {
		log.Fatal(err)
	}
	for _, i := range []int{0, 52, 180} {
		if i >= data.Len() {
			continue
		}
		out, err := network.Predict(data.Samples[i].Vector)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%.3f -> %.3f\n", data.Samples[i].Value, out[0])
	}
}
