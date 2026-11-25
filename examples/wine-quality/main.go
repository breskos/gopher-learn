// Command wine-quality trains a regressor on the UCI wine quality data set.
// The target column must be scaled to [0, 1].
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"math/rand/v2"
	"os"

	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

const threshold = 0.04 // a prediction within this distance counts as correct

func main() {
	dataFile := flag.String("data", "winequality-red.csv", "CSV file with the wine data")
	networkFile := flag.String("out", "network.json", "where to store the trained network")
	scale := flag.Bool("scale", true, "standardize the features before training")
	flag.Parse()

	data := learn.NewSet(nn.Regression)
	if err := data.LoadCSV(*dataFile); err != nil {
		log.Fatal(err)
	}

	// Hold back a test set that the training never sees.
	train, test, err := data.Split(0.8, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		log.Fatal(err)
	}

	// The features range from below 1 to several hundred. Standardizing them
	// lets the network learn from all of them. Fit the scaler on the
	// training data only and apply it to the test data as well.
	if *scale {
		scaler, err := learn.FitScaler(train, learn.Standardize)
		if err != nil {
			log.Fatal(err)
		}
		if train, err = scaler.TransformSet(train); err != nil {
			log.Fatal(err)
		}
		if test, err = scaler.TransformSet(test); err != nil {
			log.Fatal(err)
		}
	}

	cfg := engine.DefaultConfig()
	cfg.Epochs = 200
	cfg.LearningRate = 0.02
	cfg.Decay = 0.001
	cfg.RegressionThreshold = threshold
	cfg.Seed = 1
	cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, nil))

	e, err := engine.New(train, []int{50}, cfg)
	if err != nil {
		log.Fatal(err)
	}
	// Distance selects the regressor closest to the targets.
	if err := e.Run(context.Background(), engine.Distance); err != nil {
		log.Fatal(err)
	}
	network, _ := e.Winner()

	eval, err := evaluation.Evaluate(network, test, threshold)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print("test set: ")
	if err := eval.WriteRegressionSummary(os.Stdout); err != nil {
		log.Fatal(err)
	}

	if err := persist.SaveNetwork(*networkFile, network); err != nil {
		log.Fatal(err)
	}
	for _, s := range test.Samples[:3] {
		out, err := network.Predict(s.Vector)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%.3f -> %.3f\n", s.Value, out[0])
	}
}
