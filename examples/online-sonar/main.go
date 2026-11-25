// Command online-sonar streams the sonar data set into an online learner.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
	"github.com/breskos/gopher-learn/persist"
)

func main() {
	dataFile := flag.String("data", "data.csv", "CSV file with the sonar data")
	learnerFile := flag.String("out", "online.json", "where to store the online learner")
	flag.Parse()

	// The full data set is only used as the source of the stream.
	stream := learn.NewSet(nn.Classification)
	if err := stream.LoadCSV(*dataFile); err != nil {
		log.Fatal(err)
	}

	// The online learner starts with an empty set that knows the classes.
	set := learn.NewSet(nn.Classification)
	for _, label := range stream.Labels {
		if err := set.AddClass(label); err != nil {
			log.Fatal(err)
		}
	}
	o, err := online.New(set, stream.InputSize(), []int{30}, online.DefaultConfig())
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	for i, s := range stream.Samples {
		sample := learn.NewClassificationSample(s.Vector, s.Label)
		if err := o.Inject(sample, false); err != nil && !errors.Is(err, online.ErrSampleExists) {
			log.Fatal(err)
		}
		if i%5 != 0 {
			continue
		}
		score, err := o.Iterate(ctx)
		if errors.Is(err, online.ErrNotEnoughData) {
			continue
		}
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("after %3d samples: F1 %.3f\n", i+1, score)
	}

	if o.LastEvaluation != nil {
		if err := o.LastEvaluation.WriteConfusionMatrix(os.Stdout); err != nil {
			log.Fatal(err)
		}
	}
	if err := persist.SaveOnline(*learnerFile, o); err != nil {
		log.Fatal(err)
	}
}
