// Command ngram classifies questions by answer type using an n-gram encoder.
//
// The data file contains one "<question>#<label>" per line, with the labels
// FACT, PARAGRAPH and LIST.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

const dimension = "question"

func main() {
	dataFile := flag.String("data", "data.phrase", "file with <question>#<label> lines")
	encoderFile := flag.String("encoder", "encoder.json", "where to store the encoder")
	flag.Parse()

	rows, err := readRows(*dataFile)
	if err != nil {
		log.Fatal(err)
	}

	enc := encoders.New("questions")
	in := encoders.NewInput(encoders.String)
	for _, r := range rows {
		in.AddString(r[0])
	}
	if err := enc.Scan(dimension, in, encoders.StringNGrams); err != nil {
		log.Fatal(err)
	}
	if err := enc.Fit(dimension, in); err != nil {
		log.Fatal(err)
	}
	if err := persist.SaveEncoder(*encoderFile, enc); err != nil {
		log.Fatal(err)
	}
	if err := enc.Explain(os.Stdout); err != nil {
		log.Fatal(err)
	}

	set := learn.NewSet(nn.Classification)
	for _, label := range []string{"FACT", "PARAGRAPH", "LIST"} {
		if err := set.AddClass(label); err != nil {
			log.Fatal(err)
		}
	}
	for _, r := range rows {
		vector, err := enc.EncodeString(dimension, r[0])
		if err != nil {
			log.Fatal(err)
		}
		sample := learn.NewClassificationSample(vector, r[1])
		if set.Contains(sample) {
			continue
		}
		if err := set.Add(sample); err != nil {
			log.Fatal(err)
		}
	}

	e, err := engine.New(set, []int{80}, engine.DefaultConfig())
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

	loaded, err := persist.LoadEncoder(*encoderFile)
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{
		"Wieviel Saft ist drin?",
		"Welche Optionen gibt es um einen Ork zu besiegen?",
		"Was ist ein Haus?",
		"Woraus besteht ein Garten?",
	} {
		vector, err := loaded.EncodeString(dimension, q)
		if err != nil {
			log.Fatal(err)
		}
		label, err := network.Classify(vector)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s -> %s\n", q, label)
	}
}

func readRows(path string) ([][2]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var rows [][2]string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		question, label, ok := strings.Cut(sc.Text(), "#")
		if !ok {
			continue
		}
		rows = append(rows, [2]string{question, strings.TrimSpace(label)})
	}
	return rows, sc.Err()
}
