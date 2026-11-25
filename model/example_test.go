package model_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"github.com/breskos/gopher-learn/model"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

// Train a model on raw CSV records, save it as one file, load it again and
// predict from a raw record.
func Example() {
	// Machines with a temperature, a vibration level and a vendor; machines
	// of vendor "acme" run hot without failing.
	var csv strings.Builder
	csv.WriteString("machine,temperature,vibration,vendor,status\n")
	rng := rand.New(rand.NewPCG(1, 1))
	for i := range 400 {
		temp := 40 + rng.Float64()*60
		vib := rng.Float64() * 10
		vendor := []string{"acme", "globex", "initech"}[rng.IntN(3)]
		limit := 80.0
		if vendor == "acme" {
			limit = 95
		}
		status := "ok"
		if temp > limit || vib > 8 {
			status = "failing"
		}
		fmt.Fprintf(&csv, "m%d,%.1f,%.2f,%s,%s\n", i, temp, vib, vendor, status)
	}

	dir, err := os.MkdirTemp("", "model-example")
	if err != nil {
		panic(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	data := filepath.Join(dir, "machines.csv")
	if err := os.WriteFile(data, []byte(csv.String()), 0o644); err != nil {
		panic(err)
	}

	records, err := model.LoadRecords(data)
	if err != nil {
		panic(err)
	}
	m, _, err := model.Train(context.Background(), records, model.Config{
		Target:           "status",
		Ignore:           []string{"machine"},
		HiddenActivation: nn.Tanh,
		Fit:              nn.FitConfig{Epochs: 300},
		Rand:             rng,
	})
	if err != nil {
		panic(err)
	}
	for _, c := range m.Columns {
		fmt.Println(c.Name, c.Kind)
	}

	path := filepath.Join(dir, "model.json")
	if err := persist.SaveModel(path, m); err != nil {
		panic(err)
	}
	loaded, err := persist.LoadModel(path)
	if err != nil {
		panic(err)
	}
	for _, r := range []model.Record{
		{"temperature": "90", "vibration": "2", "vendor": "acme"},
		{"temperature": "90", "vibration": "2", "vendor": "globex"},
		{"temperature": "50", "vibration": "9.5", "vendor": "initech"},
	} {
		p, err := loaded.Predict(r)
		if err != nil {
			panic(err)
		}
		fmt.Println(r["vendor"], p.Label)
	}
	// Output:
	// temperature numeric
	// vendor categorical
	// vibration numeric
	// acme ok
	// globex failing
	// initech failing
}

// Predict a house price; the model scales the target for training and
// returns predictions in the original units.
func ExampleTrain_regression() {
	rng := rand.New(rand.NewPCG(2, 2))
	var records []model.Record
	for range 500 {
		area := 30 + rng.Float64()*120
		price := 100000 + 3000*area
		records = append(records, model.Record{"area": fmt.Sprint(area), "price": fmt.Sprint(price)})
	}
	m, _, err := model.Train(context.Background(), records, model.Config{
		Target: "price", Task: nn.Regression, Hidden: []int{}, Rand: rng,
		Fit: nn.FitConfig{Epochs: 300, LearningRate: 0.05},
	})
	if err != nil {
		panic(err)
	}
	p, err := m.Predict(model.Record{"area": "100"})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%.0fk\n", p.Value/1000)
	// Output: 400k
}
