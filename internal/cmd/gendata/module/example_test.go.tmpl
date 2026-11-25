package synthetic_test

import (
	"fmt"
	"math"

	"github.com/breskos/gopher-learn-data/synthetic"
)

// Batch classification: read all rows, then train on them.
func Example_classification() {
	d, _ := synthetic.Lookup("classification/predictive-maintenance.csv")
	labels := map[string]bool{}
	rows := 0
	err := d.Read(".", func(r synthetic.Record) error {
		rows++
		labels[r.Target] = true
		return nil
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("features:", d.Features())
	fmt.Println("target:", d.Target, "with", len(labels), "classes")
	fmt.Println("all rows read:", rows == d.Rows)
	// Output:
	// features: [rpm temperature_c rms_g peak_g crest_factor kurtosis amp_1x_g amp_2x_g amp_bpfo_g]
	// target: state with 4 classes
	// all rows read: true
}

// Regression: targets are scaled to [0, 1].
func Example_regression() {
	d, _ := synthetic.Lookup("regression/friedman1.csv")
	lo, hi := math.Inf(1), math.Inf(-1)
	_ = d.Read(".", func(r synthetic.Record) error {
		var y float64
		_, _ = fmt.Sscan(r.Target, &y)
		lo, hi = math.Min(lo, y), math.Max(hi, y)
		return nil
	})
	fmt.Println("features:", len(d.Features()))
	fmt.Println("targets within [0, 1]:", lo >= 0 && hi <= 1)
	// Output:
	// features: 10
	// targets within [0, 1]: true
}

// Online learning: read the stream in order, one row at a time. The
// concept column shows where the drift happens.
func Example_online() {
	d, _ := synthetic.Lookup("online/abrupt-drift.csv")
	var concepts []string
	_ = d.Read(".", func(r synthetic.Record) error {
		c := r.Ignored["concept"]
		if len(concepts) == 0 || concepts[len(concepts)-1] != c {
			concepts = append(concepts, c) // the concept changed
		}
		// feed r.Features and r.Target to an online learner here
		return nil
	})
	fmt.Println("ordered stream:", d.Ordered)
	fmt.Println("concepts in order:", concepts)
	// Output:
	// ordered stream: true
	// concepts in order: [1 2]
}

// Text: one utterance per row for the string encoders.
func Example_text() {
	d, _ := synthetic.Lookup("text/intents.phrase")
	intents := map[string]bool{}
	_ = d.Read(".", func(r synthetic.Record) error {
		intents[r.Target] = true
		return nil
	})
	fmt.Println("intents:", len(intents))
	// Output:
	// intents: 4
}

// Loading: missing values arrive as NaN, LIBSVM rows as sparse maps.
func Example_loading() {
	missing, _ := synthetic.Lookup("loading/missing-values.csv")
	nan := 0
	_ = missing.Read(".", func(r synthetic.Record) error {
		for _, v := range r.Features {
			if math.IsNaN(v) {
				nan++
			}
		}
		return nil
	})
	sparse, _ := synthetic.Lookup("loading/sparse.svm")
	most := 0
	_ = sparse.Read(".", func(r synthetic.Record) error {
		most = max(most, len(r.Sparse))
		return nil
	})
	fmt.Println("has missing values:", nan > 0)
	fmt.Println("sparse rows with at most", most, "values")
	// Output:
	// has missing values: true
	// sparse rows with at most 10 values
}
