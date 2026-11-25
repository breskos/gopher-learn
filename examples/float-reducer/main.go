// Command float-reducer shows how the float reducer removes redundant dimensions.
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/breskos/gopher-learn/encoders"
)

const dimension = "measurements"

func main() {
	enc := encoders.New("float reducer")
	in := encoders.NewInput(encoders.Floats)
	// Dimension 1 rises with dimension 0 (rank correlation 1) and is dropped.
	// Dimension 3 is constant and carries no information, so it is dropped too.
	in.AddFloats([]float64{1.0, 0.0, 3.7, 5})
	in.AddFloats([]float64{2.0, 0.9, 0.3, 5})
	in.AddFloats([]float64{3.0, 1.6, 1.3, 5})
	in.AddFloats([]float64{4.0, 2.9, 4.2, 5})

	if err := enc.Scan(dimension, in, encoders.FloatReducer); err != nil {
		log.Fatal(err)
	}
	if err := enc.Fit(dimension, in); err != nil {
		log.Fatal(err)
	}
	if err := enc.Explain(os.Stdout); err != nil {
		log.Fatal(err)
	}
	vector, err := enc.EncodeFloats(dimension, []float64{1.0, 0.0, 3.7, 5})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("encoded vector:", vector)
}
