package encoders_test

import (
	"fmt"
	"os"

	"github.com/breskos/gopher-learn/encoders"
)

func Example() {
	enc := encoders.New("device")
	in := encoders.NewInput(encoders.String)
	for _, s := range []string{"ON", "off", "Standby", "on"} {
		in.AddString(s)
	}
	// Fit scans automatically and picks a dictionary for few distinct values.
	if err := enc.Fit("state", in); err != nil {
		panic(err)
	}
	_ = enc.Explain(os.Stdout)

	v, _ := enc.EncodeString("state", "Standby!")
	fmt.Println(v)
	// Output:
	// [state] string input, string-dictionary, 3 dimensions
	// [0 0 1]
}

func ExampleFloatReducerModel() {
	enc := encoders.New("sensors")
	in := encoders.NewInput(encoders.Floats)
	// Dimension 1 rises with dimension 0, dimension 3 is constant.
	in.AddFloats([]float64{1, 10, 0.3, 7})
	in.AddFloats([]float64{2, 20, 0.1, 7})
	in.AddFloats([]float64{3, 30, 0.4, 7})
	in.AddFloats([]float64{4, 40, 0.2, 7})
	if err := enc.Scan("readings", in, encoders.FloatReducer); err != nil {
		panic(err)
	}
	if err := enc.Fit("readings", in); err != nil {
		panic(err)
	}
	v, _ := enc.EncodeFloats("readings", []float64{5, 50, 0.9, 7})
	fmt.Println(v)
	// Output:
	// [5 0.9]
}

func ExampleNGramModel() {
	enc := encoders.New("words")
	in := encoders.NewInput(encoders.String)
	in.AddString("hello")
	in.AddString("help")
	if err := enc.Scan("word", in, encoders.StringNGrams); err != nil {
		panic(err)
	}
	if err := enc.Fit("word", in); err != nil {
		panic(err)
	}
	m := enc.Dimensions["word"].Model.(*encoders.NGramModel)
	fmt.Println(m.Grams)
	v, _ := enc.EncodeString("word", "hello")
	fmt.Println(v)
	// Output:
	// [hel ell elp llo]
	// [1 1 0 1]
}
