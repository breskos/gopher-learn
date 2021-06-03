package encoders

import (
	"fmt"
	"log"
)

type EncoderType int

const (
	// Automatic means that the encoder decides based on heuristics what to do
	Automatic EncoderType = iota
	// StringDictionary uses exact matches on strings as dictionary approach
	StringDictionary
	// StringSplittedDictionary
	StringSplitDictionary
	// StringTopics uses topic modelling on strings
	StringTopics
	// StringNGrams uses N-Gram modelling on strings
	StringNGrams
	// FloatExact just uses the float value it gets from input
	FloatExact
	// FloatReducer reduces a large number of floats to a smaller input space
	FloatReducer
)

func (e EncoderType) String() string {
	return [...]string{
		"Automatic",
		"StringDictionary",
		"StringSplitDictionary",
		"StringTopics",
		"StringNGrams",
		"FloatExact",
		"FloatReducer",
	}[e]
}

type EncoderModel interface {
	Fit(*Input)
	CalculateString(string) []float64
	CalculateFloats([]float64) []float64
	GetDimensions() int
	GetQuality() float64
	Name() string
}

type Encoder struct {
	// Name of the encoder
	Name string
	// Dimensions hold the EncoderModel for the dimensions
	Models map[string]*Dimension
	// Config of the Encoder
	Config *EncoderConfig
	// Scanned determines if scan was executed
	Scanned bool
}

type Dimension struct {
	Inputs    int
	InputType InputType
	Type      EncoderType
	Model     EncoderModel
}

func NewEncoder(name string) *Encoder {
	return &Encoder{
		Name:   name,
		Models: make(map[string]*Dimension),
		Config: DefaultConfig(),
	}
}

func (e *Encoder) Encode(name string, input Unified) []float64 {
	vector := make([]float64, 0)
	if _, ok := e.Models[name]; !ok {
		log.Fatalf("Model %s is not part of encoder %s", name, e.Name)
	}
	switch e.Models[name].InputType {
	case String:
		vector = append(vector, e.Models[name].Model.CalculateString(input.String)...)
	case Floats:
		vector = append(vector, e.Models[name].Model.CalculateFloats(input.Float)...)
	}
	return vector
}

func (e *Encoder) Scan(name string, input *Input, encoder EncoderType) {
	samples := len(input.Values)
	if samples == 0 {
		log.Fatalf("no data samples loaded")
		return
	}
	// if encoder != automatic we execute here
	if encoder != Automatic {
		e.Models[name] = &Dimension{
			InputType: input.Type,
			Type:      encoder,
		}
		e.Scanned = true
		return
	}
	if input.Type == Floats {
		dims := len(input.Values[0].Float)
		if e.Config.FloatReducerThreshold <= dims {
			log.Printf("experimental (not executed): we would apply float reducer here")
		}
		e.Models[name] = &Dimension{
			Type:      FloatExact,
			InputType: Floats,
		}

	} else {
		// input.Type == String
		e.Models[name] = &Dimension{
			InputType: String,
			Type:      evaluateStrings(e.Config, input),
		}
	}
	e.Scanned = true
}

func (e *Encoder) Transform(name string, set *Input) {
	if !e.Scanned {
		log.Printf("no set was scanned before, running scan()")
		e.Scan(name, set, Automatic)
	}
	if _, ok := e.Models[name]; !ok {
		log.Fatalf("no model: %s available to transform", name)
	}
	model := e.Models[name]
	switch model.Type {
	case StringDictionary:
		model.Model = NewDictionaryModel()
		model.Model.Fit(set)
	case StringSplitDictionary:
		model.Model = NewSplitDictionaryModel(e.Config)
		model.Model.Fit(set)
	case StringNGrams:
		model.Model = NewNGramModel(e.Config)
		model.Model.Fit(set)
	case StringTopics:
		log.Fatal("not implemented")
	case FloatReducer:
		model.Model = NewFloatReducerModel(e.Config)
		model.Model.Fit(set)
	case FloatExact:
		model.Model = NewFloatExactModel()
		model.Model.Fit(set)
	}
}

// reporting of what the encoder did
func (e *Encoder) Explain() {
	for k, v := range e.Models {
		fmt.Printf("[%s] => %s, %s, to: %d dimensions", k, v.InputType.String(), v.Type.String(), v.Model.GetDimensions())
	}
}
