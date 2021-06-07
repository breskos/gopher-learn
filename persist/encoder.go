package persist

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/breskos/gopher-learn/encoders"
)

type EncoderDump struct {
	// Name of the encoder
	Name string
	// Dimensions hold the EncoderModel for the dimensions
	Models map[string]Models
	// Config of the Encoder
	Config encoders.EncoderConfig
	// Scanned determines if scan was executed
	Scanned bool
}

type Models struct {
	Inputs    int
	InputType encoders.InputType
	Type      encoders.EncoderType
	Model     []byte
}

// FromFile loads a NetworkDump from File and creates Network out of it
func EncoderFromFile(path string) (*encoders.Encoder, error) {
	dump, err := encoderDumpFromFile(path)
	if nil != err {
		return nil, err
	}
	n := encoderFromDump(dump)
	return n, nil
}

// ToFile takes a network and creats a NetworkDump out of it and writes it to a file
func EncoderToFile(path string, n *encoders.Encoder) error {
	dump := encoderToDump(n)
	return encoderDumpToFile(path, dump)
}

// encoderDrumpFromFile loads an EncoderDump from file
func encoderDumpFromFile(path string) (*EncoderDump, error) {
	b, err := ioutil.ReadFile(path)
	if nil != err {
		return nil, err
	}
	dump := &EncoderDump{}
	err = json.Unmarshal(b, dump)
	if nil != err {
		return nil, err
	}

	return dump, nil
}

// encoderDumpFromFile writes an EncoderDump to file
func encoderDumpToFile(path string, dump *EncoderDump) error {
	j, err := json.Marshal(dump)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path, j, 0644)
	return err
}

// encoderToDump creates an EncoderDump out of a Encoder
func encoderToDump(n *encoders.Encoder) *EncoderDump {
	dimensions := make(map[string]Models, 0)
	for k, v := range n.Models {
		dump, err := v.Model.ToDump()
		if err != nil {
			fmt.Printf("error serializing encoder: %s (%s, %s)", k, v.InputType.String(), v.Type.String())
		}
		dimensions[k] = Models{
			Inputs:    v.Inputs,
			InputType: v.InputType,
			Type:      v.Type,
			Model:     dump,
		}
	}
	return &EncoderDump{
		Name:    n.Name,
		Config:  *n.Config,
		Models:  dimensions,
		Scanned: n.Scanned,
	}
}

// encoderFromDump creates a Encoder out of a Encoder dump
func encoderFromDump(dump *EncoderDump) *encoders.Encoder {
	n := &encoders.Encoder{
		Name:    dump.Name,
		Scanned: dump.Scanned,
		Config:  &dump.Config,
	}
	n.Models = make(map[string]*encoders.Dimension)
	for k, v := range dump.Models {
		n.Models[k] = &encoders.Dimension{
			Inputs:    v.Inputs,
			InputType: v.InputType,
			Type:      v.Type,
		}

		switch v.Type {
		case encoders.StringDictionary:
			n.Models[k].Model = encoders.NewDictionaryModel()
			n.Models[k].Model.FromDump(v.Model)
		case encoders.StringSplitDictionary:
			n.Models[k].Model = encoders.NewSplitDictionaryModel()
			n.Models[k].Model.FromDump(v.Model)
		case encoders.StringNGrams:
			n.Models[k].Model = encoders.NewNGramModel()
			n.Models[k].Model.FromDump(v.Model)
		case encoders.FloatExact:
			n.Models[k].Model = encoders.NewFloatExactModel()
			n.Models[k].Model.FromDump(v.Model)
		case encoders.FloatReducer:
			n.Models[k].Model = encoders.NewFloatReducerModel()
			n.Models[k].Model.FromDump(v.Model)
		}

	}
	return n
}
