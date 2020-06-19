package persist

import (
	"encoding/json"
	"io/ioutil"

	learn "github.com/breskos/gopher-learn/learn"
)

// SetFromFile reads a data set from a file
func SetFromFile(path string) (*learn.Set, error) {
	b, err := ioutil.ReadFile(path)
	if nil != err {
		return nil, err
	}
	set := &learn.Set{}
	err = json.Unmarshal(b, set)
	if nil != err {
		return nil, err
	}

	return set, nil
}

// SetToFile writes a set to a file
func SetToFile(path string, set *learn.Set) error {
	j, err := json.Marshal(set)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path, j, 0644)
	return err
}
