package persist

import (
	"encoding/json"
	"io/ioutil"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	neural "github.com/breskos/gopher-learn/net"
	"github.com/breskos/gopher-learn/online"
)

// OnlineDump is the json representation of the network stucture
type OnlineDump struct {
	NetworkInput   int
	NetworkLayer   []int
	NetworkOutput  int
	Data           *learn.Set
	Network        *NetworkDump
	LastEvaluation *evaluation.Evaluation
	Verbose        bool
	Usage          neural.NetworkType
	AddedPoints    int
	Config         *online.Config
}

// FromOnlineFile loads a OnlineDump from File and creates Online out of it
func OnlineFromFile(path string) (*online.Online, error) {
	dump, err := onlineDumpFromFile(path)
	if nil != err {
		return nil, err
	}
	n := fromOnlineDump(dump)
	return n, nil
}

// OnlineToFile takes a network and creats a NetworkDump out of it and writes it to a file
func OnlineToFile(path string, n *online.Online) error {
	dump := toOnlineDump(n)
	return dumpToOnlineFile(path, dump)
}

// FromOnlineDump creates a Online out of an OnlineDump
func fromOnlineDump(d *OnlineDump) *online.Online {
	return &online.Online{
		NetworkInput:   d.NetworkOutput,
		NetworkLayer:   d.NetworkLayer,
		NetworkOutput:  d.NetworkOutput,
		Data:           d.Data,
		Network:        FromDump(d.Network),
		LastEvaluation: d.LastEvaluation,
		Verbose:        d.Verbose,
		Usage:          d.Usage,
		AddedPoints:    d.AddedPoints,
		Config:         d.Config,
	}
}

// dumpToOnlineFile writes a NetworkDump to file
func dumpToOnlineFile(path string, dump *OnlineDump) error {
	j, err := json.Marshal(dump)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path, j, 0644)
	return err
}

// onlineDumpFromFile loads an OnlineDump from file
func onlineDumpFromFile(path string) (*OnlineDump, error) {
	b, err := ioutil.ReadFile(path)
	if nil != err {
		return nil, err
	}
	dump := &OnlineDump{}
	err = json.Unmarshal(b, dump)
	if nil != err {
		return nil, err
	}

	return dump, nil
}

// toOnlineDump creates a OnlineDump out of an Online
func toOnlineDump(d *online.Online) *OnlineDump {
	return &OnlineDump{
		NetworkInput:   d.NetworkOutput,
		NetworkLayer:   d.NetworkLayer,
		NetworkOutput:  d.NetworkOutput,
		Data:           d.Data,
		Network:        ToDump(d.Network),
		LastEvaluation: d.LastEvaluation,
		Verbose:        d.Verbose,
		Usage:          d.Usage,
		AddedPoints:    d.AddedPoints,
		Config:         d.Config,
	}
}
