// Package persist saves and loads models, networks, data sets, online
// learners, encoders and scalers as versioned JSON documents.
//
// Every document is wrapped in an envelope that records its kind and format
// version, so incompatible files are rejected with a clear error instead of
// silently producing a broken model.
//
// Format version 2 adds models and scalers. Files written with version 1
// stay readable; new files are always written with the current version.
package persist

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/model"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

// Version is the current format version.
const Version = 2

const (
	kindNetwork = "network"
	kindSet     = "set"
	kindOnline  = "online"
	kindEncoder = "encoder"
	kindModel   = "model"
	kindScaler  = "scaler"
)

// minVersion is the oldest format version that can be read per kind.
var minVersion = map[string]int{
	kindNetwork: 1,
	kindSet:     1,
	kindOnline:  1,
	kindEncoder: 1,
	kindModel:   2,
	kindScaler:  2,
}

// ErrFormat is returned for documents of the wrong kind or an unsupported version.
var ErrFormat = errors.New("persist: unsupported format")

// ErrTooLarge is returned for documents larger than MaxDocumentSize.
var ErrTooLarge = errors.New("persist: document too large")

// MaxDocumentSize is the largest document the Read and Load functions
// accept, in bytes. It protects services that load files from untrusted
// sources from running out of memory; lower it on small devices.
var MaxDocumentSize int64 = 1 << 30

// limitedReader fails with ErrTooLarge after MaxDocumentSize bytes.
type limitedReader struct {
	r    io.Reader
	left int64
}

func (l *limitedReader) Read(p []byte) (int, error) {
	if l.left <= 0 {
		return 0, ErrTooLarge
	}
	if int64(len(p)) > l.left {
		p = p[:l.left]
	}
	n, err := l.r.Read(p)
	l.left -= int64(n)
	return n, err
}

type envelope struct {
	Kind    string          `json:"kind"`
	Version int             `json:"version"`
	Data    json.RawMessage `json:"data"`
}

func write(w io.Writer, kind string, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return json.NewEncoder(w).Encode(envelope{Kind: kind, Version: Version, Data: data})
}

func read(r io.Reader, kind string, v any) error {
	var env envelope
	if err := json.NewDecoder(&limitedReader{r: r, left: MaxDocumentSize}).Decode(&env); err != nil {
		return err
	}
	if env.Kind != kind {
		return fmt.Errorf("%w: got kind %q, want %q", ErrFormat, env.Kind, kind)
	}
	if lo := minVersion[kind]; env.Version < lo || env.Version > Version {
		return fmt.Errorf("%w: %s version %d (supported: %d to %d)", ErrFormat, kind, env.Version, lo, Version)
	}
	return json.Unmarshal(env.Data, v)
}

// save writes atomically by writing to a temporary file and renaming it.
func save(path string, fn func(io.Writer) error) (err error) {
	f, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = f.Close()
			_ = os.Remove(f.Name())
		}
	}()
	if err = f.Chmod(0o644); err != nil {
		return err
	}
	if err = fn(f); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func load[T any](path string, fn func(io.Reader) (T, error)) (T, error) {
	f, err := os.Open(path)
	if err != nil {
		var zero T
		return zero, err
	}
	defer func() { _ = f.Close() }()
	return fn(f)
}

// WriteNetwork writes a network to w.
func WriteNetwork(w io.Writer, n *nn.Network) error { return write(w, kindNetwork, n) }

// ReadNetwork reads and validates a network from r.
func ReadNetwork(r io.Reader) (*nn.Network, error) {
	n := &nn.Network{}
	if err := read(r, kindNetwork, n); err != nil {
		return nil, err
	}
	if err := n.Validate(); err != nil {
		return nil, err
	}
	return n, nil
}

// SaveNetwork writes a network to a file.
func SaveNetwork(path string, n *nn.Network) error {
	return save(path, func(w io.Writer) error { return WriteNetwork(w, n) })
}

// LoadNetwork reads a network from a file.
func LoadNetwork(path string) (*nn.Network, error) { return load(path, ReadNetwork) }

// WriteSet writes a data set to w.
func WriteSet(w io.Writer, s *learn.Set) error { return write(w, kindSet, s) }

// ReadSet reads a data set from r.
func ReadSet(r io.Reader) (*learn.Set, error) {
	s := &learn.Set{}
	if err := read(r, kindSet, s); err != nil {
		return nil, err
	}
	if err := s.Validate(0, 0); err != nil {
		return nil, err
	}
	return s, nil
}

// SaveSet writes a data set to a file.
func SaveSet(path string, s *learn.Set) error {
	return save(path, func(w io.Writer) error { return WriteSet(w, s) })
}

// LoadSet reads a data set from a file.
func LoadSet(path string) (*learn.Set, error) { return load(path, ReadSet) }

// WriteOnline writes an online learner including its data set to w.
func WriteOnline(w io.Writer, o *online.Online) error { return write(w, kindOnline, o) }

// ReadOnline reads an online learner from r. The logger is not persisted.
func ReadOnline(r io.Reader) (*online.Online, error) {
	o := &online.Online{}
	if err := read(r, kindOnline, o); err != nil {
		return nil, err
	}
	if o.Network == nil || o.Data == nil {
		return nil, fmt.Errorf("%w: online learner without network or data", ErrFormat)
	}
	if err := o.Network.Validate(); err != nil {
		return nil, err
	}
	if o.Config.PrequentialWindow == 0 { // written before v0.6
		o.Config.PrequentialWindow = 500
	}
	if err := o.Config.Validate(); err != nil {
		return nil, err
	}
	if err := o.Data.Validate(o.Network.InputSize(), o.Network.OutputSize()); err != nil {
		return nil, err
	}
	return o, nil
}

// SaveOnline writes an online learner to a file.
func SaveOnline(path string, o *online.Online) error {
	return save(path, func(w io.Writer) error { return WriteOnline(w, o) })
}

// LoadOnline reads an online learner from a file.
func LoadOnline(path string) (*online.Online, error) { return load(path, ReadOnline) }

// WriteEncoder writes an encoder to w.
func WriteEncoder(w io.Writer, e *encoders.Encoder) error { return write(w, kindEncoder, e) }

// ReadEncoder reads an encoder from r.
func ReadEncoder(r io.Reader) (*encoders.Encoder, error) {
	e := &encoders.Encoder{}
	if err := read(r, kindEncoder, e); err != nil {
		return nil, err
	}
	return e, nil
}

// SaveEncoder writes an encoder to a file.
func SaveEncoder(path string, e *encoders.Encoder) error {
	return save(path, func(w io.Writer) error { return WriteEncoder(w, e) })
}

// LoadEncoder reads an encoder from a file.
func LoadEncoder(path string) (*encoders.Encoder, error) { return load(path, ReadEncoder) }

// WriteModel writes a model with its encoders, scaler and network to w.
func WriteModel(w io.Writer, m *model.Model) error { return write(w, kindModel, m) }

// ReadModel reads and validates a model from r.
func ReadModel(r io.Reader) (*model.Model, error) {
	m := &model.Model{}
	if err := read(r, kindModel, m); err != nil {
		return nil, err
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return m, nil
}

// SaveModel writes a model to a file.
func SaveModel(path string, m *model.Model) error {
	return save(path, func(w io.Writer) error { return WriteModel(w, m) })
}

// LoadModel reads a model from a file.
func LoadModel(path string) (*model.Model, error) { return load(path, ReadModel) }

// WriteScaler writes a scaler to w.
func WriteScaler(w io.Writer, sc *learn.Scaler) error { return write(w, kindScaler, sc) }

// ReadScaler reads and validates a scaler from r.
func ReadScaler(r io.Reader) (*learn.Scaler, error) {
	sc := &learn.Scaler{}
	if err := read(r, kindScaler, sc); err != nil {
		return nil, err
	}
	if err := sc.Validate(); err != nil {
		return nil, err
	}
	return sc, nil
}

// SaveScaler writes a scaler to a file.
func SaveScaler(path string, sc *learn.Scaler) error {
	return save(path, func(w io.Writer) error { return WriteScaler(w, sc) })
}

// LoadScaler reads a scaler from a file.
func LoadScaler(path string) (*learn.Scaler, error) { return load(path, ReadScaler) }
