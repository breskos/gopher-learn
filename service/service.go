// Package service runs a network as a production online learning service:
// it serves live predictions, receives the true labels or values later, and
// keeps learning from them without risking the live model.
//
// The data flow of one event:
//
//	Predict(x)               → event ID and live prediction; x is parked
//	Feedback(id, label)      → the parked x gets its label (or Value for regression)
//	  outlier filter         → implausible feedback is rejected
//	  replay buffer          → the labeled sample is stored under its ID
//	  online update          → the new sample plus a replay mini-batch,
//	                           with gradient clipping, on a shadow copy
//	  drift check            → the error of the live prediction feeds a drift
//	                           detector; a drift raises the learning rate
//	                           for a while and reports a warning
//	every PublishEvery updates:
//	  guardrail              → the shadow copy is scored on a reference set
//	  live swap or rollback  → better or equal: it becomes the live model;
//	                           worse: the shadow copy is reset to the live one
//
// Feedback for an event that was already labeled is a correction: the
// stored sample is replaced and trained on again.
//
// Predictions never wait for training. The live model is an immutable copy
// behind an atomic pointer; training changes only the shadow copy, and a
// new version is published by swapping the pointer. Predict is safe to
// call from any number of goroutines, also while feedback is processed.
package service

import (
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"math"
	mrand "math/rand/v2"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/breskos/gopher-learn/drift"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/internal/train"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

var (
	// ErrUnknownEvent is returned for feedback on an event ID that was never
	// issued, has expired, or was evicted from the store.
	ErrUnknownEvent = errors.New("service: unknown or expired event")
	// ErrDuplicateEvent is returned by PredictID for an ID that is still pending.
	ErrDuplicateEvent = errors.New("service: duplicate event ID")
	// ErrOutlier is returned for feedback rejected by the plausibility checks.
	// The event stays pending, so a corrected feedback can still arrive.
	ErrOutlier = errors.New("service: implausible feedback")
)

// Range is a closed interval of valid values.
type Range struct {
	Min, Max float64
}

// Contains reports whether v lies in the range.
func (r Range) Contains(v float64) bool { return v >= r.Min && v <= r.Max }

// Config configures a service. The zero value of every field selects its
// default; see DefaultConfig.
type Config struct {
	// PendingTTL is how long a prediction waits for its feedback. Default 24 h.
	PendingTTL time.Duration
	// PendingSize bounds the number of parked predictions; when it is
	// reached, the oldest one is dropped. Default 100,000.
	PendingSize int

	// ReplaySize is the number of labeled samples kept for experience
	// replay (a ring buffer addressed by event ID). Default 5000.
	ReplaySize int
	// ReplayBatch is the number of stored samples trained together with each
	// new or corrected sample. Default 15; negative turns replay off.
	ReplayBatch int

	// LearningRate is the base learning rate of the online updates.
	// Default 0.05.
	LearningRate float64
	// Steps is the number of training steps per feedback. Default 1.
	Steps int
	// Optimizer returns a new optimizer for the shadow network. Nil uses SGD.
	Optimizer func() nn.Optimizer
	// ClipNorm bounds the L2 norm of the mean gradient of each step, so the
	// weights change by at most LearningRate·ClipNorm per SGD step.
	// Default 1; negative turns clipping off.
	ClipNorm float64

	// Drift selects the drift detector fed with the errors of the live
	// predictions. Default online.ADWIN; online.NoDrift turns it off.
	Drift online.Detector
	// DriftDelta is the confidence of ADWIN (default 0.002) or the
	// threshold of Page-Hinkley (default 50).
	DriftDelta float64
	// DriftCooldown is the number of feedbacks after a drift during which
	// further detections are ignored. Default 200.
	DriftCooldown int
	// DriftBoost multiplies the learning rate right after a drift; the
	// boost fades out linearly over DriftBoostUpdates updates. Default 3;
	// 1 keeps the learning rate.
	DriftBoost float64
	// DriftBoostUpdates is the length of the boost. Default 200.
	DriftBoostUpdates int
	// OnDrift is called after a drift was detected (the warning).
	OnDrift func(DriftEvent)

	// Scaler, if set, scales every input before the network sees it, so
	// Predict takes raw features. Use the scaler the network was trained
	// with, e.g. learn.FitScaler on the training data.
	Scaler *learn.Scaler
	// TargetScaler, if set, maps regression values to the network's units:
	// FeedbackValue takes raw values and Prediction.Value is a raw value.
	TargetScaler *learn.Scaler

	// Inputs bounds every raw feature (len = network inputs) or is nil.
	// Predictions with values outside are rejected.
	Inputs []Range
	// Values bounds the raw regression targets, or is nil.
	Values *Range
	// OutlierZ rejects a regression value whose robust z-score against the
	// recent values exceeds it. Default 6; negative turns it off.
	OutlierZ float64
	// OutlierWarmup is the number of values accepted before OutlierZ applies.
	// Default 50.
	OutlierWarmup int
	// Check is called for every feedback before training with the sample
	// as the network sees it (scaled); an error rejects it as an outlier.
	Check func(*learn.Sample) error

	// Reference is a stable labeled set (raw features and values; the
	// scalers are applied) the shadow network has to do well
	// on before it goes live (the guardrail). Without it, the holdout set
	// is used if there is one, and otherwise new versions are published
	// unchecked (only non-finite weights are caught).
	Reference *learn.Set
	// Holdout is the size of a rolling holdout set built from the feedback:
	// every HoldoutEvery-th first feedback is kept aside instead of being
	// trained on. Without a Reference, the guardrail uses it once it holds
	// MinHoldout samples. Because it follows the stream, it also follows
	// a concept drift, which a fixed reference set does not. Default 0 (off).
	Holdout int
	// HoldoutEvery is the share of feedback kept aside. Default 10.
	HoldoutEvery int
	// MinHoldout is the number of holdout samples the guardrail waits for.
	// Default min(Holdout, 50).
	MinHoldout int
	// Metric scores the reference set. Default evaluation.Accuracy for
	// classification and evaluation.RMSE for regression.
	Metric evaluation.Metric
	// Tolerance is the relative worsening of the metric that still passes
	// the guardrail, against the best score published since the reference
	// was set. Default 0.02 (2 %); negative means no worsening at all.
	Tolerance float64
	// Threshold is the maximum error for a regression prediction to count as
	// correct in the evaluations. Default 0.05.
	Threshold float64
	// PublishEvery is the number of updates between guardrail checks.
	// Default 100.
	PublishEvery int
	// History is the number of published versions kept for Rollback.
	// Default 5.
	History int
	// SnapshotDir, if set, receives every published version as
	// model-<version>.json; only the History most recent files are kept.
	SnapshotDir string
	// OnPublish is called after a new version went live, OnRollback after
	// the shadow network was reset because it failed the guardrail.
	OnPublish  func(Version)
	OnRollback func(Rollback)

	// Epsilon is the exploration rate for classification: with this
	// probability a prediction returns a random other class instead of the
	// best one (ε-greedy). Default 0.
	Epsilon float64

	// Seed makes replay sampling and exploration reproducible. Zero uses a
	// random seed.
	Seed uint64
	// Now returns the current time. Nil uses time.Now.
	Now func() time.Time
	// Logger receives warnings and progress. Nil disables logging.
	Logger *slog.Logger
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		PendingTTL:        24 * time.Hour,
		PendingSize:       100_000,
		ReplaySize:        5000,
		ReplayBatch:       15,
		LearningRate:      0.05,
		Steps:             1,
		ClipNorm:          1,
		Drift:             online.ADWIN,
		DriftCooldown:     200,
		DriftBoost:        3,
		DriftBoostUpdates: 200,
		OutlierZ:          6,
		OutlierWarmup:     50,
		Tolerance:         0.02,
		Threshold:         0.05,
		PublishEvery:      100,
		History:           5,
	}
}

// withDefaults fills zero fields with the defaults.
func (c Config) withDefaults(task nn.Task) Config {
	d := DefaultConfig()
	set := func(v *int, def int) {
		if *v == 0 {
			*v = def
		}
	}
	setF := func(v *float64, def float64) {
		if *v == 0 {
			*v = def
		}
	}
	if c.PendingTTL == 0 {
		c.PendingTTL = d.PendingTTL
	}
	set(&c.PendingSize, d.PendingSize)
	set(&c.ReplaySize, d.ReplaySize)
	set(&c.ReplayBatch, d.ReplayBatch)
	setF(&c.LearningRate, d.LearningRate)
	set(&c.Steps, d.Steps)
	setF(&c.ClipNorm, d.ClipNorm)
	set(&c.DriftCooldown, d.DriftCooldown)
	setF(&c.DriftBoost, d.DriftBoost)
	set(&c.DriftBoostUpdates, d.DriftBoostUpdates)
	setF(&c.OutlierZ, d.OutlierZ)
	set(&c.OutlierWarmup, d.OutlierWarmup)
	setF(&c.Tolerance, d.Tolerance)
	setF(&c.Threshold, d.Threshold)
	set(&c.PublishEvery, d.PublishEvery)
	set(&c.History, d.History)
	set(&c.HoldoutEvery, 10)
	if c.MinHoldout == 0 {
		c.MinHoldout = min(c.Holdout, 50)
	}
	if c.Metric.Value == nil {
		c.Metric = evaluation.Accuracy
		if task == nn.Regression {
			c.Metric = evaluation.RMSE
		}
	}
	return c
}

func (c Config) validate(task nn.Task, net *nn.Network) error {
	switch {
	case c.PendingTTL < 0 || c.PendingSize < 0 || c.ReplaySize < 0:
		return errors.New("service: PendingTTL, PendingSize and ReplaySize must not be negative")
	case !(c.LearningRate > 0) || c.Steps < 0:
		return errors.New("service: LearningRate must be positive and Steps not negative")
	case c.Drift < online.NoDrift || c.Drift > online.PageHinkley:
		return fmt.Errorf("service: invalid drift detector %v", c.Drift)
	case c.DriftDelta < 0 || c.DriftCooldown < 0 || c.DriftBoostUpdates < 0:
		return errors.New("service: DriftDelta, DriftCooldown and DriftBoostUpdates must not be negative")
	case !(c.DriftBoost >= 1):
		return errors.New("service: DriftBoost must be at least 1")
	case c.Inputs != nil && len(c.Inputs) != net.InputSize():
		return fmt.Errorf("service: %d input ranges for %d inputs", len(c.Inputs), net.InputSize())
	case c.Values != nil && (task != nn.Regression || !(c.Values.Min <= c.Values.Max)):
		return errors.New("service: Values needs a regression task and Min <= Max")
	case c.Holdout < 0 || c.HoldoutEvery < 2 || c.MinHoldout < 0 || c.MinHoldout > c.Holdout:
		return errors.New("service: need Holdout >= 0, HoldoutEvery >= 2 and 0 <= MinHoldout <= Holdout")
	case c.OutlierWarmup < 0 || c.PublishEvery < 0 || c.History < 1:
		return errors.New("service: OutlierWarmup and PublishEvery must not be negative, History must be positive")
	case !(c.Epsilon >= 0 && c.Epsilon < 1):
		return errors.New("service: Epsilon must be in [0, 1)")
	case c.Epsilon > 0 && task != nn.Classification:
		return errors.New("service: Epsilon needs a classification task")
	}
	if c.Scaler != nil {
		if err := c.Scaler.Validate(); err != nil {
			return fmt.Errorf("service: %w", err)
		}
		if c.Scaler.Size() != net.InputSize() {
			return fmt.Errorf("service: scaler for %d features, network has %d inputs", c.Scaler.Size(), net.InputSize())
		}
	}
	if c.TargetScaler != nil {
		if task != nn.Regression || c.TargetScaler.Size() != 1 {
			return errors.New("service: TargetScaler needs a regression task and one value")
		}
		if err := c.TargetScaler.Validate(); err != nil {
			return fmt.Errorf("service: %w", err)
		}
	}
	for i, r := range c.Inputs {
		if !(r.Min <= r.Max) {
			return fmt.Errorf("service: input range %d has Min > Max", i)
		}
	}
	if c.Reference != nil {
		if c.Reference.Task != task {
			return errors.New("service: reference set has another task")
		}
		if err := c.Reference.Validate(net.InputSize(), net.OutputSize()); err != nil {
			return fmt.Errorf("service: reference set: %w", err)
		}
		if c.Reference.Len() == 0 {
			return errors.New("service: empty reference set")
		}
	}
	return nil
}

// Version is a published network. Its network is never trained and may be
// used for predictions directly, but must not be modified.
type Version struct {
	Number  int
	Network *nn.Network
	// Score is the guardrail score on the reference set (NaN without one).
	Score float64
	// Updates is the number of updates the version includes.
	Updates   int
	Published time.Time
}

// Rollback describes a shadow network that failed the guardrail.
type Rollback struct {
	// Score is the shadow network's score, Baseline the score it had to reach.
	Score, Baseline float64
	// Discarded is the number of updates thrown away.
	Discarded int
	// Live is the version that stays live.
	Live int
}

// DriftEvent describes a detected drift.
type DriftEvent struct {
	// Feedback is the number of first feedbacks received, including the
	// one that completed the detection.
	Feedback int
	// Kept and Forgotten are the numbers of replay samples kept and dropped.
	Kept, Forgotten int
	// LearningRate is the boosted learning rate.
	LearningRate float64
}

// Prediction is the live answer to an input.
type Prediction struct {
	// ID identifies the event for its feedback.
	ID string
	// Output is the output of the live network.
	Output []float64
	// Label is the returned class (classification). With exploration it may
	// differ from the best class; Explored is then true.
	Label    string
	Explored bool
	// Value is the predicted value (regression).
	Value float64
	// Version is the number of the live version that answered.
	Version int
}

// pending is a prediction waiting for its feedback.
type pending struct {
	x       []float64
	out     []float64
	expires time.Time
}

type queued struct {
	id      string
	expires time.Time
}

// Service is a production online learner. Create it with New.
type Service struct {
	cfg    Config
	task   nn.Task
	labels []string
	set    *learn.Set // prepares samples (one-hot targets)
	ref    *learn.Set // Config.Reference, scaled

	live atomic.Pointer[Version]

	// pmu guards the pending store, the ID counter and the exploration
	// random source. Predict holds it only briefly and never waits for
	// training.
	pmu     sync.Mutex
	pending map[string]*pending
	queue   []queued // insertion order, for expiry and eviction
	head    int
	prefix  string
	counter uint64
	erng    *mrand.Rand
	expired int
	evicted int

	// tmu guards everything that training changes. Lock order: tmu, then pmu.
	tmu        sync.Mutex
	shadow     *nn.Network
	opt        nn.Optimizer
	rng        *mrand.Rand
	replay     replay
	holdout    replay
	prequent   *evaluation.Prequential
	adwin      *drift.ADWIN
	ph         *drift.PageHinkley
	values     robust
	history    []*Version
	boostLeft  int
	stats      Stats
	sinceCheck int
	lastDrift  int
	lastFiles  []string
	xs, ts     [][]float64 // batch buffers of train
}

// New starts a service with a trained network. The network is copied; the
// copy becomes version 1. For classification, the network's labels are the
// classes.
func New(net *nn.Network, task nn.Task, cfg Config) (*Service, error) {
	if net == nil {
		return nil, errors.New("service: nil network")
	}
	if err := net.Validate(); err != nil {
		return nil, fmt.Errorf("service: %w", err)
	}
	set := learn.NewSet(task)
	switch task {
	case nn.Classification:
		if len(net.Labels) != net.OutputSize() || len(net.Labels) < 2 {
			return nil, errors.New("service: classification needs a network with one label per output")
		}
		for _, l := range net.Labels {
			if err := set.AddClass(l); err != nil {
				return nil, fmt.Errorf("service: %w", err)
			}
		}
	case nn.Regression:
		if net.OutputSize() != 1 {
			return nil, errors.New("service: regression needs a network with one output")
		}
	default:
		return nil, fmt.Errorf("service: invalid task %v", task)
	}
	cfg = cfg.withDefaults(task)
	var err error
	if err := cfg.validate(task, net); err != nil {
		return nil, err
	}
	s := &Service{
		cfg:     cfg,
		task:    task,
		labels:  set.Labels,
		set:     set,
		pending: map[string]*pending{},
		shadow:  net.Clone(),
		rng:     train.NewRand(cfg.Seed),
		replay:  replay{index: map[string]int{}, size: cfg.ReplaySize},
		holdout: replay{index: map[string]int{}, size: cfg.Holdout},
	}
	if s.ref, err = s.scaleSet(cfg.Reference); err != nil {
		return nil, err
	}
	s.erng = mrand.New(mrand.NewPCG(s.rng.Uint64(), s.rng.Uint64()))
	s.prefix = newPrefix()
	s.opt = s.newOptimizer()
	if s.prequent, err = evaluation.NewPrequential(task, s.labels, 500); err != nil {
		return nil, err
	}
	v := &Version{Number: 1, Network: net.Clone(), Score: math.NaN(), Published: s.now()}
	if ref := s.reference(); ref != nil {
		v.Score = s.score(v.Network, ref)
	}
	s.history = []*Version{v}
	s.live.Store(v)
	s.stats.Version = 1
	if err := s.snapshot(v); err != nil {
		return nil, err
	}
	return s, nil
}

func newPrefix() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		binary.LittleEndian.PutUint64(b[:], uint64(time.Now().UnixNano()))
	}
	return strconv.FormatUint(binary.LittleEndian.Uint64(b[:]), 36) + "-"
}

func (s *Service) newOptimizer() nn.Optimizer {
	if s.cfg.Optimizer != nil {
		return s.cfg.Optimizer()
	}
	return nn.SGD()
}

func (s *Service) now() time.Time {
	if s.cfg.Now != nil {
		return s.cfg.Now()
	}
	return time.Now()
}

func (s *Service) logger() *slog.Logger {
	if s.cfg.Logger == nil {
		return slog.New(slog.DiscardHandler)
	}
	return s.cfg.Logger
}

// Live returns the live version. Its network must not be modified.
func (s *Service) Live() *Version { return s.live.Load() }

// Predict answers with the live network and parks the input under a new
// event ID until its feedback arrives. It never waits for training.
func (s *Service) Predict(x []float64) (Prediction, error) {
	return s.predict("", x)
}

// PredictID is Predict with an ID chosen by the caller, e.g. a request ID.
// It fails with ErrDuplicateEvent if the ID is still pending.
func (s *Service) PredictID(id string, x []float64) (Prediction, error) {
	if id == "" {
		return Prediction{}, errors.New("service: empty event ID")
	}
	return s.predict(id, x)
}

func (s *Service) predict(id string, x []float64) (Prediction, error) {
	if err := s.checkInput(x); err != nil {
		return Prediction{}, err
	}
	xn := s.scaleInput(x)
	v := s.live.Load()
	out, err := v.Network.Predict(xn)
	if err != nil {
		return Prediction{}, fmt.Errorf("service: %w", err)
	}
	p := Prediction{Output: out, Version: v.Number}
	best := 0
	if s.task == nn.Classification {
		best = nn.ArgMax(out)
		p.Label = s.labels[best]
	} else {
		p.Value = s.rawValue(out[0])
	}

	s.pmu.Lock()
	defer s.pmu.Unlock()
	if id == "" {
		s.counter++
		id = s.prefix + strconv.FormatUint(s.counter, 36)
	} else if _, ok := s.pending[id]; ok {
		return Prediction{}, fmt.Errorf("%w: %q", ErrDuplicateEvent, id)
	}
	if s.cfg.Epsilon > 0 && s.erng.Float64() < s.cfg.Epsilon {
		other := s.erng.IntN(len(s.labels) - 1)
		if other >= best {
			other++
		}
		p.Label, p.Explored = s.labels[other], true
	}
	now := s.now()
	s.expire(now)
	for len(s.pending) >= s.cfg.PendingSize {
		s.evictOldest()
	}
	exp := now.Add(s.cfg.PendingTTL)
	s.pending[id] = &pending{x: xn, out: out, expires: exp}
	s.queue = append(s.queue, queued{id, exp})
	if len(s.queue)-s.head > 2*len(s.pending)+1024 {
		s.prune()
	}
	p.ID = id
	return p, nil
}

// scaleInput returns a scaled copy of a raw input.
func (s *Service) scaleInput(x []float64) []float64 {
	out := make([]float64, len(x))
	if sc := s.cfg.Scaler; sc != nil {
		for i, v := range x {
			out[i] = (v - sc.Offset[i]) * sc.Factor[i]
		}
		return out
	}
	copy(out, x)
	return out
}

// netValue maps a raw regression value to network units, rawValue back.
func (s *Service) netValue(v float64) float64 {
	if sc := s.cfg.TargetScaler; sc != nil {
		return (v - sc.Offset[0]) * sc.Factor[0]
	}
	return v
}

func (s *Service) rawValue(v float64) float64 {
	if sc := s.cfg.TargetScaler; sc != nil {
		return v/sc.Factor[0] + sc.Offset[0]
	}
	return v
}

// scaleSet returns a copy of a raw labeled set in network units.
func (s *Service) scaleSet(raw *learn.Set) (*learn.Set, error) {
	if raw == nil {
		return nil, nil
	}
	out := learn.NewSet(s.task)
	out.Labels = s.labels
	out.Samples = make([]*learn.Sample, len(raw.Samples))
	for i, sm := range raw.Samples {
		c := &learn.Sample{Vector: s.scaleInput(sm.Vector), Label: sm.Label, Value: s.netValue(sm.Value)}
		if err := s.set.Prepare(c); err != nil {
			return nil, fmt.Errorf("service: reference sample %d: %w", i, err)
		}
		out.Samples[i] = c
	}
	return out, nil
}

func (s *Service) checkInput(x []float64) error {
	n := s.shadowInputs()
	if len(x) != n {
		return fmt.Errorf("service: %w: input has %d values, network expects %d", nn.ErrDimension, len(x), n)
	}
	for i, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("service: %w: feature %d is %v", learn.ErrNonFinite, i, v)
		}
	}
	for i, r := range s.cfg.Inputs {
		if !r.Contains(x[i]) {
			return fmt.Errorf("%w: feature %d = %v outside [%v, %v]", ErrOutlier, i, x[i], r.Min, r.Max)
		}
	}
	return nil
}

func (s *Service) shadowInputs() int { return s.live.Load().Network.InputSize() }

// expire drops pending predictions whose TTL has passed. Called with pmu.
func (s *Service) expire(now time.Time) {
	for s.head < len(s.queue) {
		q := s.queue[s.head]
		if p, ok := s.pending[q.id]; ok && p.expires.Equal(q.expires) {
			if now.Before(q.expires) {
				break
			}
			delete(s.pending, q.id)
			s.expired++
		}
		s.queue[s.head] = queued{}
		s.head++
	}
	s.compact()
}

// evictOldest drops the oldest pending prediction. Called with pmu.
func (s *Service) evictOldest() {
	for s.head < len(s.queue) {
		q := s.queue[s.head]
		s.queue[s.head] = queued{}
		s.head++
		if p, ok := s.pending[q.id]; ok && p.expires.Equal(q.expires) {
			delete(s.pending, q.id)
			s.evicted++
			break
		}
	}
	s.compact()
}

// compact reuses the queue's memory once its consumed head dominates.
func (s *Service) compact() {
	if s.head > 1024 && s.head > len(s.queue)/2 {
		n := copy(s.queue, s.queue[s.head:])
		clear(s.queue[n:])
		s.queue = s.queue[:n]
		s.head = 0
	}
}

// prune drops queue entries of predictions that got their feedback, so
// the queue stays proportional to the pending store. Called with pmu.
func (s *Service) prune() {
	live := s.queue[:0]
	for _, q := range s.queue[s.head:] {
		if p, ok := s.pending[q.id]; ok && p.expires.Equal(q.expires) {
			live = append(live, q)
		}
	}
	clear(s.queue[len(live):])
	s.queue = live
	s.head = 0
}

// Pending returns the number of predictions waiting for feedback.
func (s *Service) Pending() int {
	s.pmu.Lock()
	defer s.pmu.Unlock()
	s.expire(s.now())
	return len(s.pending)
}
