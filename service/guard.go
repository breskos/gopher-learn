package service

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

// reference returns the set the guardrail checks against, or nil.
func (s *Service) reference() *learn.Set {
	if s.ref != nil {
		return s.ref
	}
	if s.cfg.Holdout > 0 && s.holdout.len() >= max(s.cfg.MinHoldout, 1) {
		return s.holdout.set(s.task, s.labels)
	}
	return nil
}

// score evaluates a network on a set with the configured metric. A failed
// evaluation scores NaN.
func (s *Service) score(net *nn.Network, set *learn.Set) float64 {
	e, err := evaluation.Evaluate(net, set, s.cfg.Threshold)
	if err != nil {
		return math.NaN()
	}
	return s.cfg.Metric.Value(e)
}

// finite reports whether all parameters of a network are finite.
func finite(net *nn.Network) bool {
	for _, l := range net.Layers {
		for _, p := range [][]float64{l.Weights, l.Biases, l.Gain, l.Shift} {
			for _, v := range p {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return false
				}
			}
		}
	}
	return true
}

// passes reports whether a candidate score is good enough compared with
// the live score.
func (s *Service) passes(candidate, live float64) bool {
	if math.IsNaN(candidate) {
		return false
	}
	if math.IsNaN(live) {
		return true
	}
	slack := max(s.cfg.Tolerance, 0) * math.Abs(live)
	if s.cfg.Metric.Higher {
		return candidate >= live-slack
	}
	return candidate <= live+slack
}

// check runs the guardrail on the shadow network: it either becomes the
// new live version or is reset to the live one. Called with tmu.
func (s *Service) check() (published, rolledBack bool) {
	live := s.live.Load()
	discarded := s.sinceCheck
	s.sinceCheck = 0
	candidate, baseline := math.NaN(), math.NaN()
	ok := finite(s.shadow)
	if ok {
		if ref := s.reference(); ref != nil {
			candidate, baseline = s.score(s.shadow, ref), s.score(live.Network, ref)
			ok = s.passes(candidate, baseline)
		}
	}
	if !ok {
		s.shadow = live.Network.Clone()
		s.opt = s.newOptimizer()
		s.stats.Rollbacks++
		rb := Rollback{Score: candidate, Baseline: baseline, Discarded: discarded, Live: live.Number}
		s.logger().Warn("guardrail failed, shadow network reset", "score", candidate, "baseline", baseline, "discarded", discarded, "live", live.Number)
		if s.cfg.OnRollback != nil {
			s.cfg.OnRollback(rb)
		}
		return false, true
	}
	s.publish(candidate)
	return true, false
}

// publish makes a copy of the shadow network the live version. Called with tmu.
func (s *Service) publish(score float64) {
	v := &Version{
		Number:    s.live.Load().Number + 1,
		Network:   s.shadow.Clone(),
		Score:     score,
		Updates:   s.stats.Updates,
		Published: s.now(),
	}
	s.history = append(s.history, v)
	if len(s.history) > s.cfg.History {
		s.history = append(s.history[:0], s.history[len(s.history)-s.cfg.History:]...)
	}
	s.live.Store(v)
	s.stats.Published++
	s.logger().Info("published", "version", v.Number, "score", score, "updates", v.Updates)
	if err := s.snapshot(v); err != nil {
		s.stats.SnapshotErrors++
		s.logger().Warn("snapshot failed", "version", v.Number, "error", err)
	}
	if s.cfg.OnPublish != nil {
		s.cfg.OnPublish(*v)
	}
}

// snapshot saves a version to SnapshotDir and removes files of versions
// that left the history.
func (s *Service) snapshot(v *Version) error {
	if s.cfg.SnapshotDir == "" {
		return nil
	}
	path := filepath.Join(s.cfg.SnapshotDir, fmt.Sprintf("model-%06d.json", v.Number))
	if err := persist.SaveNetwork(path, v.Network); err != nil {
		return err
	}
	s.lastFiles = append(s.lastFiles, path)
	for len(s.lastFiles) > s.cfg.History {
		if err := os.Remove(s.lastFiles[0]); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		s.lastFiles = s.lastFiles[1:]
	}
	return nil
}

// Publish runs the guardrail on the shadow network now instead of waiting
// for PublishEvery updates. It reports whether a new version went live.
func (s *Service) Publish() bool {
	s.tmu.Lock()
	defer s.tmu.Unlock()
	published, _ := s.check()
	return published
}

// Rollback puts the previous version of the history live again and resets
// the shadow network to it, e.g. after a problem was noticed outside the
// service. It fails if there is no earlier version.
func (s *Service) Rollback() (*Version, error) {
	s.tmu.Lock()
	defer s.tmu.Unlock()
	if len(s.history) < 2 {
		return nil, errors.New("service: no earlier version")
	}
	s.history = s.history[:len(s.history)-1]
	prev := s.history[len(s.history)-1]
	s.live.Store(prev)
	s.shadow = prev.Network.Clone()
	s.opt = s.newOptimizer()
	s.sinceCheck = 0
	s.stats.Rollbacks++
	s.logger().Warn("rolled back", "version", prev.Number)
	return prev, nil
}

// History returns the kept versions, oldest first; the last one is live.
func (s *Service) History() []*Version {
	s.tmu.Lock()
	defer s.tmu.Unlock()
	return append([]*Version(nil), s.history...)
}

// SetReference replaces the reference set of the guardrail, e.g. after a
// concept drift made the old one obsolete. Nil falls back to the holdout
// set or unchecked publishing.
func (s *Service) SetReference(ref *learn.Set) error {
	s.tmu.Lock()
	defer s.tmu.Unlock()
	cfg := s.cfg
	cfg.Reference = ref
	if err := cfg.validate(s.task, s.shadow); err != nil {
		return err
	}
	scaled, err := s.scaleSet(ref)
	if err != nil {
		return err
	}
	s.cfg.Reference, s.ref = ref, scaled
	return nil
}

// Stats describes the state of a service.
type Stats struct {
	// Version is the live version.
	Version int
	// Pending, Replay and Holdout are the numbers of parked predictions,
	// replay samples and holdout samples.
	Pending, Replay, Holdout int
	// Feedback counts accepted first feedbacks, Corrections the accepted
	// corrections, Rejected the feedbacks the outlier filter refused.
	Feedback, Corrections, Rejected int
	// Expired and Evicted count predictions dropped without feedback by
	// PendingTTL and PendingSize.
	Expired, Evicted int
	// Updates counts training updates, Skipped those dropped because of a
	// non-finite gradient.
	Updates, Skipped int
	// Drifts counts reported drifts.
	Drifts int
	// Published and Rollbacks count guardrail outcomes (Rollbacks includes
	// calls of Rollback); SnapshotErrors failed snapshot writes.
	Published, Rollbacks, SnapshotErrors int
	// LearningRate is the current learning rate, boosted after a drift.
	LearningRate float64
	// Accuracy (classification) or MAE and RMSE (regression) of the live
	// predictions over the last 500 first feedbacks.
	Accuracy, MAE, RMSE float64
}

// Stats returns the current state.
func (s *Service) Stats() Stats {
	s.tmu.Lock()
	defer s.tmu.Unlock()
	st := s.stats
	st.Version = s.live.Load().Number
	st.Replay, st.Holdout = s.replay.len(), s.holdout.len()
	st.LearningRate = s.learningRate()
	if s.task == nn.Classification {
		st.Accuracy = s.prequent.Accuracy()
	} else {
		st.MAE, st.RMSE = s.prequent.MAE(), s.prequent.RMSE()
	}
	s.pmu.Lock()
	s.expire(s.now())
	st.Pending, st.Expired, st.Evicted = len(s.pending), s.expired, s.evicted
	s.pmu.Unlock()
	return st
}
