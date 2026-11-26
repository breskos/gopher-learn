package service

import (
	"errors"
	"fmt"
	"math"

	"github.com/breskos/gopher-learn/drift"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

// Update is the outcome of a feedback.
type Update struct {
	// Corrected reports that the feedback replaced an earlier one.
	Corrected bool
	// Holdout reports that the sample went to the holdout set instead of
	// training (see Config.Holdout).
	Holdout bool
	// Error is the error of the live prediction: 0 or 1 for classification,
	// the absolute error for regression. Zero for corrections.
	Error float64
	// Loss is the training loss before the update; LearningRate the rate used.
	Loss         float64
	LearningRate float64
	// Drift reports that this feedback completed the detection of a drift.
	Drift bool
	// Published and RolledBack report the outcome of a guardrail check
	// triggered by this update.
	Published, RolledBack bool
	// Version is the live version after the update.
	Version int
}

// Feedback reports the true class of an event. Feedback for an event that
// already has one is a correction: the stored sample gets the new label
// and is trained on again.
func (s *Service) Feedback(id, label string) (Update, error) {
	if s.task != nn.Classification {
		return Update{}, errors.New("service: Feedback needs a classification task, use FeedbackValue")
	}
	return s.feedback(id, &learn.Sample{Label: label})
}

// FeedbackValue reports the true value of a regression event; see Feedback.
func (s *Service) FeedbackValue(id string, value float64) (Update, error) {
	if s.task != nn.Regression {
		return Update{}, errors.New("service: FeedbackValue needs a regression task, use Feedback")
	}
	return s.feedback(id, &learn.Sample{Value: value})
}

func (s *Service) feedback(id string, sample *learn.Sample) (Update, error) {
	raw := sample.Value
	sample.Value = s.netValue(raw)
	s.tmu.Lock()
	defer s.tmu.Unlock()
	if i, ok := s.replay.index[id]; ok {
		return s.correct(&s.replay, i, sample, raw, true)
	}
	if i, ok := s.holdout.index[id]; ok {
		return s.correct(&s.holdout, i, sample, raw, false)
	}

	s.pmu.Lock()
	s.expire(s.now())
	p, ok := s.pending[id]
	s.pmu.Unlock()
	if !ok {
		return Update{}, fmt.Errorf("%w: %q", ErrUnknownEvent, id)
	}
	sample.Vector = p.x
	if err := s.set.Prepare(sample); err != nil {
		return Update{}, fmt.Errorf("service: %w", err)
	}
	if err := s.plausible(sample, raw, true); err != nil {
		s.stats.Rejected++
		s.logger().Warn("rejected feedback", "event", id, "error", err)
		return Update{}, err
	}
	s.pmu.Lock()
	if s.pending[id] == p {
		delete(s.pending, id)
	}
	s.pmu.Unlock()

	// The live prediction was made before the label was known: its error
	// is a genuine test-then-train result.
	u := Update{}
	var err error
	if s.task == nn.Classification {
		if u.Error, err = s.prequent.AddClassification(p.out, nn.ArgMax(p.out), sample.Class); err != nil {
			return Update{}, fmt.Errorf("service: %w", err)
		}
	} else {
		u.Error = s.prequent.AddRegression(raw, s.rawValue(p.out[0]))
	}
	s.stats.Feedback++
	// The detector sees the error in network units: with a TargetScaler
	// these are standardized, which the detector's bounds expect.
	derr := u.Error
	if s.task == nn.Regression {
		derr = math.Abs(sample.Value - p.out[0])
	}
	if s.detect(derr) {
		u.Drift = s.onDrift()
	}

	if s.cfg.Holdout > 0 && s.stats.Feedback%s.cfg.HoldoutEvery == 0 {
		s.holdout.add(id, sample)
		u.Holdout = true
		u.Version = s.live.Load().Number
		return u, nil
	}
	s.replay.add(id, sample)
	return s.train(sample, u)
}

// correct replaces the label of a stored sample and trains on it again
// (unless it is a holdout sample).
func (s *Service) correct(r *replay, i int, sample *learn.Sample, raw float64, train bool) (Update, error) {
	sample.Vector = r.entries[i].sample.Vector
	if err := s.set.Prepare(sample); err != nil {
		return Update{}, fmt.Errorf("service: %w", err)
	}
	if err := s.plausible(sample, raw, false); err != nil {
		s.stats.Rejected++
		return Update{}, err
	}
	r.entries[i].sample = sample
	s.stats.Corrections++
	u := Update{Corrected: true, Holdout: !train}
	if !train {
		u.Version = s.live.Load().Number
		return u, nil
	}
	return s.train(sample, u)
}

// plausible applies the outlier filter. Only first feedbacks update the
// robust statistics of the values.
func (s *Service) plausible(sample *learn.Sample, v float64, first bool) error {
	if s.task == nn.Regression {
		if r := s.cfg.Values; r != nil && !r.Contains(v) {
			return fmt.Errorf("%w: value %v outside [%v, %v]", ErrOutlier, v, r.Min, r.Max)
		}
		if s.cfg.OutlierZ > 0 {
			z := s.values.z(v)
			ok := s.values.n < s.cfg.OutlierWarmup || z <= s.cfg.OutlierZ
			if first {
				// Rejected values enter clamped: a single spike barely moves
				// the statistics, a lasting shift is accepted after a while.
				s.values.add(v, s.cfg.OutlierZ, s.values.n >= s.cfg.OutlierWarmup)
			}
			if !ok {
				return fmt.Errorf("%w: value %v has robust z-score %.1f", ErrOutlier, v, z)
			}
		}
	}
	if s.cfg.Check != nil {
		if err := s.cfg.Check(sample); err != nil {
			return fmt.Errorf("%w: %w", ErrOutlier, err)
		}
	}
	return nil
}

// train applies the online update: the sample plus a replay mini-batch,
// with gradient clipping, on the shadow network.
func (s *Service) train(sample *learn.Sample, u Update) (Update, error) {
	// The batch buffers are reused; replay samples are drawn with replacement.
	xs := append(s.xs[:0], sample.Vector)
	ts := append(s.ts[:0], sample.Output)
	if n := s.replay.len(); n > 0 {
		for range s.cfg.ReplayBatch {
			r := s.replay.entries[s.rng.IntN(n)].sample
			xs = append(xs, r.Vector)
			ts = append(ts, r.Output)
		}
	}
	s.xs, s.ts = xs, ts
	u.LearningRate = s.learningRate()
	clip := max(s.cfg.ClipNorm, 0)
	for k := range s.cfg.Steps {
		loss, err := s.shadow.TrainBatchWith(xs, ts, nn.Step{
			Optimizer:    s.opt,
			LearningRate: u.LearningRate,
			ClipNorm:     clip,
			Rand:         s.rng,
		})
		if errors.Is(err, nn.ErrNonFinite) {
			s.stats.Skipped++
			s.logger().Warn("skipped an update with a non-finite gradient")
			break
		}
		if err != nil {
			return Update{}, fmt.Errorf("service: %w", err)
		}
		if k == 0 {
			u.Loss = loss
		}
	}
	s.stats.Updates++
	s.sinceCheck++
	if s.boostLeft > 0 {
		s.boostLeft--
	}
	if s.cfg.PublishEvery > 0 && s.sinceCheck >= s.cfg.PublishEvery {
		u.Published, u.RolledBack = s.check()
	}
	u.Version = s.live.Load().Number
	return u, nil
}

// learningRate is the base rate, boosted after a drift.
func (s *Service) learningRate() float64 {
	lr := s.cfg.LearningRate
	if s.boostLeft > 0 && s.cfg.DriftBoostUpdates > 0 {
		lr *= 1 + (s.cfg.DriftBoost-1)*float64(s.boostLeft)/float64(s.cfg.DriftBoostUpdates)
	}
	return lr
}

// detect feeds an error to the drift detector. Only a rising error counts.
func (s *Service) detect(e float64) bool {
	switch s.cfg.Drift {
	case online.ADWIN:
		if s.adwin == nil {
			s.adwin = drift.NewADWIN(s.cfg.DriftDelta)
		}
		return s.adwin.Add(e) && s.adwin.Change > 0
	case online.PageHinkley:
		if s.ph == nil {
			s.ph = &drift.PageHinkley{Threshold: s.cfg.DriftDelta}
		}
		return s.ph.Add(e)
	}
	return false
}

// onDrift reacts to a detected drift: the replay buffer forgets the samples
// from before the change, the learning rate is boosted and, once per
// cooldown, the drift is reported. It returns whether it was reported.
func (s *Service) onDrift() bool {
	keep := s.cfg.ReplayBatch + 1
	if s.adwin != nil && s.cfg.Drift == online.ADWIN {
		keep = s.adwin.Width
	}
	kept, dropped := s.replay.keepNewest(keep)
	s.values.n = min(s.values.n, s.cfg.OutlierWarmup/2) // re-learn the value range quickly
	s.boostLeft = s.cfg.DriftBoostUpdates
	if s.stats.Drifts > 0 && s.stats.Feedback-s.lastDrift <= s.cfg.DriftCooldown {
		return false
	}
	s.stats.Drifts++
	s.lastDrift = s.stats.Feedback
	ev := DriftEvent{Feedback: s.stats.Feedback, Kept: kept, Forgotten: dropped, LearningRate: s.learningRate()}
	s.logger().Warn("drift detected", "feedback", ev.Feedback, "kept", kept, "forgotten", dropped, "learning_rate", ev.LearningRate)
	if s.cfg.OnDrift != nil {
		s.cfg.OnDrift(ev)
	}
	return true
}

// replay is a ring buffer of labeled samples addressed by event ID.
type replay struct {
	entries []entry
	next    int // oldest entry once the buffer is full
	index   map[string]int
	size    int
}

type entry struct {
	id     string
	sample *learn.Sample
}

func (r *replay) len() int { return len(r.entries) }

func (r *replay) add(id string, s *learn.Sample) {
	if r.size <= 0 {
		return
	}
	if len(r.entries) < r.size {
		r.index[id] = len(r.entries)
		r.entries = append(r.entries, entry{id, s})
		return
	}
	delete(r.index, r.entries[r.next].id)
	r.entries[r.next] = entry{id, s}
	r.index[id] = r.next
	r.next = (r.next + 1) % r.size
}

// keepNewest keeps the n most recent entries and returns the numbers of
// kept and dropped ones.
func (r *replay) keepNewest(n int) (kept, dropped int) {
	n = max(min(n, len(r.entries)), 0)
	dropped = len(r.entries) - n
	if dropped == 0 {
		return n, 0
	}
	ordered := make([]entry, 0, n)
	for k := range len(r.entries) {
		ordered = append(ordered, r.entries[(r.next+k)%len(r.entries)])
	}
	ordered = ordered[dropped:]
	clear(r.index)
	for i, e := range ordered {
		r.index[e.id] = i
	}
	clear(r.entries)
	r.entries = append(r.entries[:0], ordered...)
	r.next = 0
	return n, dropped
}

// set returns the stored samples as a data set.
func (r *replay) set(task nn.Task, labels []string) *learn.Set {
	set := learn.NewSet(task)
	set.Labels = labels
	set.Samples = make([]*learn.Sample, len(r.entries))
	for i, e := range r.entries {
		set.Samples[i] = e.sample
	}
	return set
}

// robust tracks a location and scale of a stream of values that single
// outliers cannot move far: the mean and mean absolute deviation of the
// recent values, each clamped to the outlier bound before it counts.
type robust struct {
	n         int
	mean, dev float64
}

// robustWindow is the effective number of recent values the statistics follow.
const robustWindow = 200

func (r *robust) scale() float64 {
	// 1.2533 turns the mean absolute deviation of a normal distribution
	// into its standard deviation.
	return max(1.2533*r.dev, 1e-9*(1+math.Abs(r.mean)))
}

func (r *robust) z(v float64) float64 {
	if r.n == 0 {
		return 0
	}
	return math.Abs(v-r.mean) / r.scale()
}

func (r *robust) add(v, bound float64, clamp bool) {
	if clamp {
		lim := bound * r.scale()
		v = min(max(v, r.mean-lim), r.mean+lim)
	}
	r.n++
	a := 1 / float64(min(r.n, robustWindow))
	r.mean += a * (v - r.mean)
	if r.n > 1 {
		r.dev += a * (math.Abs(v-r.mean) - r.dev)
	}
}
