package service

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

func classifierNet(t testing.TB, seed uint64) *nn.Network {
	t.Helper()
	net, err := nn.New(nn.Config{
		Inputs: 2, Hidden: []int{8}, Outputs: 2, Labels: []string{"A", "B"},
		HiddenActivation: nn.Tanh, OutputActivation: nn.Softmax,
		Rand: rand.New(rand.NewPCG(seed, seed)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return net
}

func regressorNet(t testing.TB) *nn.Network {
	t.Helper()
	net, err := nn.New(nn.Config{
		Inputs: 2, Hidden: []int{8}, Outputs: 1,
		HiddenActivation: nn.Tanh, OutputActivation: nn.Linear,
		Rand: rand.New(rand.NewPCG(1, 1)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return net
}

// concept labels a point: x > 0 is A, or the opposite if flipped.
func concept(rng *rand.Rand, flipped bool) ([]float64, string) {
	x := []float64{rng.NormFloat64(), rng.NormFloat64()}
	if (x[0] > 0) != flipped {
		return x, "A"
	}
	return x, "B"
}

// accuracy measures the live network on fresh points of a concept.
func accuracy(t testing.TB, s *Service, flipped bool) float64 {
	t.Helper()
	rng := rand.New(rand.NewPCG(99, 99))
	correct := 0
	for range 1000 {
		x, label := concept(rng, flipped)
		l, err := s.Live().Network.Classify(x)
		if err != nil {
			t.Fatal(err)
		}
		if l == label {
			correct++
		}
	}
	return float64(correct) / 1000
}

// delayed runs a stream whose feedback arrives up to maxDelay events late
// and in a random order. flip says from which event on the concept is flipped.
func delayed(t testing.TB, s *Service, rng *rand.Rand, n, maxDelay int, flip func(int) bool) []Update {
	t.Helper()
	type waiting struct {
		id    string
		label string
		due   int
	}
	var queue []waiting
	var updates []Update
	deliver := func(w waiting) {
		u, err := s.Feedback(w.id, w.label)
		if err != nil {
			t.Fatal(err)
		}
		updates = append(updates, u)
	}
	for i := range n {
		x, label := concept(rng, flip(i))
		p, err := s.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		queue = append(queue, waiting{p.ID, label, i + rng.IntN(maxDelay+1)})
		rng.Shuffle(len(queue), func(a, b int) { queue[a], queue[b] = queue[b], queue[a] })
		rest := queue[:0]
		for _, w := range queue {
			if w.due <= i {
				deliver(w)
			} else {
				rest = append(rest, w)
			}
		}
		queue = rest
	}
	for _, w := range queue {
		deliver(w)
	}
	return updates
}

func TestDelayedFeedback(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Seed = 1
	s, err := New(classifierNet(t, 1), nn.Classification, cfg)
	if err != nil {
		t.Fatal(err)
	}
	before := accuracy(t, s, false)
	delayed(t, s, rand.New(rand.NewPCG(2, 2)), 3000, 50, func(int) bool { return false })
	st := s.Stats()
	after := accuracy(t, s, false)
	t.Logf("accuracy %.3f → %.3f, %+v", before, after, st)
	if after < 0.95 {
		t.Fatalf("accuracy %.3f after 3000 delayed feedbacks", after)
	}
	if st.Feedback != 3000 || st.Pending != 0 || st.Updates != 3000 || st.Published != 30 || st.Version != 31 {
		t.Fatalf("stats %+v", st)
	}
	if st.Replay != 3000 || len(s.History()) != 5 {
		t.Fatalf("replay %d, history %d", st.Replay, len(s.History()))
	}
}

func TestCorrection(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PublishEvery = 1
	s, err := New(classifierNet(t, 1), nn.Classification, cfg)
	if err != nil {
		t.Fatal(err)
	}
	p, _ := s.Predict([]float64{1, 0})
	u, err := s.Feedback(p.ID, "B") // wrong
	if err != nil || u.Corrected || !u.Published {
		t.Fatalf("first feedback %+v, %v", u, err)
	}
	u, err = s.Feedback(p.ID, "A") // correction
	if err != nil || !u.Corrected || u.Version != 3 {
		t.Fatalf("correction %+v, %v", u, err)
	}
	i := s.replay.index[p.ID]
	if s.replay.entries[i].sample.Label != "A" || s.replay.len() != 1 {
		t.Fatalf("stored %+v", s.replay.entries[i].sample)
	}
	if _, err := s.Feedback(p.ID, "C"); err == nil {
		t.Fatal("unknown label accepted")
	}
	if _, err := s.Feedback("nope", "A"); !errors.Is(err, ErrUnknownEvent) {
		t.Fatalf("unknown event: %v", err)
	}
	st := s.Stats()
	if st.Feedback != 1 || st.Corrections != 1 || st.Updates != 2 {
		t.Fatalf("stats %+v", st)
	}
}

// Corrections repair a stream with many wrong labels.
func TestCorrectionsRepair(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Seed = 3
	s, _ := New(classifierNet(t, 3), nn.Classification, cfg)
	rng := rand.New(rand.NewPCG(4, 4))
	type event struct{ id, label string }
	var wrong []event
	for range 2000 {
		x, label := concept(rng, false)
		p, _ := s.Predict(x)
		given := label
		if rng.Float64() < 0.4 { // 40 % wrong at first
			given = map[string]string{"A": "B", "B": "A"}[label]
			wrong = append(wrong, event{p.ID, label})
		}
		if _, err := s.Feedback(p.ID, given); err != nil {
			t.Fatal(err)
		}
	}
	noisy := accuracy(t, s, false)
	for _, e := range wrong {
		if _, err := s.Feedback(e.id, e.label); err != nil {
			t.Fatal(err)
		}
	}
	// a few more clean updates let the replay of corrected samples settle
	for range 500 {
		x, label := concept(rng, false)
		p, _ := s.Predict(x)
		_, _ = s.Feedback(p.ID, label)
	}
	s.Publish()
	repaired := accuracy(t, s, false)
	t.Logf("with 40 %% wrong labels %.3f, after %d corrections %.3f", noisy, len(wrong), repaired)
	if repaired < 0.95 || s.Stats().Corrections != len(wrong) {
		t.Fatalf("accuracy %.3f after corrections", repaired)
	}
}

func TestPendingTTLAndCapacity(t *testing.T) {
	now := time.Unix(1000, 0)
	cfg := DefaultConfig()
	cfg.PendingTTL = time.Minute
	cfg.PendingSize = 3
	cfg.Now = func() time.Time { return now }
	s, _ := New(classifierNet(t, 1), nn.Classification, cfg)
	var ids []string
	for range 4 {
		p, err := s.Predict([]float64{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, p.ID)
		now = now.Add(time.Second)
	}
	if s.Pending() != 3 {
		t.Fatalf("%d pending", s.Pending())
	}
	if _, err := s.Feedback(ids[0], "A"); !errors.Is(err, ErrUnknownEvent) {
		t.Fatalf("evicted event: %v", err)
	}
	if _, err := s.Feedback(ids[1], "A"); err != nil {
		t.Fatal(err)
	}
	now = now.Add(time.Minute)
	if _, err := s.Feedback(ids[2], "A"); !errors.Is(err, ErrUnknownEvent) {
		t.Fatalf("expired event: %v", err)
	}
	st := s.Stats()
	if st.Pending != 0 || st.Evicted != 1 || st.Expired != 2 {
		t.Fatalf("stats %+v", st)
	}
	// own IDs, duplicates
	if _, err := s.PredictID("req-1", []float64{0, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PredictID("req-1", []float64{0, 0}); !errors.Is(err, ErrDuplicateEvent) {
		t.Fatalf("duplicate: %v", err)
	}
	if _, err := s.PredictID("", []float64{0, 0}); err == nil {
		t.Fatal("empty ID")
	}
	if _, err := s.Feedback("req-1", "B"); err != nil {
		t.Fatal(err)
	}
	// the ID is free again once its feedback arrived
	if _, err := s.PredictID("req-1", []float64{0, 0}); err != nil {
		t.Fatal(err)
	}
	// many events: the queue is compacted
	for range 5000 {
		p, _ := s.Predict([]float64{0, 0})
		_, _ = s.Feedback(p.ID, "A")
	}
	if len(s.queue) > 3000 {
		t.Fatalf("queue holds %d entries", len(s.queue))
	}
}

// Experience replay keeps what a region of the input space taught while
// the feedback comes only from another region.
func TestReplayPreventsForgetting(t *testing.T) {
	// region L (x < 0): A if y > 0; region R (x > 0): A if y < 0
	point := func(rng *rand.Rand, right bool) ([]float64, string) {
		x := []float64{-1 - rng.Float64(), rng.NormFloat64()}
		if right {
			x[0] = -x[0]
		}
		if (x[1] > 0) != right {
			return x, "A"
		}
		return x, "B"
	}
	run := func(batch int) float64 {
		cfg := DefaultConfig()
		cfg.ReplayBatch = batch
		cfg.Drift = online.NoDrift
		cfg.Seed = 5
		cfg.LearningRate = 0.2
		s, _ := New(classifierNet(t, 5), nn.Classification, cfg)
		rng := rand.New(rand.NewPCG(6, 6))
		for i := range 6000 {
			x, label := point(rng, i >= 3000 || i%2 == 0) // both regions, then only R
			p, _ := s.Predict(x)
			if _, err := s.Feedback(p.ID, label); err != nil {
				t.Fatal(err)
			}
		}
		correct := 0
		for range 1000 {
			x, label := point(rng, false)
			if l, _ := s.Live().Network.Classify(x); l == label {
				correct++
			}
		}
		return float64(correct) / 1000
	}
	without, with := run(-1), run(15)
	t.Logf("accuracy on the old region: without replay %.3f, with replay %.3f", without, with)
	if with < 0.9 || with < without+0.1 {
		t.Fatalf("replay %.3f vs none %.3f", with, without)
	}
}

func TestDriftBoostAndForget(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Seed = 7
	var events []DriftEvent
	cfg.OnDrift = func(e DriftEvent) { events = append(events, e) }
	s, _ := New(classifierNet(t, 7), nn.Classification, cfg)
	updates := delayed(t, s, rand.New(rand.NewPCG(8, 8)), 5000, 20, func(i int) bool { return i >= 2500 })
	t.Logf("drift events %+v", events)
	if len(events) == 0 || events[0].Feedback < 2500 || events[0].Feedback > 2700 {
		t.Fatalf("drift events %+v", events)
	}
	if events[0].LearningRate != 3*cfg.LearningRate || events[0].Forgotten == 0 {
		t.Fatalf("event %+v", events[0])
	}
	// the learning rate fades back to the base
	var boosted, base int
	for _, u := range updates {
		if u.LearningRate > cfg.LearningRate {
			boosted++
		} else if u.LearningRate == cfg.LearningRate {
			base++
		}
	}
	if boosted < 150 || base < 4000 {
		t.Fatalf("%d boosted, %d base updates", boosted, base)
	}
	if acc := accuracy(t, s, true); acc < 0.95 {
		t.Fatalf("accuracy %.3f on the new concept", acc)
	}
}

// The boost makes the service recover faster from a drift.
func TestDriftBoostRecovers(t *testing.T) {
	recovery := func(boost float64) int {
		cfg := DefaultConfig()
		cfg.Seed = 9
		cfg.DriftBoost = boost
		cfg.PublishEvery = 10
		s, _ := New(classifierNet(t, 9), nn.Classification, cfg)
		rng := rand.New(rand.NewPCG(10, 10))
		for i := range 6000 {
			x, label := concept(rng, i >= 2000)
			p, _ := s.Predict(x)
			if _, err := s.Feedback(p.ID, label); err != nil {
				t.Fatal(err)
			}
			if i > 2000 && i%50 == 0 && accuracy(t, s, true) >= 0.9 {
				return i - 2000
			}
		}
		return 4000
	}
	plain, boosted := recovery(1), recovery(3)
	t.Logf("feedbacks to 90 %% after the drift: plain %d, boosted %d", plain, boosted)
	if boosted >= plain {
		t.Fatalf("boost does not help: %d vs %d", boosted, plain)
	}
}

func TestOutlierFilter(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Seed = 11
	cfg.PublishEvery = 1
	s, err := New(regressorNet(t), nn.Regression, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(12, 12))
	f := func(x []float64) float64 { return 2*x[0] - x[1] + 0.1*rng.NormFloat64() }
	feed := func(n int, shift float64) {
		for range n {
			x := []float64{rng.NormFloat64(), rng.NormFloat64()}
			p, _ := s.Predict(x)
			if _, err := s.FeedbackValue(p.ID, f(x)+shift); err != nil {
				t.Fatal(err)
			}
		}
	}
	feed(1000, 0)
	probe := []float64{0.5, 0.5}
	before, _ := s.Live().Network.Predict(probe)
	// corrupt values: rejected, the event stays pending, the model is untouched
	var ids []string
	for _, bad := range []float64{1e6, -5e4, 400} {
		p, _ := s.Predict(probe)
		if _, err := s.FeedbackValue(p.ID, bad); !errors.Is(err, ErrOutlier) {
			t.Fatalf("value %v: %v", bad, err)
		}
		ids = append(ids, p.ID)
	}
	after, _ := s.Live().Network.Predict(probe)
	if after[0] != before[0] {
		t.Fatalf("prediction moved from %v to %v", before[0], after[0])
	}
	for _, id := range ids {
		if _, err := s.FeedbackValue(id, 0.5); err != nil {
			t.Fatalf("corrected value after rejection: %v", err)
		}
	}
	if st := s.Stats(); st.Rejected != 3 {
		t.Fatalf("rejected %d", st.Rejected)
	}
	// a lasting level shift is accepted after a while
	rejected := s.Stats().Rejected
	for range 2000 {
		x := []float64{rng.NormFloat64(), rng.NormFloat64()}
		p, _ := s.Predict(x)
		_, _ = s.FeedbackValue(p.ID, f(x)+30)
	}
	st := s.Stats()
	t.Logf("level shift of 30: %d rejected, MAE %.3f", st.Rejected-rejected, st.MAE)
	if st.Rejected-rejected > 500 || st.MAE > 1 {
		t.Fatalf("shift not accepted: %+v", st)
	}

	// ranges and the custom check
	cfg.Values = &Range{-10, 10}
	cfg.Inputs = []Range{{-5, 5}, {-5, 5}}
	cfg.Check = func(s *learn.Sample) error {
		if s.Value == 7 {
			return errors.New("seven is not allowed")
		}
		return nil
	}
	s, _ = New(regressorNet(t), nn.Regression, cfg)
	if _, err := s.Predict([]float64{6, 0}); !errors.Is(err, ErrOutlier) {
		t.Fatalf("input range: %v", err)
	}
	if _, err := s.Predict([]float64{math.NaN(), 0}); !errors.Is(err, learn.ErrNonFinite) {
		t.Fatalf("NaN input: %v", err)
	}
	p, _ := s.Predict([]float64{1, 1})
	for _, v := range []float64{11, 7, math.Inf(1)} {
		if _, err := s.FeedbackValue(p.ID, v); err == nil {
			t.Fatalf("value %v accepted", v)
		}
	}
}

// One extreme sample with the filter off moves the weights by at most
// LearningRate·ClipNorm.
func TestClipping(t *testing.T) {
	for _, clip := range []float64{1, -1} {
		cfg := DefaultConfig()
		cfg.OutlierZ = -1
		cfg.ReplayBatch = -1
		cfg.ClipNorm = clip
		cfg.PublishEvery = 1
		s, _ := New(regressorNet(t), nn.Regression, cfg)
		before := s.Live().Network.Clone()
		p, _ := s.Predict([]float64{1, 1})
		if _, err := s.FeedbackValue(p.ID, 1e6); err != nil {
			t.Fatal(err)
		}
		var d2 float64
		for i, l := range s.Live().Network.Layers {
			for j, w := range l.Weights {
				d2 += (w - before.Layers[i].Weights[j]) * (w - before.Layers[i].Weights[j])
			}
			for j, b := range l.Biases {
				d2 += (b - before.Layers[i].Biases[j]) * (b - before.Layers[i].Biases[j])
			}
		}
		step := math.Sqrt(d2)
		t.Logf("clip %v: step %.4g", clip, step)
		if clip > 0 && step > cfg.LearningRate*clip*(1+1e-9) {
			t.Fatalf("step %v exceeds %v", step, cfg.LearningRate*clip)
		}
		if clip < 0 && step < 1000 {
			t.Fatalf("unclipped step only %v", step)
		}
	}
}

func referenceSet(n int, flipped bool) *learn.Set {
	rng := rand.New(rand.NewPCG(20, 20))
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("A")
	_ = set.AddClass("B")
	for range n {
		x, label := concept(rng, flipped)
		_ = set.Add(learn.NewClassificationSample(x, label))
	}
	return set
}

// Poisoned feedback fails the guardrail; the live model stays good.
func TestGuardrailRollback(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Seed = 13
	cfg.Reference = referenceSet(500, false)
	cfg.SnapshotDir = t.TempDir()
	var rollbacks []Rollback
	cfg.OnRollback = func(r Rollback) { rollbacks = append(rollbacks, r) }
	published := 0
	cfg.OnPublish = func(Version) { published++ }
	s, err := New(classifierNet(t, 13), nn.Classification, cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(14, 14))
	for range 2000 {
		x, label := concept(rng, false)
		p, _ := s.Predict(x)
		_, _ = s.Feedback(p.ID, label)
	}
	good := accuracy(t, s, false)
	version := s.Live().Number
	// an attacker or a broken upstream system sends inverted labels
	for range 1000 {
		x, label := concept(rng, true)
		p, _ := s.Predict(x)
		_, _ = s.Feedback(p.ID, label)
	}
	st := s.Stats()
	t.Logf("good %.3f, after poisoning %.3f, %d rollbacks, %+v", good, accuracy(t, s, false), len(rollbacks), st)
	if good < 0.95 || accuracy(t, s, false) < good-0.03 || len(rollbacks) == 0 || st.Rollbacks != len(rollbacks) {
		t.Fatalf("guardrail failed: good %.3f, now %.3f, %d rollbacks", good, accuracy(t, s, false), len(rollbacks))
	}
	if rollbacks[0].Score >= rollbacks[0].Baseline || rollbacks[0].Discarded != 100 || s.Live().Number < version {
		t.Fatalf("rollback %+v", rollbacks[0])
	}
	if published != st.Published || st.Published < 15 {
		t.Fatalf("%d published (%d callbacks)", st.Published, published)
	}
	files, _ := filepath.Glob(filepath.Join(cfg.SnapshotDir, "model-*.json"))
	if len(files) != cfg.History || !slices.Contains(files, filepath.Join(cfg.SnapshotDir, fmt.Sprintf("model-%06d.json", s.Live().Number))) {
		t.Fatalf("snapshot files %v", files)
	}

	// manual rollback walks back the history
	h := s.History()
	prev, err := s.Rollback()
	if err != nil || prev != h[len(h)-2] || s.Live() != prev {
		t.Fatalf("Rollback: %v", err)
	}
	for range 10 {
		_, _ = s.Rollback()
	}
	if _, err := s.Rollback(); err == nil || len(s.History()) != 1 {
		t.Fatal("rollback past the history")
	}

	// a new reference: the guardrail follows it
	if err := s.SetReference(referenceSet(100, true)); err != nil {
		t.Fatal(err)
	}
	if err := s.SetReference(learn.NewSet(nn.Classification)); err == nil {
		t.Fatal("empty reference accepted")
	}
	for range 1500 {
		x, label := concept(rng, true)
		p, _ := s.Predict(x)
		_, _ = s.Feedback(p.ID, label)
	}
	if acc := accuracy(t, s, true); acc < 0.95 {
		t.Fatalf("accuracy %.3f on the new reference concept", acc)
	}
}

// The rolling holdout set guards without a reference set and follows a drift.
func TestHoldoutGuardrail(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Seed = 15
	cfg.Holdout = 300
	s, err := New(classifierNet(t, 15), nn.Classification, cfg)
	if err != nil {
		t.Fatal(err)
	}
	updates := delayed(t, s, rand.New(rand.NewPCG(16, 16)), 6000, 10, func(i int) bool { return i >= 3000 })
	st := s.Stats()
	held := 0
	for _, u := range updates {
		if u.Holdout {
			held++
		}
	}
	t.Logf("%+v", st)
	if held != 600 || st.Holdout != 300 || st.Updates != 5400 {
		t.Fatalf("%d held out, stats %+v", held, st)
	}
	if acc := accuracy(t, s, true); acc < 0.95 {
		t.Fatalf("accuracy %.3f after the drift", acc)
	}
	// correcting a holdout sample does not train
	var id string
	for k := range s.holdout.index {
		id = k
		break
	}
	u, err := s.Feedback(id, "A")
	if err != nil || !u.Holdout || !u.Corrected || s.Stats().Updates != st.Updates {
		t.Fatalf("holdout correction %+v, %v", u, err)
	}
}

func TestExploration(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Epsilon = 0.2
	cfg.Seed = 17
	s, _ := New(classifierNet(t, 17), nn.Classification, cfg)
	explored := 0
	for range 5000 {
		p, err := s.Predict([]float64{1, 0})
		if err != nil {
			t.Fatal(err)
		}
		best := s.labels[nn.ArgMax(p.Output)]
		if p.Explored {
			explored++
			if p.Label == best {
				t.Fatal("explored label is the best one")
			}
		} else if p.Label != best {
			t.Fatal("label is not the best one")
		}
	}
	if explored < 900 || explored > 1100 {
		t.Fatalf("%d of 5000 explored", explored)
	}
	cfg.Epsilon = 0.1
	if _, err := New(regressorNet(t), nn.Regression, cfg); err == nil {
		t.Fatal("exploration for regression")
	}
}

// Predictions run while feedback trains; every answer comes from one
// consistent published version.
func TestConcurrentPredictions(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PublishEvery = 5
	cfg.History = 1000
	s, _ := New(classifierNet(t, 19), nn.Classification, cfg)
	var stop atomic.Bool
	var wg sync.WaitGroup
	var checked atomic.Int64
	ids := make(chan Prediction, 4096)
	for g := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(g), 1))
			for !stop.Load() {
				x, _ := concept(rng, false)
				p, err := s.Predict(x)
				if err != nil {
					t.Error(err)
					return
				}
				select {
				case ids <- p:
				default:
				}
				// the answer matches the version that gave it
				for _, v := range s.History() {
					if v.Number == p.Version {
						out, _ := v.Network.Predict(x)
						if !slices.Equal(out, p.Output) {
							t.Error("inconsistent prediction")
						}
						checked.Add(1)
					}
				}
			}
		}()
	}
	rng := rand.New(rand.NewPCG(21, 21))
	n := 3000
	if testing.Short() {
		n = 500
	}
	for range n {
		x, label := concept(rng, false)
		p, _ := s.Predict(x)
		if _, err := s.Feedback(p.ID, label); err != nil {
			t.Fatal(err)
		}
		select {
		case q := <-ids:
			_, _ = s.Feedback(q.ID, "A")
		default:
		}
	}
	stop.Store(true)
	wg.Wait()
	if checked.Load() == 0 || s.Live().Number < 50 {
		t.Fatalf("%d checked, version %d", checked.Load(), s.Live().Number)
	}
}

func TestNonFiniteWeightsRollBack(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PublishEvery = 1
	cfg.OutlierZ = -1
	s, _ := New(regressorNet(t), nn.Regression, cfg)
	s.shadow.Layers[0].Weights[0] = math.Inf(1) // e.g. memory corruption
	p, _ := s.Predict([]float64{1, 1})
	u, err := s.FeedbackValue(p.ID, 1)
	if err != nil || !u.RolledBack || u.Published || s.Live().Number != 1 {
		t.Fatalf("update %+v, %v", u, err)
	}
	if s.shadow.Validate() != nil {
		t.Fatal("shadow not reset")
	}
}

func TestNewErrors(t *testing.T) {
	net := classifierNet(t, 1)
	bad := func(mutate func(*Config)) Config {
		c := DefaultConfig()
		mutate(&c)
		return c
	}
	for name, c := range map[string]struct {
		net  *nn.Network
		task nn.Task
		cfg  Config
	}{
		"nil":          {nil, nn.Classification, Config{}},
		"task":         {net, nn.Task(9), Config{}},
		"labels":       {regressorNet(t), nn.Classification, Config{}},
		"outputs":      {net, nn.Regression, Config{}},
		"lr":           {net, nn.Classification, bad(func(c *Config) { c.LearningRate = -1 })},
		"drift":        {net, nn.Classification, bad(func(c *Config) { c.Drift = 7 })},
		"boost":        {net, nn.Classification, bad(func(c *Config) { c.DriftBoost = 0.5 })},
		"inputs":       {net, nn.Classification, bad(func(c *Config) { c.Inputs = []Range{{0, 1}} })},
		"input range":  {net, nn.Classification, bad(func(c *Config) { c.Inputs = []Range{{0, 1}, {2, 1}} })},
		"values":       {net, nn.Classification, bad(func(c *Config) { c.Values = &Range{0, 1} })},
		"epsilon":      {net, nn.Classification, bad(func(c *Config) { c.Epsilon = 1 })},
		"holdout":      {net, nn.Classification, bad(func(c *Config) { c.Holdout = 10; c.MinHoldout = 20 })},
		"history":      {net, nn.Classification, bad(func(c *Config) { c.History = -1 })},
		"reference":    {net, nn.Classification, bad(func(c *Config) { c.Reference = learn.NewSet(nn.Regression) })},
		"empty ref":    {net, nn.Classification, bad(func(c *Config) { c.Reference = learn.NewSet(nn.Classification) })},
		"pending":      {net, nn.Classification, bad(func(c *Config) { c.PendingSize = -1 })},
		"snapshot dir": {net, nn.Classification, bad(func(c *Config) { c.SnapshotDir = "/nonexistent/dir" })},
	} {
		if _, err := New(c.net, c.task, c.cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	s, _ := New(net, nn.Classification, Config{})
	if _, err := s.FeedbackValue("x", 1); err == nil {
		t.Error("FeedbackValue on classification")
	}
	if _, err := s.Predict([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Error("Predict dimension")
	}
	r, _ := New(regressorNet(t), nn.Regression, Config{Metric: evaluation.MAE})
	if _, err := r.Feedback("x", "A"); err == nil {
		t.Error("Feedback on regression")
	}
	if r.Live().Score == r.Live().Score { // NaN without a reference
		t.Error("score without a reference")
	}
}

// With scalers, the service takes raw features and values and answers in
// raw units.
func TestScalers(t *testing.T) {
	rng := rand.New(rand.NewPCG(30, 30))
	// minutes = 4·km + 20·traffic + 10, with km in [0, 20]
	event := func() ([]float64, float64) {
		x := []float64{20 * rng.Float64(), rng.Float64()}
		return x, 4*x[0] + 20*x[1] + 10 + rng.NormFloat64()
	}
	ref := learn.NewSet(nn.Regression)
	xs, ys := [][]float64{}, [][]float64{}
	for range 300 {
		x, y := event()
		_ = ref.Add(learn.NewRegressionSample(x, y))
		xs, ys = append(xs, x), append(ys, []float64{y})
	}
	cfg := DefaultConfig()
	cfg.Seed = 31
	cfg.Reference = ref
	cfg.Scaler, _ = learn.FitScalerVectors(xs, learn.Standardize)
	cfg.TargetScaler, _ = learn.FitScalerVectors(ys, learn.Standardize)
	cfg.Values = &Range{0, 300}
	cfg.Inputs = []Range{{0, 100}, {0, 1}}
	s, err := New(regressorNet(t), nn.Regression, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for range 4000 {
		x, y := event()
		p, err := s.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.FeedbackValue(p.ID, y); err != nil {
			t.Fatal(err)
		}
	}
	p, _ := s.Predict([]float64{10, 0.5})
	st := s.Stats()
	t.Logf("prediction %.2f (true 60), MAE %.2f minutes, %+v", p.Value, st.MAE, st)
	if math.Abs(p.Value-60) > 2 || st.MAE > 2 || st.Published < 10 {
		t.Fatalf("prediction %.2f, stats %+v", p.Value, st)
	}
	// a unit error (seconds instead of minutes) is rejected in raw units
	if _, err := s.FeedbackValue(p.ID, 60*60); !errors.Is(err, ErrOutlier) {
		t.Fatalf("unit error: %v", err)
	}
	if _, err := s.FeedbackValue(p.ID, 290); !errors.Is(err, ErrOutlier) {
		t.Fatalf("z-score in raw units: %v", err)
	}
	if _, err := s.Predict([]float64{150, 0.5}); !errors.Is(err, ErrOutlier) {
		t.Fatalf("raw input range: %v", err)
	}
	bad := cfg
	bad.Scaler = &learn.Scaler{Offset: []float64{0}, Factor: []float64{1}}
	if _, err := New(regressorNet(t), nn.Regression, bad); err == nil {
		t.Fatal("scaler size")
	}
	bad = cfg
	bad.TargetScaler = &learn.Scaler{Offset: []float64{0}, Factor: []float64{0}}
	if _, err := New(regressorNet(t), nn.Regression, bad); err == nil {
		t.Fatal("invalid target scaler")
	}
	if _, err := New(classifierNet(t, 1), nn.Classification, Config{TargetScaler: cfg.TargetScaler}); err == nil {
		t.Fatal("target scaler for classification")
	}
	if _, err := New(regressorNet(t), nn.Regression, Config{Scaler: &learn.Scaler{}}); err == nil {
		t.Fatal("empty scaler")
	}
}
