// Command online-service shows the service package in a realistic setting:
// a model is trained offline on the history of an event log, then runs as
// a live service whose feedback arrives later, is sometimes wrong or
// corrupt, gets corrected, and whose world changes halfway through.
//
// Replay an event log and compare the learning service with the frozen
// offline model:
//
//	go run . -log offers                                       # data generated in memory
//	go run . -log deliveries -data path/to/service/deliveries.csv
//
// Or run it as an HTTP service:
//
//	go run . -log offers -serve :8080
package main

import (
	"container/heap"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"time"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/service"
)

func main() {
	name := flag.String("log", "offers", "event log: offers (classification) or deliveries (regression)")
	data := flag.String("data", "", "path to the event log CSV; empty generates it in memory")
	rows := flag.Int("rows", 40_000, "number of events to generate without -data")
	serve := flag.String("serve", "", "address to serve HTTP on, e.g. :8080, instead of replaying")
	flag.Parse()

	spec, ok := specs[*name]
	if !ok {
		log.Fatalf("unknown event log %q", *name)
	}
	r, err := open(*data, *name, *rows)
	if err != nil {
		log.Fatal(err)
	}
	events, err := readEvents(r, spec)
	if err != nil {
		log.Fatal(err)
	}
	split := len(events) / 5 // the first 20 % are the history
	s, frozen, err := bootstrap(events[:split], spec)
	if err != nil {
		log.Fatal(err)
	}
	if *serve != "" {
		log.Fatal(listen(*serve, s, spec))
	}
	if _, err := replay(os.Stdout, s, frozen, events[split:], spec); err != nil {
		log.Fatal(err)
	}
}

// plausible is the valid range of delivery times in minutes. It cleans the
// history and is the service's Values range.
var plausible = service.Range{Min: 1, Max: 300}

// bootstrap trains the first network offline on the history, as you would
// before going live, and starts the service with it. It also returns a
// frozen copy of the offline model for comparison.
func bootstrap(history []event, spec logSpec) (*service.Service, *frozenModel, error) {
	var xs [][]float64
	var targets []string
	for _, e := range history {
		label := e.feedback
		if e.corr >= 0 {
			label = e.truth // corrected by now
		}
		if label == "" {
			continue // never got feedback
		}
		if spec.task == nn.Regression {
			if v, err := strconv.ParseFloat(label, 64); err != nil || !plausible.Contains(v) {
				continue // clean the history
			}
		}
		xs = append(xs, e.x)
		targets = append(targets, label)
	}
	scaler, err := learn.FitScalerVectors(xs, learn.Standardize)
	if err != nil {
		return nil, nil, err
	}
	set := learn.NewSet(spec.task)
	for _, l := range spec.labels {
		if err := set.AddClass(l); err != nil {
			return nil, nil, err
		}
	}
	var targetScaler *learn.Scaler
	if spec.task == nn.Regression {
		ys := make([][]float64, len(targets))
		for i, t := range targets {
			v, _ := strconv.ParseFloat(t, 64)
			ys[i] = []float64{v}
		}
		if targetScaler, err = learn.FitScalerVectors(ys, learn.Standardize); err != nil {
			return nil, nil, err
		}
	}
	for i, x := range xs {
		sx, _ := scaler.Transform(x)
		sample := learn.NewClassificationSample(sx, targets[i])
		if spec.task == nn.Regression {
			v, _ := strconv.ParseFloat(targets[i], 64)
			y, _ := targetScaler.Transform([]float64{v})
			sample = learn.NewRegressionSample(sx, y[0])
		}
		if err := set.Add(sample); err != nil {
			return nil, nil, err
		}
	}

	rng := rand.New(rand.NewPCG(1, 2))
	cfg := nn.Config{Inputs: len(spec.features), Hidden: []int{16}, HiddenActivation: nn.Tanh, Rand: rng}
	if spec.task == nn.Classification {
		cfg.Outputs, cfg.Labels, cfg.OutputActivation = len(spec.labels), spec.labels, nn.Softmax
	} else {
		cfg.Outputs, cfg.OutputActivation = 1, nn.Linear
	}
	net, err := nn.New(cfg)
	if err != nil {
		return nil, nil, err
	}
	train, val, err := set.Split(0.9, rng)
	if err != nil {
		return nil, nil, err
	}
	if _, err := nn.Fit(context.Background(), net, train.Data(), val.Data(), nn.FitConfig{Epochs: 60, Patience: 8, Rand: rng}); err != nil {
		return nil, nil, err
	}

	sc := service.DefaultConfig()
	sc.Scaler, sc.TargetScaler = scaler, targetScaler
	sc.Holdout = 500 // a rolling holdout set guards the live model and follows drift
	// Corrections arrive up to several days later; the replay buffer must
	// still hold the event then (about 4,000 events per day here).
	sc.ReplaySize = 20_000
	sc.PendingTTL = 3 * 24 * time.Hour
	sc.Seed = 1
	if spec.task == nn.Regression {
		sc.Values = &plausible
	}
	s, err := service.New(net, spec.task, sc)
	if err != nil {
		return nil, nil, err
	}
	return s, &frozenModel{net: net.Clone(), scaler: scaler, target: targetScaler}, nil
}

// frozenModel is the offline model that never learns again.
type frozenModel struct {
	net            *nn.Network
	scaler, target *learn.Scaler
}

func (f *frozenModel) predict(x []float64) (string, float64) {
	sx, _ := f.scaler.Transform(x)
	out, _ := f.net.Predict(sx)
	if f.target != nil {
		v, _ := f.target.Inverse(out)
		return "", v[0]
	}
	return f.net.Labels[nn.ArgMax(out)], 0
}

// action is something that happens at a point in time: a prediction
// request, the feedback for it, or a correction.
type action struct {
	t, kind, event int
}

const (
	predict = iota
	feedback
	correction
)

type timeline []action

func (q timeline) Len() int { return len(q) }
func (q timeline) Less(i, j int) bool {
	if q[i].t != q[j].t {
		return q[i].t < q[j].t
	}
	if q[i].kind != q[j].kind {
		return q[i].kind < q[j].kind
	}
	return q[i].event < q[j].event
}
func (q timeline) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *timeline) Push(x any)   { *q = append(*q, x.(action)) }
func (q *timeline) Pop() any {
	old := *q
	a := old[len(old)-1]
	*q = old[:len(old)-1]
	return a
}

// summary holds the results of a replay after the drift.
type summary struct {
	live, frozen   float64 // accuracy or MAE on the events after the drift
	stats          service.Stats
	unknown        int
	feedbackErrors int
}

// replay plays the live part of an event log against the service in time
// order and prints how the learning service and the frozen model do.
func replay(w io.Writer, s *service.Service, frozen *frozenModel, events []event, spec logSpec) (summary, error) {
	var werr error
	printf := func(format string, args ...any) {
		if werr == nil {
			_, werr = fmt.Fprintf(w, format, args...)
		}
	}
	q := &timeline{}
	for i, e := range events {
		heap.Push(q, action{e.t, predict, i})
	}
	var sum summary
	var segLive, segFrozen, segN, afterLive, afterFrozen, afterN float64
	predicted, segment := 0, max(len(events)/10, 1)
	start := events[0].t
	metric := "accuracy"
	if spec.task == nn.Regression {
		metric = "MAE (min)"
	}
	printf("%8s %7s %7s  %-10s %-10s %7s %6s %9s %8s %11s %7s\n", "events", "hours", "concept",
		"live "+metric[:3], "frozen "+metric[:3], "version", "drifts", "rollbacks", "rejected", "corrections", "pending")
	send := func(e event, value string) error {
		var err error
		if spec.task == nn.Classification {
			_, err = s.Feedback(e.id, value)
		} else {
			v, perr := strconv.ParseFloat(value, 64)
			if perr != nil {
				return perr
			}
			_, err = s.FeedbackValue(e.id, v)
		}
		switch {
		case errors.Is(err, service.ErrOutlier):
			// rejected; the event stays pending for a correction
		case errors.Is(err, service.ErrUnknownEvent):
			sum.unknown++
		case err != nil:
			sum.feedbackErrors++
		}
		return nil
	}
	for q.Len() > 0 {
		a := heap.Pop(q).(action)
		e := events[a.event]
		switch a.kind {
		case predict:
			p, err := s.PredictID(e.id, e.x)
			if err != nil {
				return sum, err
			}
			label, value := frozen.predict(e.x)
			var live, old float64
			if spec.task == nn.Classification {
				live, old = b2f(p.Label == e.truth), b2f(label == e.truth)
			} else {
				truth, _ := strconv.ParseFloat(e.truth, 64)
				live, old = math.Abs(p.Value-truth), math.Abs(value-truth)
			}
			segLive, segFrozen, segN = segLive+live, segFrozen+old, segN+1
			if e.concept == 2 {
				afterLive, afterFrozen, afterN = afterLive+live, afterFrozen+old, afterN+1
			}
			if e.feedback != "" {
				heap.Push(q, action{e.t + e.fbDelay, feedback, a.event})
				if e.corr >= 0 {
					heap.Push(q, action{e.t + e.fbDelay + e.corr, correction, a.event})
				}
			}
			predicted++
			if predicted%segment == 0 || predicted == len(events) {
				st := s.Stats()
				printf("%8d %7.1f %7d  %-10.3f %-10.3f %7d %6d %9d %8d %11d %7d\n", predicted, float64(e.t-start)/3600,
					e.concept, segLive/segN, segFrozen/segN, st.Version, st.Drifts, st.Rollbacks, st.Rejected, st.Corrections, st.Pending)
				segLive, segFrozen, segN = 0, 0, 0
			}
		case feedback:
			if err := send(e, e.feedback); err != nil {
				return sum, err
			}
		case correction:
			if err := send(e, e.truth); err != nil {
				return sum, err
			}
		}
	}
	sum.stats = s.Stats()
	if afterN > 0 {
		sum.live, sum.frozen = afterLive/afterN, afterFrozen/afterN
	}
	st := sum.stats
	printf("\nafter the drift: live %s %.3f, frozen model %.3f\n", metric, sum.live, sum.frozen)
	printf("%d feedbacks, %d corrections, %d rejected, %d updates, %d versions published, %d rollbacks, %d drifts\n",
		st.Feedback, st.Corrections, st.Rejected, st.Updates, st.Published, st.Rollbacks, st.Drifts)
	printf("%d predictions still pending (no feedback), %d feedbacks for unknown or expired events\n", st.Pending, sum.unknown)
	return sum, werr
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
