package cluster

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

// gaussians returns n points around k well separated centers and the true
// group of every point.
func gaussians(n, k, d int, seed uint64) ([][]float64, []int) {
	rng := rand.New(rand.NewPCG(seed, seed))
	centers := make([][]float64, k)
	for c := range centers {
		centers[c] = make([]float64, d)
		for j := range centers[c] {
			centers[c][j] = float64(c*10) + rng.Float64()
		}
	}
	points := make([][]float64, n)
	truth := make([]int, n)
	for i := range points {
		c := rng.IntN(k)
		truth[i] = c
		points[i] = make([]float64, d)
		for j := range points[i] {
			points[i][j] = centers[c][j] + rng.NormFloat64()
		}
	}
	return points, truth
}

// purity is the share of points whose cluster's majority group is their own.
func purity(labels, truth []int, k int) float64 {
	counts := make([][]int, k)
	for c := range counts {
		counts[c] = make([]int, k)
	}
	for i, l := range labels {
		counts[l][truth[i]]++
	}
	correct := 0
	for _, row := range counts {
		correct += slices.Max(row)
	}
	return float64(correct) / float64(len(labels))
}

func TestKMeans(t *testing.T) {
	for _, n := range []int{300, 20000} { // the second one assigns in parallel
		points, truth := gaussians(n, 4, 5, 1)
		m, labels, err := Fit(points, Config{K: 4, Rand: rand.New(rand.NewPCG(2, 2))})
		if err != nil {
			t.Fatal(err)
		}
		if p := purity(labels, truth, 4); p < 0.999 || !m.Converged {
			t.Fatalf("n=%d: purity %.3f, converged %v", n, p, m.Converged)
		}
		// the inertia is about n·d for unit variance
		if per := m.Inertia / float64(n*5); per < 0.8 || per > 1.2 {
			t.Fatalf("n=%d: inertia per value %.3f", n, per)
		}
		for i := range 20 {
			if c, _ := m.Predict(points[i]); c != labels[i] {
				t.Fatalf("Predict %d, label %d", c, labels[i])
			}
		}
	}
}

func TestKMeansDeterministic(t *testing.T) {
	points, _ := gaussians(5000, 3, 20, 3)
	a, la, _ := Fit(points, Config{K: 3, Restarts: 3, Rand: rand.New(rand.NewPCG(9, 9))})
	b, lb, _ := Fit(points, Config{K: 3, Restarts: 3, Rand: rand.New(rand.NewPCG(9, 9))})
	if a.Inertia != b.Inertia || !slices.Equal(la, lb) {
		t.Fatal("same seed, different result")
	}
}

func TestSilhouetteChoosesK(t *testing.T) {
	points, _ := gaussians(400, 3, 2, 4)
	best, bestK := -1.0, 0
	for k := 2; k <= 5; k++ {
		_, labels, err := Fit(points, Config{K: k, Rand: rand.New(rand.NewPCG(1, 1))})
		if err != nil {
			t.Fatal(err)
		}
		s, err := Silhouette(points, labels)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("k=%d silhouette %.3f", k, s)
		if s > best {
			best, bestK = s, k
		}
	}
	if bestK != 3 || best < 0.7 {
		t.Fatalf("best k %d with %.3f", bestK, best)
	}
}

func TestEmptyClusterAndDuplicates(t *testing.T) {
	// more clusters than distinct points: no panic, every cluster used
	points := [][]float64{{0}, {0}, {0}, {1}, {1}}
	m, labels, err := Fit(points, Config{K: 3, Rand: rand.New(rand.NewPCG(1, 1))})
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Centroids) != 3 || len(labels) != 5 || m.Inertia > 1e-12 {
		t.Fatalf("model %+v, labels %v", m, labels)
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	// one cluster: the centroid is the mean
	m, _, _ = Fit([][]float64{{1, 2}, {3, 4}}, Config{K: 1})
	if !slices.Equal(m.Centroids[0], []float64{2, 3}) {
		t.Fatalf("centroid %v", m.Centroids[0])
	}
	b, _ := json.Marshal(m)
	var back KMeans
	if err := json.Unmarshal(b, &back); err != nil || back.Validate() != nil {
		t.Fatal("JSON round trip")
	}
}

func TestErrors(t *testing.T) {
	ok := [][]float64{{1}, {2}, {3}}
	for name, c := range map[string]struct {
		points [][]float64
		cfg    Config
	}{
		"k":         {ok, Config{}},
		"too few":   {ok, Config{K: 4}},
		"no values": {[][]float64{{}, {}}, Config{K: 1}},
		"ragged":    {[][]float64{{1}, {1, 2}}, Config{K: 1}},
		"nan":       {[][]float64{{1}, {math.NaN()}}, Config{K: 1}},
	} {
		if _, _, err := Fit(c.points, c.cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	m, _, _ := Fit(ok, Config{K: 2})
	if _, err := m.Predict([]float64{1, 2}); !errors.Is(err, nn.ErrDimension) {
		t.Error("Predict dimension")
	}
	for name, km := range map[string]*KMeans{
		"empty":  {},
		"ragged": {Centroids: [][]float64{{1}, {1, 2}}},
		"inf":    {Centroids: [][]float64{{math.Inf(1)}}},
	} {
		if km.Validate() == nil {
			t.Errorf("Validate %s", name)
		}
	}
	if _, err := Silhouette(ok, []int{0}); err == nil {
		t.Error("Silhouette lengths")
	}
	if _, err := Silhouette(ok, []int{0, -1, 0}); err == nil {
		t.Error("Silhouette negative label")
	}
	if s, err := Silhouette(ok, []int{0, 0, 0}); err != nil || s != 0 {
		t.Errorf("one cluster: %v, %v", s, err)
	}
	if s, err := Silhouette(ok, []int{0, 1, 2}); err != nil || s != 0 {
		t.Errorf("singletons: %v, %v", s, err)
	}
	if s, _ := Silhouette([][]float64{{1}, {1}, {5}}, []int{0, 0, 1}); s != 1.0/3*2 {
		t.Errorf("silhouette %v", s)
	}
}
