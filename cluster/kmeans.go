// Package cluster groups unlabeled data. KMeans finds k groups whose
// members are close to the group's center.
//
//	m, labels, err := cluster.Fit(points, cluster.Config{K: 4})
//	group, err := m.Predict(newPoint)
//
// k-means measures plain Euclidean distances, so standardize features on
// different scales first (learn.FitScaler).
package cluster

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"

	"github.com/breskos/gopher-learn/nn"
)

// KMeans is a fitted k-means model.
type KMeans struct {
	// Centroids holds the center of every cluster.
	Centroids [][]float64 `json:"centroids"`
	// Inertia is the sum of squared distances of the training points to
	// their centroid; lower means tighter clusters.
	Inertia float64 `json:"inertia"`
	// Iterations and Converged describe the best run.
	Iterations int  `json:"iterations"`
	Converged  bool `json:"converged"`
}

// Config configures Fit.
type Config struct {
	// K is the number of clusters. Required.
	K int
	// Restarts is the number of runs from different starting centroids;
	// the run with the lowest inertia wins. Default 10.
	Restarts int
	// MaxIter bounds the iterations per run. Default 300.
	MaxIter int
	// Rand drives the choice of starting centroids. Nil uses a randomly
	// seeded source.
	Rand *rand.Rand
}

// Fit clusters the points with k-means: k-means++ starting centroids, then
// Lloyd iterations until no point changes its cluster. It returns the
// model and the cluster of every point.
func Fit(points [][]float64, cfg Config) (*KMeans, []int, error) {
	if cfg.K < 1 {
		return nil, nil, fmt.Errorf("cluster: K must be at least 1, got %d", cfg.K)
	}
	if len(points) < cfg.K {
		return nil, nil, fmt.Errorf("cluster: %d points for %d clusters", len(points), cfg.K)
	}
	d := len(points[0])
	if d == 0 {
		return nil, nil, errors.New("cluster: points have no features")
	}
	for i, p := range points {
		if len(p) != d {
			return nil, nil, fmt.Errorf("%w: point %d has %d values, want %d", nn.ErrDimension, i, len(p), d)
		}
		for _, v := range p {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, nil, fmt.Errorf("cluster: point %d has a non-finite value", i)
			}
		}
	}
	if cfg.Restarts <= 0 {
		cfg.Restarts = 10
	}
	if cfg.MaxIter <= 0 {
		cfg.MaxIter = 300
	}
	rng := cfg.Rand
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	var best *KMeans
	var bestLabels []int
	for range cfg.Restarts {
		seed := rng.Uint64()
		m, labels := run(points, cfg.K, cfg.MaxIter, rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)))
		if best == nil || m.Inertia < best.Inertia {
			best, bestLabels = m, labels
		}
	}
	return best, bestLabels, nil
}

func sqDist(a, b []float64) float64 {
	var s float64
	for i, v := range a {
		d := v - b[i]
		s += d * d
	}
	return s
}

// nearest returns the index of the closest centroid and its squared distance.
func nearest(centroids [][]float64, x []float64) (int, float64) {
	best, bestD := 0, math.Inf(1)
	for c, ctr := range centroids {
		if d := sqDist(x, ctr); d < bestD {
			best, bestD = c, d
		}
	}
	return best, bestD
}

// seed picks starting centroids with k-means++: every next centroid is a
// point drawn with probability proportional to its squared distance to the
// closest centroid chosen so far.
func seed(points [][]float64, k int, rng *rand.Rand) [][]float64 {
	centroids := [][]float64{slices.Clone(points[rng.IntN(len(points))])}
	dist := make([]float64, len(points))
	for i, p := range points {
		dist[i] = sqDist(p, centroids[0])
	}
	for len(centroids) < k {
		var total float64
		for _, d := range dist {
			total += d
		}
		next := rng.IntN(len(points))
		if total > 0 {
			r := rng.Float64() * total
			for i, d := range dist {
				if r -= d; r <= 0 {
					next = i
					break
				}
			}
		}
		c := slices.Clone(points[next])
		centroids = append(centroids, c)
		for i, p := range points {
			dist[i] = min(dist[i], sqDist(p, c))
		}
	}
	return centroids
}

const chunks = 64

// assign gives every point its nearest centroid, in parallel for large
// inputs, and returns the inertia and whether any label changed. The work
// is split into fixed chunks summed in order, so the result does not
// depend on the machine.
func assign(points, centroids [][]float64, labels []int) (float64, bool) {
	n := len(points)
	parts := min(chunks, n)
	if n*len(centroids)*len(points[0]) < 1<<16 {
		parts = 1
	}
	size := (n + parts - 1) / parts
	inertia := make([]float64, parts)
	changed := make([]bool, parts)
	work := func(c int) {
		for i := c * size; i < min((c+1)*size, n); i++ {
			l, d := nearest(centroids, points[i])
			if l != labels[i] {
				labels[i] = l
				changed[c] = true
			}
			inertia[c] += d
		}
	}
	if parts == 1 {
		work(0)
	} else {
		var wg sync.WaitGroup
		next := make(chan int, parts)
		for c := range parts {
			next <- c
		}
		close(next)
		for range min(runtime.GOMAXPROCS(0), parts) {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for c := range next {
					work(c)
				}
			}()
		}
		wg.Wait()
	}
	var sum float64
	moved := false
	for c := range parts {
		sum += inertia[c]
		moved = moved || changed[c]
	}
	return sum, moved
}

// run performs one k-means run.
func run(points [][]float64, k, maxIter int, rng *rand.Rand) (*KMeans, []int) {
	d := len(points[0])
	centroids := seed(points, k, rng)
	labels := make([]int, len(points))
	for i := range labels {
		labels[i] = -1
	}
	m := &KMeans{}
	counts := make([]int, k)
	for it := range maxIter {
		inertia, changed := assign(points, centroids, labels)
		m.Inertia, m.Iterations = inertia, it+1
		if !changed {
			m.Converged = true
			break
		}
		// move every centroid to the mean of its points
		for c := range centroids {
			clear(centroids[c])
		}
		clear(counts)
		for i, p := range points {
			l := labels[i]
			counts[l]++
			for j, v := range p {
				centroids[l][j] += v
			}
		}
		for c := range centroids {
			if counts[c] == 0 {
				// an empty cluster takes the point farthest from its centroid
				far, farD := 0, -1.0
				for i, p := range points {
					if dd := sqDist(p, centroids[labels[i]]); dd > farD && counts[labels[i]] > 1 {
						far, farD = i, dd
					}
				}
				copy(centroids[c], points[far])
				counts[labels[far]]--
				labels[far] = c
				counts[c] = 1
				continue
			}
			for j := range d {
				centroids[c][j] /= float64(counts[c])
			}
		}
	}
	m.Centroids = centroids
	return m, labels
}

// Validate checks the model, e.g. after decoding it.
func (m *KMeans) Validate() error {
	if len(m.Centroids) == 0 || len(m.Centroids[0]) == 0 {
		return errors.New("cluster: no centroids")
	}
	d := len(m.Centroids[0])
	for c, ctr := range m.Centroids {
		if len(ctr) != d {
			return fmt.Errorf("cluster: centroid %d has %d values, want %d", c, len(ctr), d)
		}
		for _, v := range ctr {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("cluster: centroid %d has a non-finite value", c)
			}
		}
	}
	return nil
}

// Predict returns the cluster of x: the index of the nearest centroid.
func (m *KMeans) Predict(x []float64) (int, error) {
	if len(x) != len(m.Centroids[0]) {
		return 0, fmt.Errorf("%w: input has %d values, model expects %d", nn.ErrDimension, len(x), len(m.Centroids[0]))
	}
	c, _ := nearest(m.Centroids, x)
	return c, nil
}

// Silhouette returns the mean silhouette coefficient of a clustering, from
// -1 (wrong clusters) over 0 (overlapping) to 1 (dense, well separated).
// Compare it across values of K to choose the number of clusters. It
// compares every pair of points, so it takes O(n²) time; use a sample of
// a few thousand points for large data.
func Silhouette(points [][]float64, labels []int) (float64, error) {
	if len(points) != len(labels) || len(points) < 2 {
		return 0, errors.New("cluster: need at least two points with one label each")
	}
	k := 0
	for _, l := range labels {
		if l < 0 {
			return 0, fmt.Errorf("cluster: negative label %d", l)
		}
		k = max(k, l+1)
	}
	sizes := make([]int, k)
	for _, l := range labels {
		sizes[l]++
	}
	var total float64
	sums := make([]float64, k)
	for i, p := range points {
		clear(sums)
		for j, q := range points {
			if i != j {
				sums[labels[j]] += math.Sqrt(sqDist(p, q))
			}
		}
		own := labels[i]
		if sizes[own] == 1 {
			continue // a singleton scores 0
		}
		a := sums[own] / float64(sizes[own]-1)
		b := math.Inf(1)
		for c := range k {
			if c != own && sizes[c] > 0 {
				b = min(b, sums[c]/float64(sizes[c]))
			}
		}
		if math.IsInf(b, 1) {
			continue // only one cluster
		}
		if m := max(a, b); m > 0 {
			total += (b - a) / m
		}
	}
	return total / float64(len(points)), nil
}
