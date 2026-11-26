// Package pca reduces the number of features with principal component
// analysis: it finds the orthogonal directions along which the data varies
// most and describes every point by its coordinates along the first few.
//
//	p, err := pca.Fit(points, pca.Config{MinExplained: 0.95})
//	z, err := p.Transform(x) // fewer values, same information
//
// Use it to compress correlated features before training, to speed up
// models on small devices, or to plot high-dimensional data in 2D.
package pca

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sort"
	"sync"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// PCA is a fitted principal component analysis.
type PCA struct {
	// Mean is subtracted from every point before the projection.
	Mean []float64 `json:"mean"`
	// Scale divides every centered feature if the features were
	// standardized; nil otherwise.
	Scale []float64 `json:"scale,omitempty"`
	// Components holds the principal directions, one unit vector per row,
	// from the largest variance down.
	Components [][]float64 `json:"components"`
	// Variance holds the variance of the data along every component.
	Variance []float64 `json:"variance"`
	// TotalVariance is the variance of the data over all directions.
	TotalVariance float64 `json:"total_variance"`
	// Whiten scales every coordinate to unit variance.
	Whiten bool `json:"whiten,omitempty"`
}

// Config configures Fit.
type Config struct {
	// Components is the number of components to keep. Zero keeps all, or
	// as many as MinExplained needs.
	Components int
	// MinExplained keeps the fewest components that explain at least this
	// share of the variance, e.g. 0.95. It applies when Components is zero.
	MinExplained float64
	// Standardize scales every feature to unit variance before the
	// analysis, so features on large scales do not dominate.
	Standardize bool
	// Whiten scales the coordinates to unit variance.
	Whiten bool
}

// Fit computes the principal components of the points from the eigen
// decomposition of their covariance matrix (Jacobi method). It takes about
// O(n·d² + d³) time for n points with d features.
func Fit(points [][]float64, cfg Config) (*PCA, error) {
	if len(points) < 2 {
		return nil, errors.New("pca: need at least two points")
	}
	d := len(points[0])
	if d == 0 {
		return nil, errors.New("pca: points have no features")
	}
	for i, p := range points {
		if len(p) != d {
			return nil, fmt.Errorf("%w: point %d has %d values, want %d", nn.ErrDimension, i, len(p), d)
		}
	}
	if cfg.Components < 0 || cfg.Components > d {
		return nil, fmt.Errorf("pca: %d components for %d features", cfg.Components, d)
	}
	if cfg.MinExplained < 0 || cfg.MinExplained > 1 || math.IsNaN(cfg.MinExplained) {
		return nil, fmt.Errorf("pca: MinExplained must be in [0, 1], got %v", cfg.MinExplained)
	}
	sc, err := learn.FitScalerVectors(points, learn.Standardize)
	if err != nil {
		return nil, err
	}
	p := &PCA{Mean: slices.Clone(sc.Offset), Whiten: cfg.Whiten}
	for _, v := range p.Mean {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, errors.New("pca: non-finite values in the points")
		}
	}
	if cfg.Standardize {
		p.Scale = make([]float64, d)
		for j, f := range sc.Factor {
			p.Scale[j] = 1 / f // the standard deviation (1 for constant features)
		}
	}

	cov := covariance(points, p.Mean, p.Scale)
	values, vectors := eigen(cov, d)
	for _, v := range values {
		p.TotalVariance += max(v, 0)
	}
	k := cfg.Components
	if k == 0 {
		k = d
		if cfg.MinExplained > 0 && p.TotalVariance > 0 {
			var sum float64
			for i, v := range values {
				sum += max(v, 0)
				if sum >= cfg.MinExplained*p.TotalVariance*(1-1e-12) {
					k = i + 1
					break
				}
			}
		}
	}
	p.Components = vectors[:k]
	p.Variance = make([]float64, k)
	for i := range k {
		p.Variance[i] = max(values[i], 0)
	}
	return p, nil
}

// covariance returns the d×d covariance matrix of the centered (and
// optionally scaled) points, computed in fixed chunks summed in order.
func covariance(points [][]float64, mean, scale []float64) []float64 {
	n, d := len(points), len(mean)
	parts := 1
	if n*d*d >= 1<<16 {
		parts = min(64, n)
	}
	size := (n + parts - 1) / parts
	parts = (n + size - 1) / size
	sums := make([][]float64, parts)
	work := func(c int) {
		s := make([]float64, d*d)
		z := make([]float64, d)
		for _, p := range points[c*size : min((c+1)*size, n)] {
			for j, v := range p {
				z[j] = v - mean[j]
				if scale != nil {
					z[j] /= scale[j]
				}
			}
			for i, zi := range z {
				row := s[i*d : i*d+i+1]
				for j := range row {
					row[j] += zi * z[j]
				}
			}
		}
		sums[c] = s
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
	cov := make([]float64, d*d)
	for _, s := range sums {
		for i, v := range s {
			cov[i] += v
		}
	}
	for i := range d {
		for j := 0; j <= i; j++ {
			cov[i*d+j] /= float64(n - 1)
			cov[j*d+i] = cov[i*d+j]
		}
	}
	return cov
}

// eigen returns the eigenvalues of the symmetric d×d matrix a in
// descending order and the matching unit eigenvectors as rows, using the
// cyclic Jacobi method. The sign of every vector is chosen so that its
// largest entry is positive.
func eigen(a []float64, d int) ([]float64, [][]float64) {
	v := make([]float64, d*d)
	for i := range d {
		v[i*d+i] = 1
	}
	var norm float64
	for _, x := range a {
		norm += x * x
	}
	for sweep := 0; sweep < 100; sweep++ {
		var off float64
		for p := range d {
			for q := p + 1; q < d; q++ {
				off += a[p*d+q] * a[p*d+q]
			}
		}
		if off <= 1e-30*norm || off == 0 {
			break
		}
		for p := range d {
			for q := p + 1; q < d; q++ {
				apq := a[p*d+q]
				if math.Abs(apq) <= 1e-300 {
					continue
				}
				theta := (a[q*d+q] - a[p*d+p]) / (2 * apq)
				t := 1 / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				if theta < 0 {
					t = -t
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range d { // columns p and q
					akp, akq := a[k*d+p], a[k*d+q]
					a[k*d+p] = c*akp - s*akq
					a[k*d+q] = s*akp + c*akq
				}
				for k := range d { // rows p and q
					apk, aqk := a[p*d+k], a[q*d+k]
					a[p*d+k] = c*apk - s*aqk
					a[q*d+k] = s*apk + c*aqk
				}
				for k := range d {
					vkp, vkq := v[k*d+p], v[k*d+q]
					v[k*d+p] = c*vkp - s*vkq
					v[k*d+q] = s*vkp + c*vkq
				}
			}
		}
	}
	order := make([]int, d)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return a[order[i]*d+order[i]] > a[order[j]*d+order[j]] })
	values := make([]float64, d)
	vectors := make([][]float64, d)
	for r, i := range order {
		values[r] = a[i*d+i]
		vec := make([]float64, d)
		big := 0
		for k := range d {
			vec[k] = v[k*d+i]
			if math.Abs(vec[k]) > math.Abs(vec[big]) {
				big = k
			}
		}
		if vec[big] < 0 {
			for k := range vec {
				vec[k] = -vec[k]
			}
		}
		vectors[r] = vec
	}
	return values, vectors
}

// ExplainedRatio returns the share of the total variance every kept
// component explains.
func (p *PCA) ExplainedRatio() []float64 {
	out := make([]float64, len(p.Variance))
	if p.TotalVariance == 0 {
		return out
	}
	for i, v := range p.Variance {
		out[i] = v / p.TotalVariance
	}
	return out
}

// Validate checks the model, e.g. after decoding it.
func (p *PCA) Validate() error {
	d := len(p.Mean)
	if d == 0 || len(p.Components) == 0 || len(p.Components) > d || len(p.Variance) != len(p.Components) {
		return fmt.Errorf("pca: %d features, %d components, %d variances", d, len(p.Components), len(p.Variance))
	}
	if p.Scale != nil && len(p.Scale) != d {
		return errors.New("pca: scale has the wrong length")
	}
	for _, c := range p.Components {
		if len(c) != d {
			return errors.New("pca: component has the wrong length")
		}
	}
	for _, v := range p.Scale {
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("pca: invalid scale")
		}
	}
	for _, v := range p.Variance {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("pca: invalid variance")
		}
	}
	return nil
}

// TransformInto writes the coordinates of x along the components into dst,
// which needs one entry per component, without allocating.
func (p *PCA) TransformInto(dst, x []float64) error {
	if len(x) != len(p.Mean) {
		return fmt.Errorf("%w: input has %d values, model expects %d", nn.ErrDimension, len(x), len(p.Mean))
	}
	if len(dst) != len(p.Components) {
		return fmt.Errorf("%w: dst has %d values for %d components", nn.ErrDimension, len(dst), len(p.Components))
	}
	for c, comp := range p.Components {
		var s float64
		for j, w := range comp {
			z := x[j] - p.Mean[j]
			if p.Scale != nil {
				z /= p.Scale[j]
			}
			s += w * z
		}
		if p.Whiten && p.Variance[c] > 0 {
			s /= math.Sqrt(p.Variance[c])
		}
		dst[c] = s
	}
	return nil
}

// Transform returns the coordinates of x along the components.
func (p *PCA) Transform(x []float64) ([]float64, error) {
	z := make([]float64, len(p.Components))
	if err := p.TransformInto(z, x); err != nil {
		return nil, err
	}
	return z, nil
}

// Inverse maps coordinates back to the original features. With fewer
// components than features the result is the closest point in the
// subspace of the components: the reconstruction.
func (p *PCA) Inverse(z []float64) ([]float64, error) {
	if len(z) != len(p.Components) {
		return nil, fmt.Errorf("%w: %d coordinates for %d components", nn.ErrDimension, len(z), len(p.Components))
	}
	x := make([]float64, len(p.Mean))
	for c, comp := range p.Components {
		v := z[c]
		if p.Whiten && p.Variance[c] > 0 {
			v *= math.Sqrt(p.Variance[c])
		}
		for j, w := range comp {
			x[j] += v * w
		}
	}
	for j := range x {
		if p.Scale != nil {
			x[j] *= p.Scale[j]
		}
		x[j] += p.Mean[j]
	}
	return x, nil
}

// TransformSet returns a copy of the set with every feature vector replaced
// by its coordinates. Feature names become pc1, pc2, ….
func (p *PCA) TransformSet(set *learn.Set) (*learn.Set, error) {
	out := &learn.Set{Task: set.Task, Labels: slices.Clone(set.Labels), Samples: make([]*learn.Sample, set.Len())}
	for c := range p.Components {
		out.Features = append(out.Features, fmt.Sprintf("pc%d", c+1))
	}
	for i, s := range set.Samples {
		z, err := p.Transform(s.Vector)
		if err != nil {
			return nil, fmt.Errorf("pca: sample %d: %w", i, err)
		}
		c := *s
		c.Vector = z
		c.Output = slices.Clone(s.Output)
		out.Samples[i] = &c
	}
	return out, nil
}
