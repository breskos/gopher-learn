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
// decomposition of their covariance matrix. It takes O(n·d² + d³) time for
// n points with d features.
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
// optionally scaled) points. The rows of the matrix are split across
// goroutines; every entry is summed over the points in the same order, so
// the result does not depend on the machine.
func covariance(points [][]float64, mean, scale []float64) []float64 {
	n, d := len(points), len(mean)
	z := make([]float64, n*d)
	for i, p := range points {
		for j, v := range p {
			c := v - mean[j]
			if scale != nil {
				c /= scale[j]
			}
			z[i*d+j] = c
		}
	}
	cov := make([]float64, d*d)
	rows := func(lo, hi int) {
		for k := range n {
			x := z[k*d : (k+1)*d]
			for i := lo; i < hi; i++ {
				xi := x[i]
				row := cov[i*d : i*d+i+1]
				for j := range row {
					row[j] += xi * x[j]
				}
			}
		}
	}
	workers := 1
	if n*d*d >= 1<<18 {
		workers = min(runtime.GOMAXPROCS(0), d)
	}
	if workers == 1 {
		rows(0, d)
	} else {
		// row i costs i+1 multiply-adds: split the triangle into equal work
		var wg sync.WaitGroup
		lo := 0
		for w := range workers {
			hi := int(math.Round(float64(d) * math.Sqrt(float64(w+1)/float64(workers))))
			if w == workers-1 {
				hi = d
			}
			if hi > lo {
				wg.Add(1)
				go func(lo, hi int) {
					defer wg.Done()
					rows(lo, hi)
				}(lo, hi)
			}
			lo = hi
		}
		wg.Wait()
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
// descending order and the matching unit eigenvectors as rows. It reduces
// a to tridiagonal form with Householder reflections and diagonalizes that
// with the implicit QL method (the symmetric algorithms of EISPACK as used
// in JAMA), which takes O(d³) time. The sign of every vector is chosen so
// that its largest entry is positive.
func eigen(a []float64, d int) ([]float64, [][]float64) {
	V := make([][]float64, d)
	for i := range V {
		V[i] = slices.Clone(a[i*d : (i+1)*d])
	}
	vals, e := make([]float64, d), make([]float64, d)
	tred2(V, vals, e)
	// the QL rotations combine columns of V; on the transpose they combine
	// rows, which are contiguous in memory
	VT := make([][]float64, d)
	for i := range VT {
		VT[i] = make([]float64, d)
		for k := range d {
			VT[i][k] = V[k][i]
		}
	}
	tql2(VT, vals, e)
	order := make([]int, d)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return vals[order[i]] > vals[order[j]] })
	values := make([]float64, d)
	vectors := make([][]float64, d)
	for r, i := range order {
		values[r] = vals[i]
		vec := make([]float64, d)
		big := 0
		for k := range d {
			vec[k] = VT[i][k]
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

// tred2 reduces the symmetric matrix V to tridiagonal form with diagonal d
// and subdiagonal e, accumulating the orthogonal transformation in V.
func tred2(V [][]float64, d, e []float64) {
	n := len(d)
	for j := range n {
		d[j] = V[n-1][j]
	}
	for i := n - 1; i > 0; i-- {
		var scale, h float64
		for k := range i {
			scale += math.Abs(d[k])
		}
		if scale == 0 {
			e[i] = d[i-1]
			for j := range i {
				d[j] = V[i-1][j]
				V[i][j] = 0
				V[j][i] = 0
			}
		} else {
			for k := range i {
				d[k] /= scale
				h += d[k] * d[k]
			}
			f := d[i-1]
			g := math.Sqrt(h)
			if f > 0 {
				g = -g
			}
			e[i] = scale * g
			h -= f * g
			d[i-1] = f - g
			for j := range i {
				e[j] = 0
			}
			for j := range i {
				f = d[j]
				V[j][i] = f
				g = e[j] + V[j][j]*f
				for k := j + 1; k <= i-1; k++ {
					g += V[k][j] * d[k]
					e[k] += V[k][j] * f
				}
				e[j] = g
			}
			f = 0
			for j := range i {
				e[j] /= h
				f += e[j] * d[j]
			}
			hh := f / (h + h)
			for j := range i {
				e[j] -= hh * d[j]
			}
			for j := range i {
				f, g = d[j], e[j]
				for k := j; k <= i-1; k++ {
					V[k][j] -= f*e[k] + g*d[k]
				}
				d[j] = V[i-1][j]
				V[i][j] = 0
			}
		}
		d[i] = h
	}
	for i := 0; i < n-1; i++ {
		V[n-1][i] = V[i][i]
		V[i][i] = 1
		h := d[i+1]
		if h != 0 {
			for k := 0; k <= i; k++ {
				d[k] = V[k][i+1] / h
			}
			for j := 0; j <= i; j++ {
				var g float64
				for k := 0; k <= i; k++ {
					g += V[k][i+1] * V[k][j]
				}
				for k := 0; k <= i; k++ {
					V[k][j] -= g * d[k]
				}
			}
		}
		for k := 0; k <= i; k++ {
			V[k][i+1] = 0
		}
	}
	for j := range n {
		d[j] = V[n-1][j]
		V[n-1][j] = 0
	}
	V[n-1][n-1] = 1
	e[0] = 0
}

// tql2 diagonalizes the tridiagonal matrix (d, e) with the implicit QL
// method and applies the rotations to the rows of VT, the transposed
// transformation of tred2. d receives the eigenvalues, the rows of VT the
// eigenvectors.
func tql2(VT [][]float64, d, e []float64) {
	n := len(d)
	for i := 1; i < n; i++ {
		e[i-1] = e[i]
	}
	e[n-1] = 0
	var f, tst1 float64
	const eps = 0x1p-52
	for l := range n {
		tst1 = max(tst1, math.Abs(d[l])+math.Abs(e[l]))
		m := l
		for m < n-1 && math.Abs(e[m]) > eps*tst1 {
			m++
		}
		if m > l {
			for iter := 0; iter < 100; iter++ {
				g := d[l]
				p := (d[l+1] - g) / (2 * e[l])
				r := math.Hypot(p, 1)
				if p < 0 {
					r = -r
				}
				d[l] = e[l] / (p + r)
				d[l+1] = e[l] * (p + r)
				dl1 := d[l+1]
				h := g - d[l]
				for i := l + 2; i < n; i++ {
					d[i] -= h
				}
				f += h
				p = d[m]
				c, c2, c3 := 1.0, 1.0, 1.0
				el1 := e[l+1]
				var s, s2 float64
				for i := m - 1; i >= l; i-- {
					c3, c2, s2 = c2, c, s
					g = c * e[i]
					h = c * p
					r = math.Hypot(p, e[i])
					e[i+1] = s * r
					s = e[i] / r
					c = p / r
					p = c*d[i] - s*g
					d[i+1] = h + s*(c*g+s*d[i])
					a, b := VT[i], VT[i+1]
					for k := range n {
						h = b[k]
						b[k] = s*a[k] + c*h
						a[k] = c*a[k] - s*h
					}
				}
				p = -s * s2 * c3 * el1 * e[l] / dl1
				e[l] = s * p
				d[l] = c * p
				if math.Abs(e[l]) <= eps*tst1 {
					break
				}
			}
		}
		d[l] += f
		e[l] = 0
	}
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
