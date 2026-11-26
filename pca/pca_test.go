package pca

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// plane returns points that lie close to a 2D plane in 6D: two hidden
// factors with variances 9 and 1, plus small noise.
func plane(n int, seed uint64) [][]float64 {
	rng := rand.New(rand.NewPCG(seed, seed))
	a := []float64{1, 2, 0, 0, 1, 0}
	b := []float64{0, 0, 1, -1, 0, 1}
	points := make([][]float64, n)
	for i := range points {
		u, v := 3*rng.NormFloat64(), rng.NormFloat64()
		p := make([]float64, 6)
		for j := range p {
			p[j] = 10 + u*a[j]/math.Sqrt(6) + v*b[j]/math.Sqrt(3) + 0.01*rng.NormFloat64()
		}
		points[i] = p
	}
	return points
}

func TestPlane(t *testing.T) {
	points := plane(3000, 1)
	p, err := Fit(points, Config{})
	if err != nil {
		t.Fatal(err)
	}
	r := p.ExplainedRatio()
	t.Logf("explained %.4f", r)
	if r[0] < 0.88 || r[0]+r[1] < 0.9999 || len(p.Components) != 6 {
		t.Fatalf("explained %v", r)
	}
	// orthonormal components
	for i, ci := range p.Components {
		for j, cj := range p.Components {
			var dot float64
			for k := range ci {
				dot += ci[k] * cj[k]
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(dot-want) > 1e-9 {
				t.Fatalf("components %d and %d: dot %v", i, j, dot)
			}
		}
	}
	// the first component is the direction of the large factor
	a := []float64{1, 2, 0, 0, 1, 0}
	var cos float64
	for k, v := range p.Components[0] {
		cos += v * a[k] / math.Sqrt(6)
	}
	if math.Abs(cos) < 0.9999 {
		t.Fatalf("first component %v", p.Components[0])
	}
	// all components: exact round trip
	z, _ := p.Transform(points[0])
	back, _ := p.Inverse(z)
	for j := range back {
		if math.Abs(back[j]-points[0][j]) > 1e-9 {
			t.Fatalf("round trip %v vs %v", back, points[0])
		}
	}
}

func TestMinExplainedAndReconstruction(t *testing.T) {
	points := plane(2000, 2)
	p, err := Fit(points, Config{MinExplained: 0.999})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Components) != 2 {
		t.Fatalf("%d components", len(p.Components))
	}
	// two components reconstruct the points up to the noise
	var worst float64
	for _, x := range points[:100] {
		z, _ := p.Transform(x)
		back, _ := p.Inverse(z)
		for j := range x {
			worst = max(worst, math.Abs(back[j]-x[j]))
		}
	}
	if worst > 0.06 {
		t.Fatalf("reconstruction error %v", worst)
	}
	if q, _ := Fit(points, Config{Components: 3}); len(q.Components) != 3 {
		t.Fatal("Components")
	}
}

func TestWhitenStandardize(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	points := make([][]float64, 2000)
	for i := range points {
		x := rng.NormFloat64()
		points[i] = []float64{1000 * x, x + 0.5*rng.NormFloat64(), rng.NormFloat64()}
	}
	// without standardization the large feature dominates
	raw, _ := Fit(points, Config{})
	if raw.ExplainedRatio()[0] < 0.999 {
		t.Fatalf("raw: %v", raw.ExplainedRatio())
	}
	std, err := Fit(points, Config{Standardize: true, Whiten: true})
	if err != nil {
		t.Fatal(err)
	}
	if r := std.ExplainedRatio(); r[0] > 0.7 || math.Abs(std.TotalVariance-3) > 0.01 {
		t.Fatalf("standardized: %v, total %v", r, std.TotalVariance)
	}
	// whitened coordinates have unit variance
	var ss [3]float64
	for _, x := range points {
		z, _ := std.Transform(x)
		for c, v := range z {
			ss[c] += v * v
		}
	}
	for c, v := range ss {
		if math.Abs(v/float64(len(points)-1)-1) > 1e-6 {
			t.Fatalf("component %d variance %v", c, v/float64(len(points)-1))
		}
	}
	z, _ := std.Transform(points[5])
	back, _ := std.Inverse(z)
	if math.Abs(back[0]-points[5][0]) > 1e-6 {
		t.Fatalf("whitened round trip %v vs %v", back, points[5])
	}
	if err := std.Validate(); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(std)
	var p PCA
	if err := json.Unmarshal(b, &p); err != nil || p.Validate() != nil {
		t.Fatal("JSON round trip")
	}
}

func TestTransformSet(t *testing.T) {
	s := learn.NewSet(nn.Classification)
	_ = s.AddClass("a")
	for _, x := range plane(50, 4) {
		_ = s.Add(learn.NewClassificationSample(x, "a"))
	}
	p, _ := Fit(s.Data().X, Config{Components: 2})
	out, err := p.TransformSet(s)
	if err != nil {
		t.Fatal(err)
	}
	if out.InputSize() != 2 || out.Features[1] != "pc2" || out.Samples[0].Label != "a" || s.InputSize() != 6 {
		t.Fatalf("set %+v", out)
	}
	if _, err := p.TransformSet(&learn.Set{Samples: []*learn.Sample{{Vector: []float64{1}}}}); err == nil {
		t.Error("TransformSet dimension")
	}
}

func TestLargeIsFast(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	points := make([][]float64, 2000)
	for i := range points {
		points[i] = make([]float64, 100)
		f := rng.NormFloat64()
		for j := range points[i] {
			points[i][j] = f*float64(j%7) + rng.NormFloat64()
		}
	}
	start := time.Now()
	p, err := Fit(points, Config{Components: 5})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("100 features: %v", time.Since(start))
	// eigen decomposition is correct: A·v = λ·v for the covariance
	cov := covariance(points, p.Mean, nil)
	for c := range 3 {
		v := p.Components[c]
		for i := range 100 {
			var s float64
			for j := range 100 {
				s += cov[i*100+j] * v[j]
			}
			if math.Abs(s-p.Variance[c]*v[i]) > 1e-8*p.Variance[0] {
				t.Fatalf("component %d is no eigenvector", c)
			}
		}
	}
}

func TestErrors(t *testing.T) {
	ok := [][]float64{{1, 2}, {3, 5}, {4, 4}}
	for name, c := range map[string]struct {
		points [][]float64
		cfg    Config
	}{
		"one point":  {ok[:1], Config{}},
		"no values":  {[][]float64{{}, {}}, Config{}},
		"ragged":     {[][]float64{{1}, {1, 2}}, Config{}},
		"components": {ok, Config{Components: 3}},
		"negative":   {ok, Config{Components: -1}},
		"explained":  {ok, Config{MinExplained: 2}},
		"nan":        {[][]float64{{1, math.NaN()}, {2, 3}}, Config{}},
	} {
		if _, err := Fit(c.points, c.cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	p, _ := Fit(ok, Config{})
	if _, err := p.Transform([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Error("Transform dimension")
	}
	if err := p.TransformInto(make([]float64, 5), []float64{1, 2}); !errors.Is(err, nn.ErrDimension) {
		t.Error("TransformInto dst")
	}
	if _, err := p.Inverse([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Error("Inverse dimension")
	}
	for name, mutate := range map[string]func(*PCA){
		"empty":     func(p *PCA) { p.Components = nil },
		"scale":     func(p *PCA) { p.Scale = []float64{1} },
		"bad scale": func(p *PCA) { p.Scale = []float64{1, 0} },
		"component": func(p *PCA) { p.Components = [][]float64{{1}, {1}} },
		"variance":  func(p *PCA) { p.Variance = []float64{-1, 1} },
	} {
		c := *p
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("Validate %s", name)
		}
	}
	constant, err := Fit([][]float64{{1, 1}, {1, 1}}, Config{MinExplained: 0.9})
	if err != nil || constant.ExplainedRatio()[0] != 0 {
		t.Fatalf("constant data: %v", err)
	}
}
