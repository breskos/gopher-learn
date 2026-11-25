package nn

// dot returns the dot product of a and b[:len(a)]. Four independent
// accumulators break the dependency chain of a single running sum, so the
// CPU can overlap the multiply-adds.
func dot(a, b []float64) float64 {
	b = b[:len(a)]
	if len(a) < 8 {
		var s float64
		for i, x := range a {
			s += x * b[i]
		}
		return s
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float64
	i := 0
	for ; i+8 <= len(a); i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

// axpy computes y += alpha * x for y[:len(x)].
func axpy(alpha float64, x, y []float64) {
	y = y[:len(x)]
	i := 0
	for ; i+4 <= len(x); i += 4 {
		y[i] += alpha * x[i]
		y[i+1] += alpha * x[i+1]
		y[i+2] += alpha * x[i+2]
		y[i+3] += alpha * x[i+3]
	}
	for ; i < len(x); i++ {
		y[i] += alpha * x[i]
	}
}
