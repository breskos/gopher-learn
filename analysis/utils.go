package analysis

type sorter struct {
	x []float64
	y []float64
}

func (s sorter) Len() int           { return len(s.x) }
func (s sorter) Less(i, j int) bool { return s.x[i] < s.x[j] }
func (s sorter) Swap(i, j int) {
	s.x[i], s.x[j] = s.x[j], s.x[i]
	s.y[i], s.y[j] = s.y[j], s.y[i]
}
