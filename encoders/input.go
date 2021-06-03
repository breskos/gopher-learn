package encoders

type InputType int

const (
	String InputType = iota
	Floats
)

func (e InputType) String() string {
	return [...]string{"String", "Floats"}[e]
}

type Inputs struct {
	Inputs []*Input
}

type Input struct {
	Name   string
	Values []*Unified
	Type   InputType
}

type Unified struct {
	String string
	Float  []float64
	Type   InputType
	Label  string
	Target float64
}

func NewInputs() *Inputs {
	return &Inputs{
		Inputs: make([]*Input, 0),
	}
}

func (i *Inputs) Add(input *Input) {
	i.Inputs = append(i.Inputs, input)
}

func NewInput(name string, t InputType) *Input {
	return &Input{
		Name:   name,
		Type:   t,
		Values: make([]*Unified, 0),
	}
}

func (i *Input) Add(unified *Unified) {
	i.Values = append(i.Values, unified)
}

func (i *Input) AddFloats(sample []float64) {
	i.Values = append(i.Values, &Unified{Float: sample, Type: Floats})
}

func (i *Input) AddString(sample string) {
	i.Values = append(i.Values, &Unified{String: sample, Type: String})
}
