package nn

import "fmt"

// Task is the kind of problem a network solves.
type Task int

const (
	// Classification maps inputs to one of a fixed set of labels.
	Classification Task = iota
	// Regression maps inputs to continuous values.
	Regression
)

var taskNames = [...]string{"classification", "regression"}

// String returns the name of the task.
func (t Task) String() string {
	if t < 0 || int(t) >= len(taskNames) {
		return fmt.Sprintf("Task(%d)", int(t))
	}
	return taskNames[t]
}

// MarshalText implements encoding.TextMarshaler.
func (t Task) MarshalText() ([]byte, error) {
	if t < 0 || int(t) >= len(taskNames) {
		return nil, fmt.Errorf("nn: invalid task %d", int(t))
	}
	return []byte(taskNames[t]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (t *Task) UnmarshalText(b []byte) error {
	for i, name := range taskNames {
		if name == string(b) {
			*t = Task(i)
			return nil
		}
	}
	return fmt.Errorf("nn: unknown task %q", b)
}
