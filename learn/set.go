package learn

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	neural "github.com/breskos/gopher-learn"
)

const (
	classYes       = 1.0
	labelRegressor = "output"
)

// Set holds the samples and the output labels
type Set struct {
	Samples      []Sample
	VectorHashes []string
	OutputHashes []string
	ClassToLabel map[int]string
	Usage        int
}

// NewSet creates a new set of empty data samples
func NewSet(usage int) *Set {
	return &Set{
		Samples:      make([]Sample, 0),
		ClassToLabel: make(map[int]string),
		Usage:        usage,
	}
}

// GetClasses returns the classes in the set
func (s *Set) GetClasses() []string {
	classes := make([]string, len(s.ClassToLabel))
	for k, v := range s.ClassToLabel {
		classes[k] = v
	}
	return classes
}

// TODO (abresk) two options: a) remove this function, b) put regression / classifciation add logic here
func (s *Set) add(vector, output []float64, label string, classNumber int, value float64) {
	var sample Sample
	sample.Vector = vector
	sample.Output = output
	sample.Label = label
	sample.ClassNumber = classNumber
	sample.Value = value
	sample.updateHashes()
	// register hashes in data set
	s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
	s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
	s.Samples = append(s.Samples, sample)
}

func (s *Set) addSample(sample Sample) {
	sample.updateHashes()
	s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
	s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
	s.Samples = append(s.Samples, sample)
}

func (s *Set) getLabelFromClass(number int) (string, bool) {
	if val, ok := s.ClassToLabel[number]; ok {
		return val, true
	}
	return "", false
}

func (s *Set) getClassFromLabel(label string) (int, bool) {
	for k, v := range s.ClassToLabel {
		if v == label {
			return k, true
		}
	}
	return -1, false
}

func (s *Set) distributionByLabel(label string) map[string]int {
	if s.Usage == neural.Classification {
		dist := make(map[string]int)
		for sample := range s.Samples {
			c := s.Samples[sample].Label
			if _, ok := dist[c]; ok {
				dist[c]++
			} else {
				dist[c] = 1
			}
		}
		return dist
	}
	return nil
}

func (s *Set) distributionByClassNumber(number int) map[int]int {
	if s.Usage == neural.Classification {
		dist := make(map[int]int)
		for sample := range s.Samples {
			c := s.Samples[sample].ClassNumber
			if _, ok := dist[c]; ok {
				dist[c]++
			} else {
				dist[c] = 1
			}
		}
		return dist
	}
	return nil
}

// LoadFromCSV where the last dimension is the label
func (s *Set) LoadFromCSV(path string) (bool, error) {
	classNumbers := make(map[string]int)
	classNumber := 0
	f, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("error while open file: %v", path)
	}
	defer f.Close()
	r := csv.NewReader(bufio.NewReader(f))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		l := len(record)
		var sample Sample
		sample.Vector = make([]float64, l-1)
		if s.Usage == neural.Regression {
			regression, err := strconv.ParseFloat(record[l-1], 64)
			if err == nil {
				sample.Value = regression
			}
		} else if s.Usage == neural.Classification {
			sample.Label = record[l-1]
			if _, ok := classNumbers[sample.Label]; !ok {
				classNumbers[sample.Label] = classNumber
				classNumber++
			}
		}
		for value := range record {
			if value < l-1 {
				f, err := strconv.ParseFloat(record[value], 64)
				if err != nil {
					return false, fmt.Errorf("failed to parse float %v with error: %v", record[value], err)
				}
				sample.Vector[value] = f
			}
		}
		sample.updateHashes()
		// register hashes in data set
		s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
		s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
		s.Samples = append(s.Samples, sample)
	}
	s.createClassToLabel(classNumbers)
	s.addOutputVectors()
	return true, nil
}

func (s *Set) addOutputVectors() {
	if s.Usage == neural.Classification {
		dim := len(s.ClassToLabel)
		for sample := range s.Samples {
			v := make([]float64, dim)
			v[s.Samples[sample].ClassNumber] = classYes
			s.Samples[sample].Output = v
		}
	} else if s.Usage == neural.Regression {
		for sample := range s.Samples {
			s.Samples[sample].Output = make([]float64, 1)
			s.Samples[sample].Output[0] = s.Samples[sample].Value
		}
	}
}

func (s *Set) createClassToLabel(mapping map[string]int) {
	s.ClassToLabel = make(map[int]string)
	if neural.Classification == s.Usage {
		for k, v := range mapping {
			s.ClassToLabel[v] = k
		}
		for i := range s.Samples {
			s.Samples[i].ClassNumber = mapping[s.Samples[i].Label]
		}
	} else {
		s.ClassToLabel[0] = labelRegressor
	}

}

// SampleExists looks up in the set if the presented example already exists
func (s *Set) SampleExists(test *Sample) bool {
	if test.VectorHash == "" {
		test.updateHashes()
	}
	for _, vector := range s.VectorHashes {
		if vector == test.VectorHash {
			return true
		}
	}
	return false
}

// LoadFromSVMFile load data from an svm problem file
func (s *Set) LoadFromSVMFile(path string) (bool, error) {
	classNumbers := make(map[string]int)
	classNumber := 0
	highestIndex := scanSamples(path)
	file, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("error while opening file")
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		m, label, err := problemToMap(line)
		if err != nil {
			return false, fmt.Errorf("error while scanning files: %v", err)
		}
		var sample Sample
		sample.Vector = make([]float64, highestIndex)
		sample.Label = label
		regression, err := strconv.ParseFloat(label, 64)
		if err != nil {
			sample.Value = regression
		}
		if _, ok := classNumbers[sample.Label]; !ok {
			classNumbers[sample.Label] = classNumber
			classNumber++
		}
		for i := 0; i < highestIndex; i++ {
			if val, ok := m[i]; ok {
				sample.Vector[i] = val
			} else {
				sample.Vector[i] = 0.0
			}
		}
		sample.updateHashes()
		// register hashes in data set
		s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
		s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
		s.Samples = append(s.Samples, sample)
	}
	return true, nil
}
