package learn

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	neural "github.com/breskos/gopher-learn/net"
)

const (
	maxActivation  = 1.0
	minActivation  = 0.0
	labelRegressor = "output"

	errUsageTypeNotMatching = "usage type not matching"
	errClassLabelNotFound   = "class label not found in set"
)

// Set holds the samples and the output labels
type Set struct {
	Samples      []*Sample
	VectorHashes []string
	OutputHashes []string
	ClassToLabel map[int]string
	Usage        neural.NetworkType
}

// NewSet creates a new set of empty data samples
func NewSet(usage neural.NetworkType) *Set {
	return &Set{
		Samples:      make([]*Sample, 0),
		ClassToLabel: make(map[int]string),
		Usage:        usage,
	}
}

// AddClass returns the classes in the set
func (s *Set) AddClass(label string) (bool, error) {
	if s.Usage != neural.Classification {
		return false, fmt.Errorf(errUsageTypeNotMatching)
	}
	l := len(s.ClassToLabel)
	s.ClassToLabel[l] = label
	return true, nil
}

// GetClasses returns the classes in the set
func (s *Set) GetClasses() []string {
	classes := make([]string, len(s.ClassToLabel))
	for k, v := range s.ClassToLabel {
		classes[k] = v
	}
	return classes
}

// Adds a vector with the corresponding output to the data set
func (s *Set) add(vector, output []float64, label string, classNumber int, value float64) {
	sample := &Sample{}
	sample.Vector = vector
	sample.Output = output
	sample.Label = label
	sample.ClassNumber = classNumber
	sample.Value = value
	sample.UpdateHashes()
	// register hashes in data set
	s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
	s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
	s.Samples = append(s.Samples, sample)
}

// AddSample adds samples to the set
func (s *Set) AddSample(sample *Sample) error {
	index, err := s.getClassIndex(sample.Label)
	if err != nil {
		return err
	}
	sample.ClassNumber = index
	sample.UpdateHashes()
	s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
	s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
	s.Samples = append(s.Samples, sample)
	return nil
}

// Returns the label from a given class number
func (s *Set) getLabelFromClass(number int) (string, bool) {
	if val, ok := s.ClassToLabel[number]; ok {
		return val, true
	}
	return "", false
}

// Returns the class from the corresponding label
func (s *Set) getClassFromLabel(label string) (int, bool) {
	for k, v := range s.ClassToLabel {
		if v == label {
			return k, true
		}
	}
	return -1, false
}

// Shows the distribution of the data set by the label
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

// Shows the distribution by class number of a the data set
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
		sample := &Sample{}
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
		sample.UpdateHashes()
		// register hashes in data set
		s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
		s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
		s.Samples = append(s.Samples, sample)
	}
	s.createClassToLabel(classNumbers)
	s.addOutputVectors()
	return true, nil
}

// Adds the output vectors for the 2 cases classification or regression
func (s *Set) addOutputVectors() {
	if s.Usage == neural.Classification {
		dim := len(s.ClassToLabel)
		for sample := range s.Samples {
			v := make([]float64, dim)
			v[s.Samples[sample].ClassNumber] = maxActivation
			s.Samples[sample].Output = v
		}
	} else if s.Usage == neural.Regression {
		for sample := range s.Samples {
			s.Samples[sample].Output = make([]float64, 1)
			s.Samples[sample].Output[0] = s.Samples[sample].Value
		}
	}
}

// Creats a class to the label and adding it to the set
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
		test.UpdateHashes()
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
		sample := &Sample{}
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
		sample.UpdateHashes()
		// register hashes in data set
		s.VectorHashes = append(s.VectorHashes, sample.VectorHash)
		s.OutputHashes = append(s.OutputHashes, sample.OutputHash)
		s.Samples = append(s.Samples, sample)
	}
	return true, nil
}

// GenerateOutputVector generates the output vector for a classification task and a specific label
func (s *Set) GenerateOutputVector(label string) []float64 {
	var output []float64
	for _, v := range s.ClassToLabel {
		if v == label {
			output = append(output, maxActivation)
		} else {
			output = append(output, minActivation)
		}
	}
	return output
}

// Returns the class index of a string label
func (s *Set) getClassIndex(label string) (int, error) {
	for k, v := range s.ClassToLabel {
		if label == v {
			return k, nil
		}
	}
	return 0, errors.New(errUsageTypeNotMatching)
}
