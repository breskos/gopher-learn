package synth

import (
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"

	"github.com/breskos/gopher-learn/nn"
)

// All returns every data set in a stable order.
func All() []Dataset {
	return []Dataset{
		blobs(), imbalanced(), spirals(), highDim(),
		friedman1(), unscaled(),
		drift(),
		intents(),
		messySemicolon(), messyMissing(), sparseSVM(),
		large(),
	}
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func clamp01(x float64) float64 { return min(max(x, 0), 1) }

func blobs() Dataset {
	const classes, features = 4, 8
	return Dataset{
		Path: "classification/blobs.csv",
		Description: "Four Gaussian clusters (standard deviation 1) in 8 dimensions with centers drawn uniformly from [-5, 5]. " +
			"Classes are balanced and well separated, so a small network should reach close to 100 % accuracy.",
		Task:     nn.Classification,
		Rows:     20_000,
		Header:   append(featureNames(features), "label"),
		Target:   "label",
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			centers := make([][]float64, classes)
			for c := range centers {
				centers[c] = make([]float64, features)
				for i := range centers[c] {
					centers[c][i] = rng.Float64()*10 - 5
				}
			}
			row := make([]string, features+1)
			for range n {
				c := rng.IntN(classes)
				for i := range features {
					row[i] = num(centers[c][i] + rng.NormFloat64())
				}
				row[features] = "class_" + strconv.Itoa(c)
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func imbalanced() Dataset {
	const features, informative = 10, 5
	return Dataset{
		Path: "classification/imbalanced.csv",
		Description: "Binary classification with 5 % positives. Features x1–x5 are shifted by +1.5 for positives, x6–x10 are noise. " +
			"The id column must be ignored. Plain accuracy is misleading here (95 % by always predicting \"neg\"); use balanced accuracy or F1.",
		Task:     nn.Classification,
		Rows:     50_000,
		Header:   append(append([]string{"id"}, featureNames(features)...), "label"),
		Target:   "label",
		Ignore:   []string{"id"},
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, features+2)
			for r := range n {
				positive := rng.Float64() < 0.05
				row[0] = strconv.Itoa(r + 1)
				for i := range features {
					x := rng.NormFloat64()
					if positive && i < informative {
						x += 1.5
					}
					row[i+1] = num(x)
				}
				row[features+1] = "neg"
				if positive {
					row[features+1] = "pos"
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func spirals() Dataset {
	const classes = 3
	return Dataset{
		Path: "classification/spirals.csv",
		Description: "Three interleaved spirals in 2D with small noise. The classes are not linearly separable, " +
			"so this needs hidden layers and enough training; a good network reaches more than 90 % accuracy.",
		Task:     nn.Classification,
		Rows:     10_000,
		Header:   []string{"x", "y", "label"},
		Target:   "label",
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, 3)
			for range n {
				c := rng.IntN(classes)
				t := rng.Float64()
				angle := t*2*math.Pi + float64(c)*2*math.Pi/classes
				r := 0.1 + t
				row[0] = num(r*math.Cos(angle) + rng.NormFloat64()*0.03)
				row[1] = num(r*math.Sin(angle) + rng.NormFloat64()*0.03)
				row[2] = "spiral_" + strconv.Itoa(c)
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func highDim() Dataset {
	const features, informative, redundant = 200, 10, 20
	return Dataset{
		Path: "classification/high-dim.csv",
		Description: "Binary classification with 200 features: x1–x10 are informative (the label is the sign of a weighted sum plus noise), " +
			"x11–x30 are noisy copies of x1–x10 (redundant, highly correlated), x31–x200 are pure noise. " +
			"Useful for the float reducer and for feature importance.",
		Task:     nn.Classification,
		Rows:     10_000,
		Header:   append(featureNames(features), "label"),
		Target:   "label",
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			weights := make([]float64, informative)
			for i := range weights {
				weights[i] = rng.NormFloat64()
			}
			x := make([]float64, features)
			row := make([]string, features+1)
			for range n {
				var score float64
				for i := range informative {
					x[i] = rng.NormFloat64()
					score += weights[i] * x[i]
				}
				for i := informative; i < informative+redundant; i++ {
					x[i] = x[i%informative] + rng.NormFloat64()*0.1
				}
				for i := informative + redundant; i < features; i++ {
					x[i] = rng.NormFloat64()
				}
				for i, v := range x {
					row[i] = num(v)
				}
				row[features] = "no"
				if score+rng.NormFloat64()*0.5 > 0 {
					row[features] = "yes"
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func friedman1() Dataset {
	const features = 10
	return Dataset{
		Path: "regression/friedman1.csv",
		Description: "The Friedman #1 benchmark: x1–x10 uniform in [0, 1], " +
			"y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + N(0, 1), so x6–x10 are irrelevant. " +
			"The target is rescaled with (y + 3) / 36 and clipped to [0, 1] to fit a sigmoid output.",
		Task:     nn.Regression,
		Rows:     20_000,
		Header:   append(featureNames(features), "y"),
		Target:   "y",
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			x := make([]float64, features)
			row := make([]string, features+1)
			for range n {
				for i := range x {
					x[i] = rng.Float64()
					row[i] = num(x[i])
				}
				y := 10*math.Sin(math.Pi*x[0]*x[1]) + 20*(x[2]-0.5)*(x[2]-0.5) + 10*x[3] + 5*x[4] + rng.NormFloat64()
				row[features] = num(clamp01((y + 3) / 36))
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func unscaled() Dataset {
	return Dataset{
		Path: "regression/unscaled.csv",
		Description: "Regression with features on very different scales: tiny ~ N(0, 0.001), unit ~ U(0, 1), medium ~ N(50, 10), " +
			"large ~ U(0, 1000), wide ~ N(0, 100) and skewed = exp(N(0, 1)). The target is a sigmoid of a weighted sum of the " +
			"standardized features plus one interaction, with small noise. Training without feature scaling performs clearly worse.",
		Task:     nn.Regression,
		Rows:     20_000,
		Header:   []string{"tiny", "unit", "medium", "large", "wide", "skewed", "y"},
		Target:   "y",
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, 7)
			for range n {
				z := [6]float64{rng.NormFloat64(), rng.Float64()*2 - 1, rng.NormFloat64(), rng.Float64()*2 - 1, rng.NormFloat64(), rng.NormFloat64()}
				raw := [6]float64{
					z[0] * 0.001,
					(z[1] + 1) / 2,
					50 + z[2]*10,
					(z[3] + 1) * 500,
					z[4] * 100,
					math.Exp(z[5]),
				}
				for i, v := range raw {
					row[i] = num(v)
				}
				s := 1.5*z[0] - 1.0*z[1] + 0.8*z[2] + 1.2*z[3] - 0.5*z[4] + 0.7*z[5] + 0.5*z[0]*z[3]
				row[6] = num(clamp01(sigmoid(s) + rng.NormFloat64()*0.02))
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func drift() Dataset {
	return Dataset{
		Path: "stream/drift.csv",
		Description: "A stream for online learning with an abrupt concept drift in the middle. x1–x4 are uniform in [-1, 1]; " +
			"in the first half the label is A if x1 + x2 > 0, in the second half if x1 - x2 > 0. x3 and x4 are noise and 5 % of the labels are flipped. " +
			"Read the rows in order; the t column is the time step and must be ignored.",
		Task:     nn.Classification,
		Rows:     100_000,
		Header:   []string{"t", "x1", "x2", "x3", "x4", "label"},
		Target:   "label",
		Ignore:   []string{"t"},
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, 6)
			for t := range n {
				var x [4]float64
				for i := range x {
					x[i] = rng.Float64()*2 - 1
				}
				a := x[0]+x[1] > 0
				if t >= n/2 {
					a = x[0]-x[1] > 0
				}
				if rng.Float64() < 0.05 {
					a = !a
				}
				row[0] = strconv.Itoa(t)
				for i, v := range x {
					row[i+1] = num(v)
				}
				row[5] = "B"
				if a {
					row[5] = "A"
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

var intentTemplates = map[string][]string{
	"GREETING": {
		"Hallo {name}!", "Guten {daytime}, wie geht es dir?", "Hi, ich bin {name}.",
		"Servus {name}, lange nicht gesehen", "Moin! Schön, dass du da bist.", "Grüß dich {name}",
	},
	"WEATHER": {
		"Wie wird das Wetter {when} in {city}?", "Regnet es {when} in {city}?", "Brauche ich {when} einen Schirm?",
		"Wie warm wird es {when}?", "Scheint {when} in {city} die Sonne?", "Gibt es {when} Schnee in {city}?",
	},
	"ORDER": {
		"Ich möchte {count} {item} bestellen.", "Bitte liefere {count} {item} nach {city}.", "Kann ich {item} {when} abholen?",
		"Füge {count} {item} zum Warenkorb hinzu", "Bestellung: {count}x {item}", "Ich hätte gern {item}, bitte.",
	},
	"COMPLAINT": {
		"Meine Bestellung von {item} ist nicht angekommen.", "Der {item} war leider kaputt.", "Ich warte seit {count} Tagen auf {item}!",
		"Das ist schon das {count}. Mal, dass {item} fehlt.", "Ich will mein Geld für {item} zurück.", "Der Service in {city} war schlecht.",
	},
}

var intentSlots = map[string][]string{
	"{name}":    {"Anna", "Ben", "Clara", "David", "Emil", "Fatma", "Jonas", "Lea"},
	"{daytime}": {"Morgen", "Tag", "Abend"},
	"{when}":    {"heute", "morgen", "am Wochenende", "übermorgen", "nächste Woche"},
	"{city}":    {"Berlin", "Hamburg", "München", "Köln", "Leipzig", "Dresden"},
	"{count}":   {"1", "2", "3", "5", "10", "zwei", "drei"},
	"{item}":    {"Pizza", "Kaffee", "Bücher", "Schuhe", "ein Fahrrad", "Blumen", "Tee"},
}

func intents() Dataset {
	labels := []string{"COMPLAINT", "GREETING", "ORDER", "WEATHER"}
	return Dataset{
		Path: "text/intents.phrase",
		Description: "Short German utterances generated from templates for four intents (GREETING, WEATHER, ORDER, COMPLAINT), " +
			"one \"text#LABEL\" per line like the ngram example. For the string encoders (dictionary, split dictionary, n-grams).",
		Task:   nn.Classification,
		Format: Lines,
		Rows:   10_000,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			line := make([]string, 1)
			for range n {
				label := labels[rng.IntN(len(labels))]
				tpl := intentTemplates[label]
				text := tpl[rng.IntN(len(tpl))]
				for _, slot := range intentSlotOrder {
					values := intentSlots[slot]
					for strings.Contains(text, slot) {
						text = strings.Replace(text, slot, values[rng.IntN(len(values))], 1)
					}
				}
				line[0] = text + "#" + label
				if err := emit(line); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

// intentSlotOrder fixes the replacement order so the output is
// deterministic (map iteration order is random).
var intentSlotOrder = []string{"{name}", "{daytime}", "{when}", "{city}", "{count}", "{item}"}

func messySemicolon() Dataset {
	species := []string{"Art A", "Art B", "Art C"}
	return Dataset{
		Path: "messy/semicolon.csv",
		Description: "A CSV as exported by German spreadsheet tools: semicolon delimiter, an id column, column names with spaces and umlauts, " +
			"and labels with spaces. Load it with learn.WithComma(';'), learn.WithTarget(\"Art\") and learn.WithIgnore(\"id\").",
		Task:     nn.Classification,
		Rows:     1_000,
		Header:   []string{"id", "Länge (cm)", "Breite (cm)", "Höhe (cm)", "Art"},
		Target:   "Art",
		Ignore:   []string{"id"},
		Comma:    ';',
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, 5)
			for r := range n {
				c := rng.IntN(len(species))
				row[0] = "#" + strconv.Itoa(1000+r)
				row[1] = num(5 + float64(c)*1.5 + rng.NormFloat64()*0.5)
				row[2] = num(3 + float64(2-c)*0.8 + rng.NormFloat64()*0.4)
				row[3] = num(1 + float64(c)*0.3 + rng.NormFloat64()*0.2)
				row[4] = species[c]
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func messyMissing() Dataset {
	return Dataset{
		Path: "messy/missing-values.csv",
		Description: "Like blobs but with 2 % empty fields. learn.Set.ReadCSV rejects it with a \"missing value\" error that names the column; " +
			"it is the test case for a future imputation feature.",
		Task:   nn.Classification,
		Rows:   1_000,
		Header: append(featureNames(4), "label"),
		Target: "label",
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, 5)
			for range n {
				c := rng.IntN(2)
				for i := range 4 {
					row[i] = num(float64(c)*2 + rng.NormFloat64())
					if rng.Float64() < 0.02 {
						row[i] = ""
					}
				}
				row[4] = "c" + strconv.Itoa(c)
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func sparseSVM() Dataset {
	const features, active = 1000, 10
	return Dataset{
		Path: "messy/sparse.svm",
		Description: "Sparse LIBSVM data with 1,000 features and about 10 non-zero values per row. The label (+1/-1) is the sign of a weighted sum " +
			"of the active features. Load it with learn.Set.LoadSVM.",
		Task:   nn.Classification,
		Format: Lines,
		Rows:   5_000,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			weights := make([]float64, features)
			for i := range weights {
				weights[i] = rng.NormFloat64()
			}
			line := make([]string, 1)
			var b strings.Builder
			for range n {
				idx := make(map[int]float64, active)
				for len(idx) < active {
					idx[1+rng.IntN(features)] = rng.Float64()
				}
				keys := slices.Sorted(maps.Keys(idx))
				var score float64
				for _, k := range keys {
					score += weights[k-1] * idx[k]
				}
				b.Reset()
				if score > 0 {
					b.WriteString("+1")
				} else {
					b.WriteString("-1")
				}
				for _, k := range keys {
					b.WriteString(" " + strconv.Itoa(k) + ":" + num(idx[k]))
				}
				line[0] = b.String()
				if err := emit(line); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func large() Dataset {
	const features, informative = 20, 8
	return Dataset{
		Path: "large/benchmark.csv",
		Description: "A larger binary classification set for performance measurements: 20 standard normal features, " +
			"the label is drawn from a logistic model on x1–x8 (x9–x20 are noise).",
		Task:     nn.Classification,
		Rows:     250_000,
		Header:   append(featureNames(features), "label"),
		Target:   "label",
		Loadable: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			weights := make([]float64, informative)
			for i := range weights {
				weights[i] = rng.NormFloat64()
			}
			row := make([]string, features+1)
			for range n {
				var score float64
				for i := range features {
					x := rng.NormFloat64()
					if i < informative {
						score += weights[i] * x
					}
					row[i] = num(x)
				}
				row[features] = "0"
				if rng.Float64() < sigmoid(2*score) {
					row[features] = "1"
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}
