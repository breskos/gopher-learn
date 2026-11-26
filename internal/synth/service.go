package synth

import (
	"math"
	"math/rand/v2"
	"strconv"

	"github.com/breskos/gopher-learn/nn"
)

// Event logs for the service package: every row is a prediction request
// with the feedback that arrives for it later. feedback_delay is the number
// of seconds after t at which the feedback arrives (empty: never),
// feedback the label or value it reports (sometimes wrong), and
// correction_delay the seconds after the feedback at which the true label
// or value is sent again as a correction (empty: no correction).

// delaySeconds draws a log-normally distributed delay with the given median.
func delaySeconds(rng *rand.Rand, median float64) int {
	return max(1, int(median*math.Exp(0.8*rng.NormFloat64())))
}

func offers() Dataset {
	header := []string{"event", "t", "price_ratio", "visits", "basket_eur", "hour", "discount_pct", "concept",
		"accepted", "feedback_delay", "feedback", "correction_delay"}
	return Dataset{
		Path: "service/offers.csv",
		Description: "Event log of an online shop that predicts whether a customer accepts an offer (accepted: yes or no). " +
			"Features: price_ratio (own price / competitor price, 0.7–1.4), visits (previous visits, 0–20), basket_eur (value of the basket), " +
			"hour (0–23) and discount_pct (0, 5, 10 or 20). Up to 60 % of the stream, customers mainly react to visits and the basket (concept 1); " +
			"from then on they become price sensitive (concept 2). accepted is the true outcome. Its feedback arrives after feedback_delay " +
			"seconds (median 10 minutes, empty for 5 % that never get feedback); 8 % of the feedback is wrong, and 70 % of the wrong ones are " +
			"corrected correction_delay seconds later (median 1 day) with the true label. Replay the rows with the service package; t is in seconds.",
		Task:     nn.Classification,
		Rows:     40_000,
		Header:   header,
		Target:   "accepted",
		Ignore:   []string{"event", "t", "concept", "feedback_delay", "feedback", "correction_delay"},
		Loadable: true,
		Ordered:  true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, len(header))
			t := 0
			discounts := []float64{0, 0, 5, 10, 20}
			for i := range n {
				t += 1 + rng.IntN(40)
				price := 0.7 + 0.7*rng.Float64()
				visits := float64(min(20, int(rng.ExpFloat64()*4)))
				basket := min(500, 15*math.Exp(1.2*rng.NormFloat64()+1))
				hour := float64(rng.IntN(24))
				discount := discounts[rng.IntN(len(discounts))]
				evening := 0.0
				if hour >= 18 && hour <= 22 {
					evening = 1
				}
				concept := 1
				z := -2 + 0.35*visits + 0.012*basket - 2*(price-1) + 0.04*discount + 0.6*evening
				if i >= n*6/10 {
					concept = 2
					z = 0.3 + 0.08*visits + 0.002*basket - 9*(price-1) + 0.12*discount + 0.2*evening
				}
				truth := z+0.4*rng.NormFloat64() > 0
				label := func(yes bool) string {
					if yes {
						return "yes"
					}
					return "no"
				}
				row[0] = "o" + strconv.Itoa(i+1)
				row[1] = strconv.Itoa(t)
				row[2], row[3], row[4] = num(price), num(visits), num(math.Round(basket*100)/100)
				row[5], row[6] = num(hour), num(discount)
				row[7] = strconv.Itoa(concept)
				row[8] = label(truth)
				row[9], row[10], row[11] = "", "", ""
				if rng.Float64() >= 0.05 {
					row[9] = strconv.Itoa(delaySeconds(rng, 600))
					wrong := rng.Float64() < 0.08
					row[10] = label(truth != wrong)
					if wrong && rng.Float64() < 0.7 {
						row[11] = strconv.Itoa(delaySeconds(rng, 86_400))
					}
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func deliveries() Dataset {
	header := []string{"event", "t", "distance_km", "items", "traffic", "hour", "rain", "concept",
		"minutes", "feedback_delay", "feedback", "correction_delay"}
	return Dataset{
		Path: "service/deliveries.csv",
		Description: "Event log of a delivery service that predicts the delivery time in minutes. Features: distance_km (0.5–25), items (1–30), " +
			"traffic (0–1), hour (0–23) and rain (0 or 1). From 60 % of the stream on, a new depot makes deliveries about 20 % faster (concept 2). " +
			"minutes is the true time, in raw units (not scaled to [0, 1]). The feedback arrives when the delivery is done (feedback_delay ≈ minutes·60 " +
			"seconds; empty for 3 % without feedback). 1.5 % of the feedback is corrupt: reported in seconds instead of minutes, or with a typo (×10); " +
			"80 % of those are corrected correction_delay seconds later (median 2 hours). Replay the rows with the service package; t is in seconds.",
		Task:      nn.Regression,
		Rows:      40_000,
		Header:    header,
		Target:    "minutes",
		Ignore:    []string{"event", "t", "concept", "feedback_delay", "feedback", "correction_delay"},
		Loadable:  true,
		Ordered:   true,
		Applied:   true,
		RawTarget: true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, len(header))
			t := 0
			for i := range n {
				t += 1 + rng.IntN(30)
				dist := 0.5 + 24.5*math.Pow(rng.Float64(), 1.5)
				items := float64(1 + min(29, int(rng.ExpFloat64()*6)))
				hour := float64(8 + rng.IntN(14))
				rush := 0.0
				if (hour >= 11 && hour <= 13) || (hour >= 17 && hour <= 19) {
					rush = 1
				}
				traffic := clamp01(0.25 + 0.4*rush + 0.15*rng.NormFloat64())
				rain := 0.0
				if rng.Float64() < 0.2 {
					rain = 1
				}
				concept := 1
				minutes := 8 + 3*dist + 0.5*items + 25*traffic*(1+dist/25) + 6*rain
				if i >= n*6/10 {
					concept = 2
					minutes = 0.8*minutes - 2
				}
				minutes = max(5, minutes+3*rng.NormFloat64())
				minutes = math.Round(minutes*10) / 10
				row[0] = "d" + strconv.Itoa(i+1)
				row[1] = strconv.Itoa(t)
				row[2], row[3], row[4] = num(math.Round(dist*100)/100), num(items), num(math.Round(traffic*1000)/1000)
				row[5], row[6] = num(hour), num(rain)
				row[7] = strconv.Itoa(concept)
				row[8] = num(minutes)
				row[9], row[10], row[11] = "", "", ""
				if rng.Float64() >= 0.03 {
					row[9] = strconv.Itoa(int(minutes*60) + rng.IntN(120))
					reported := minutes
					corrupt := rng.Float64() < 0.015
					if corrupt {
						if rng.IntN(2) == 0 {
							reported = math.Round(minutes * 60) // seconds instead of minutes
						} else {
							reported = minutes * 10 // a typo
						}
					}
					row[10] = num(reported)
					if corrupt && rng.Float64() < 0.8 {
						row[11] = strconv.Itoa(delaySeconds(rng, 7_200))
					}
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}
