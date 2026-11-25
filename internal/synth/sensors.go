package synth

import (
	"math"
	"math/rand/v2"
	"strconv"
	"time"

	"github.com/breskos/gopher-learn/nn"
)

// The sensor data sets below are generated from simple physical and
// behavioral models. They have the structure of real sensor data, but they
// are not measurements.

// sensors returns the applied sensor data sets.
func sensors() []Dataset {
	return []Dataset{
		predictiveMaintenance(), runToFailure(),
		activity(),
		roomClimate(),
		airQualityCalibration(),
		sensorHealth(),
		wifiLocalization(),
		smartMeter(),
		airQualityAging(),
	}
}

var epoch = time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

func stamp(t time.Time) string { return t.Format("2006-01-02T15:04") }

func logNormal(rng *rand.Rand, sigma float64) float64 { return math.Exp(rng.NormFloat64() * sigma) }

// vibration holds the features of one accelerometer window at a bearing.
type vibration struct {
	rpm, temp, x1, x2, bpfo, noise, kurtosis float64
}

func (v vibration) row(rng *rand.Rand) []string {
	rms := math.Sqrt(v.noise*v.noise + (v.x1*v.x1+v.x2*v.x2+v.bpfo*v.bpfo)/2)
	crest := 2.9 + 0.35*(v.kurtosis-3) + rng.NormFloat64()*0.15
	return []string{
		num(v.rpm), num(v.temp), num(rms), num(rms * crest), num(crest), num(v.kurtosis),
		num(v.x1), num(v.x2), num(v.bpfo),
	}
}

var vibrationHeader = []string{"rpm", "temperature_c", "rms_g", "peak_g", "crest_factor", "kurtosis", "amp_1x_g", "amp_2x_g", "amp_bpfo_g"}

func predictiveMaintenance() Dataset {
	states := []string{"normal", "imbalance", "misalignment", "bearing_fault"}
	return Dataset{
		Path: "classification/predictive-maintenance.csv",
		Description: "Condition monitoring of electric motors with an accelerometer at the bearing (e.g. ADXL345), a temperature probe and the speed. " +
			"Each row is one window: overall RMS and peak acceleration, crest factor, kurtosis and the spectral amplitudes at the rotation frequency (1x), " +
			"twice the rotation frequency (2x) and the bearing outer race fault frequency (BPFO). " +
			"Imbalance raises 1x, misalignment raises 2x and the temperature, bearing faults raise BPFO, kurtosis and crest factor. " +
			"Fault severity varies, so mild faults overlap with normal operation. The machine column is an ID and must be ignored.",
		Task:     nn.Classification,
		Rows:     20_000,
		Header:   append(append([]string{"machine"}, vibrationHeader...), "state"),
		Target:   "state",
		Ignore:   []string{"machine"},
		Loadable: true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			for range n {
				machine := rng.IntN(40)
				state := 0
				if rng.Float64() < 0.4 { // 60 % normal, the rest split across faults
					state = 1 + rng.IntN(3)
				}
				rpm := 900 + rng.Float64()*2100
				load := (rpm / 1500) * (rpm / 1500)
				severity := 0.15 + rng.Float64()*0.85
				v := vibration{
					rpm:      rpm,
					temp:     22 + rng.Float64()*10 + rpm*0.008 + rng.NormFloat64(),
					x1:       0.02 * load * logNormal(rng, 0.3),
					x2:       0.008 * load * logNormal(rng, 0.3),
					bpfo:     0.004 * logNormal(rng, 0.3),
					noise:    0.03 * math.Pow(rpm/1500, 1.5) * logNormal(rng, 0.2),
					kurtosis: 3 + rng.NormFloat64()*0.3,
				}
				switch state {
				case 1:
					v.x1 *= 1 + 6*severity
				case 2:
					v.x2 *= 1 + 8*severity
					v.temp += 6 * severity
				case 3:
					v.bpfo *= 1 + 25*severity
					v.kurtosis += 8 * severity * logNormal(rng, 0.2)
					v.temp += 8 * severity
				}
				row := append([]string{"M" + strconv.Itoa(100+machine)}, v.row(rng)...)
				if err := emit(append(row, states[state])); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func runToFailure() Dataset {
	return Dataset{
		Path: "regression/run-to-failure.csv",
		Description: "Bearings run until failure. Every machine has a random life of 800 to 2,000 operating hours and is measured every 10 hours. " +
			"Wear grows slowly at first and accelerates towards the end (health = (hours / life)^2.5): BPFO amplitude, kurtosis and temperature rise. " +
			"The target rul is the remaining useful life as a fraction of the total life (1 = new, 0 = failed). The machine column must be ignored; " +
			"for a fair evaluation split by machine, not by row.",
		Task:     nn.Regression,
		Rows:     20_000,
		Header:   append(append([]string{"machine", "hours"}, vibrationHeader...), "rul"),
		Target:   "rul",
		Ignore:   []string{"machine"},
		Loadable: true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			machine, hours, life, rpm := 0, 0.0, 0.0, 0.0
			for range n {
				if hours >= life {
					machine++
					hours, life, rpm = 0, 800+rng.Float64()*1200, 1200+rng.Float64()*1500
				}
				health := math.Pow(hours/life, 2.5)
				load := (rpm / 1500) * (rpm / 1500)
				v := vibration{
					rpm:      rpm + rng.NormFloat64()*15,
					temp:     40 + 18*health + rng.NormFloat64()*1.5,
					x1:       0.02 * load * logNormal(rng, 0.2) * (1 + health),
					x2:       0.008 * load * logNormal(rng, 0.2),
					bpfo:     (0.004 + 0.25*health) * logNormal(rng, 0.25),
					noise:    0.03 * math.Pow(rpm/1500, 1.5) * logNormal(rng, 0.15),
					kurtosis: 3 + 9*health*logNormal(rng, 0.2) + rng.NormFloat64()*0.3,
				}
				row := append([]string{"B" + strconv.Itoa(machine), num(hours)}, v.row(rng)...)
				if err := emit(append(row, num(clamp01(1-hours/life)))); err != nil {
					return err
				}
				hours += 10
			}
			return nil
		},
	}
}

type activityProfile struct {
	name                  string
	freq, accStd, gyroStd float64
	gravity               [3]float64 // direction of gravity in phone coordinates
}

var activities = []activityProfile{
	{"walking", 1.8, 0.25, 0.6, [3]float64{0.1, 0.98, 0.15}},
	{"running", 2.7, 0.80, 1.5, [3]float64{0.15, 0.95, 0.25}},
	{"upstairs", 1.6, 0.30, 0.7, [3]float64{0.25, 0.92, 0.3}},
	{"downstairs", 1.9, 0.40, 0.8, [3]float64{0.05, 0.96, 0.25}},
	{"cycling", 1.3, 0.15, 1.2, [3]float64{0.2, 0.3, 0.93}},
	{"sitting", 0, 0.01, 0.02, [3]float64{0.1, 0.25, 0.96}},
	{"standing", 0, 0.015, 0.03, [3]float64{0.05, 0.99, 0.1}},
}

func activity() Dataset {
	header := []string{"user",
		"acc_mean_x", "acc_mean_y", "acc_mean_z", "acc_std_x", "acc_std_y", "acc_std_z",
		"acc_mag_std", "acc_dom_freq_hz", "gyro_std_x", "gyro_std_y", "gyro_std_z", "gyro_mag_mean", "activity"}
	return Dataset{
		Path: "classification/activity.csv",
		Description: "Human activity recognition with a smartphone in the trouser pocket (3-axis accelerometer in g and gyroscope in rad/s, e.g. MPU-6050, 50 Hz). " +
			"Each row summarizes a 2.56 s window: mean and standard deviation per axis, the standard deviation of the magnitude, the dominant step frequency " +
			"and the gyroscope activity. Seven activities: walking, running, upstairs, downstairs, cycling, sitting, standing. " +
			"30 users differ in step frequency, intensity and how the phone sits in the pocket, so walking and the stairs overlap. " +
			"The user column must be ignored; for a fair evaluation split by user.",
		Task:     nn.Classification,
		Rows:     30_000,
		Header:   header,
		Target:   "activity",
		Ignore:   []string{"user"},
		Loadable: true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			type user struct{ pace, intensity, tilt float64 }
			users := make([]user, 30)
			for i := range users {
				users[i] = user{0.85 + rng.Float64()*0.3, 0.75 + rng.Float64()*0.5, rng.NormFloat64() * 0.15}
			}
			row := make([]string, len(header))
			for range n {
				u := rng.IntN(len(users))
				a := activities[rng.IntN(len(activities))]
				usr := users[u]
				// tilt rotates gravity between y and z
				g := a.gravity
				gy := g[1]*math.Cos(usr.tilt) - g[2]*math.Sin(usr.tilt)
				gz := g[1]*math.Sin(usr.tilt) + g[2]*math.Cos(usr.tilt)
				mean := [3]float64{g[0], gy, gz}
				std := a.accStd * usr.intensity * logNormal(rng, 0.15)
				row[0] = "U" + strconv.Itoa(u+1)
				for i := range 3 {
					row[1+i] = num(mean[i] + rng.NormFloat64()*0.03)
					axis := []float64{0.6, 1, 0.7}[i]
					row[4+i] = num(std * axis * logNormal(rng, 0.1))
				}
				row[7] = num(std * 1.1 * logNormal(rng, 0.1))
				freq := a.freq * usr.pace * logNormal(rng, 0.05)
				if a.freq == 0 {
					freq = rng.Float64() * 0.4 // no periodic movement
				}
				row[8] = num(freq)
				gyro := a.gyroStd * usr.intensity * logNormal(rng, 0.15)
				for i := range 3 {
					row[9+i] = num(gyro * []float64{1, 0.6, 0.8}[i] * logNormal(rng, 0.1))
				}
				row[12] = num(gyro * 1.3 * logNormal(rng, 0.1))
				row[13] = a.name
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func roomClimate() Dataset {
	header := []string{"timestamp", "hour", "weekday", "temperature_c", "humidity_pct", "pressure_hpa", "co2_ppm", "light_lux", "motion_events", "people", "occupied"}
	return Dataset{
		Path: "online/room-occupancy.csv",
		Description: "An office room measured every 5 minutes (e.g. BME280 for temperature, humidity and pressure, SCD30 for CO2, a light sensor and a PIR motion sensor), " +
			"starting 2025-01-01. People come on working days roughly from 8:00 to 18:00 with a lunch break, sometimes stay late or come on a weekend. " +
			"CO2 rises with every person and falls with ventilation and open windows, people warm the room and add humidity, lamps go on when someone is there, " +
			"lamps are often left off on bright days, and the PIR sensor counts movements but misses people sitting still and has a few false triggers. The target occupied says whether anyone is in the room; " +
			"people (the head count) can be used as an alternative regression target. Ignore timestamp and people when predicting occupied. " +
			"The rows form a time series; read them in order for online learning.",
		Task:     nn.Classification,
		Rows:     8 * 7 * 288, // eight weeks
		Header:   header,
		Target:   "occupied",
		Ignore:   []string{"timestamp", "people"},
		Loadable: true,
		Applied:  true,
		Ordered:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			co2, temp, hum, pressure := 450.0, 20.0, 40.0, 1013.0
			people, windowOpen := 0, 0
			lampsOn := false
			row := make([]string, len(header))
			for i := range n {
				t := epoch.Add(time.Duration(i) * 5 * time.Minute)
				hour := float64(t.Hour()) + float64(t.Minute())/60
				weekday := int(t.Weekday())
				workday := weekday >= 1 && weekday <= 5
				// expected head count by time of day
				expected := 0.0
				switch {
				case workday && hour >= 8 && hour < 18:
					expected = 4
					if hour >= 12 && hour < 13 {
						expected = 1.5
					}
				case workday && hour >= 7 && hour < 8, workday && hour >= 18 && hour < 20:
					expected = 0.6
				case !workday && hour >= 10 && hour < 15:
					expected = 0.2
				}
				// head count changes slowly towards the expectation
				if rng.Float64() < 0.15 {
					target := int(math.Round(expected * logNormal(rng, 0.4)))
					if expected < 1 && rng.Float64() > expected {
						target = 0
					}
					people = min(max(target, 0), 8)
				}
				if windowOpen > 0 {
					windowOpen--
				} else if people > 0 && co2 > 1200 && rng.Float64() < 0.3 {
					windowOpen = 2 + rng.IntN(3)
				}
				outdoor := 4 + 5*math.Sin((hour-9)/24*2*math.Pi)
				vent := 0.04
				if windowOpen > 0 {
					vent = 0.35
				}
				co2 += float64(people)*18 - vent*(co2-420) + rng.NormFloat64()*4
				heating := 21.0
				if !workday || hour < 6 || hour >= 20 {
					heating = 17
				}
				temp += 0.08*(heating-temp) + 0.05*float64(people) + vent*0.3*(outdoor-temp) + rng.NormFloat64()*0.05
				hum += 0.4*float64(people)*0.25 - vent*(hum-35)*0.5 - 0.02*(hum-40) + rng.NormFloat64()*0.3
				pressure += rng.NormFloat64()*0.08 - 0.001*(pressure-1013)
				daylight := math.Max(0, math.Sin((hour-7.5)/9*math.Pi)) * 450 * logNormal(rng, 0.4)
				// Lamps are switched on when someone is there and it is dark;
				// on bright days people often leave them off.
				if people == 0 {
					lampsOn = false
				} else if !lampsOn && rng.Float64() < 0.3 {
					lampsOn = daylight < 300 || rng.Float64() < 0.1
				}
				lamps := 0.0
				if lampsOn {
					lamps = 400
				}
				// A PIR sensor misses people who sit still.
				motion := 0
				if people > 0 {
					motion = poisson(rng, 0.4*float64(people))
				} else if rng.Float64() < 0.02 {
					motion = 1 // false trigger
				}
				row[0] = stamp(t)
				row[1] = num(hour)
				row[2] = strconv.Itoa(weekday)
				row[3] = num(temp + rng.NormFloat64()*0.1)
				row[4] = num(clampRange(hum+rng.NormFloat64()*0.5, 15, 90))
				row[5] = num(pressure)
				row[6] = num(math.Max(400, co2+rng.NormFloat64()*15))
				row[7] = num(daylight + lamps + rng.Float64()*5)
				row[8] = strconv.Itoa(motion)
				row[9] = strconv.Itoa(people)
				row[10] = "no"
				if people > 0 {
					row[10] = "yes"
				}
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func airQualityCalibration() Dataset {
	header := []string{"timestamp", "hour", "month", "raw_pm25", "temperature_c", "humidity_pct", "raw_no2", "pm25_reference"}
	return Dataset{
		Path: "regression/air-quality-calibration.csv",
		Description: "Calibration of a low-cost optical particle sensor (e.g. SDS011) against a reference instrument, one row per hour for a year. " +
			"The true PM2.5 concentration follows the seasons (higher in winter), rush hours and occasional pollution episodes. " +
			"The low-cost sensor has a gain error and overestimates strongly at high humidity (up to about five times) because particles absorb water; " +
			"a cheap gas sensor (e.g. MQ-135) adds a correlated, temperature dependent signal. " +
			"The target pm25_reference is the reference value divided by 150 µg/m³ and clipped to [0, 1]. Ignore timestamp.",
		Task:     nn.Regression,
		Rows:     365 * 24,
		Header:   header,
		Target:   "pm25_reference",
		Ignore:   []string{"timestamp"},
		Loadable: true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			episode := 0.0
			gain := 1.3 + rng.NormFloat64()*0.05
			row := make([]string, len(header))
			for i := range n {
				t := epoch.Add(time.Duration(i) * time.Hour)
				hour := float64(t.Hour())
				day := float64(t.YearDay())
				winter := 0.5 + 0.5*math.Cos((day-15)/365*2*math.Pi)
				traffic := math.Exp(-sq(hour-8)/4) + 0.8*math.Exp(-sq(hour-18)/5)
				if rng.Float64() < 0.004 {
					episode = 30 + rng.Float64()*60
				}
				episode *= 0.93
				truePM := (6+14*winter+8*traffic)*logNormal(rng, 0.25) + episode
				temp := 10 - 9*math.Cos((day-15)/365*2*math.Pi) + 4*math.Sin((hour-9)/24*2*math.Pi) + rng.NormFloat64()*1.5
				rh := clampRange(70-1.2*(temp-10)+rng.NormFloat64()*12, 20, 99)
				// hygroscopic growth, bounded: the sensor overestimates up to about 5x
				f := math.Min(rh, 95) / 100
				growth := 1 + 0.25*f*f/(1-f)
				raw := truePM*gain*growth + 1.5 + rng.NormFloat64()*2
				no2 := 0.6*truePM + 15*traffic + 0.8*temp + rng.NormFloat64()*4
				row[0] = stamp(t)
				row[1] = num(hour)
				row[2] = strconv.Itoa(int(t.Month()))
				row[3] = num(math.Max(0, raw))
				row[4] = num(temp)
				row[5] = num(rh)
				row[6] = num(math.Max(0, no2))
				row[7] = num(clamp01(truePM / 150))
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func sensorHealth() Dataset {
	states := []string{"ok", "drift", "stuck", "spikes", "dropout", "noisy"}
	header := []string{"t", "mean_c", "std_c", "min_c", "max_c", "slope_c_per_min", "diff_to_redundant_c", "missing_samples", "state"}
	return Dataset{
		Path: "online/sensor-health.csv",
		Description: "Health of an industrial temperature probe (e.g. PT100) in a process with setpoint changes, monitored together with a redundant second probe. " +
			"Each row summarizes one minute of 1 Hz readings: mean, standard deviation, min, max, slope, the difference to the redundant probe and the number of missing samples. " +
			"Faults occur as episodes of varying length: drift (slowly growing offset), stuck (frozen value), spikes (single outliers), dropout (missing samples) and noisy (electrical noise). " +
			"About 75 % of the time the probe is ok. The rows form a stream; read them in order. Ignore t.",
		Task:     nn.Classification,
		Rows:     60_000,
		Header:   header,
		Target:   "state",
		Ignore:   []string{"t"},
		Loadable: true,
		Applied:  true,
		Ordered:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			setpoint, process := 80.0, 80.0
			state, left, driftOffset, stuckValue := 0, 200, 0.0, 0.0
			readings := make([]float64, 60)
			row := make([]string, len(header))
			for t := range n {
				if left == 0 {
					if state != 0 {
						state, left = 0, 50+rng.IntN(300)
					} else {
						state, left = 1+rng.IntN(len(states)-1), 20+rng.IntN(100)
						driftOffset, stuckValue = 0, process
					}
				}
				left--
				if rng.Float64() < 0.003 {
					setpoint = 60 + rng.Float64()*60
				}
				start := process
				var sum, sq float64
				lo, hi := math.Inf(1), math.Inf(-1)
				missing, valid := 0, 0
				for s := range readings {
					process += 0.02*(setpoint-process) + rng.NormFloat64()*0.03
					v := process + rng.NormFloat64()*0.05
					switch state {
					case 1:
						driftOffset += 0.0015
						v += driftOffset
					case 2:
						v = stuckValue
					case 3:
						if rng.Float64() < 0.05 {
							v += (rng.Float64()*2 - 1) * 25
						}
					case 4:
						if rng.Float64() < 0.3 {
							missing++
							continue
						}
					case 5:
						v += rng.NormFloat64() * 1.2
					}
					readings[s] = v
					valid++
					sum += v
					sq += v * v
					lo, hi = min(lo, v), max(hi, v)
				}
				mean := sum / float64(valid)
				std := math.Sqrt(math.Max(0, sq/float64(valid)-mean*mean))
				redundant := (start+process)/2 + rng.NormFloat64()*0.08
				row[0] = strconv.Itoa(t)
				row[1] = num(mean)
				row[2] = num(std)
				row[3] = num(lo)
				row[4] = num(hi)
				row[5] = num(process - start)
				row[6] = num(mean - redundant)
				row[7] = strconv.Itoa(missing)
				row[8] = states[state]
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func wifiLocalization() Dataset {
	// office floor of 40 m x 20 m with 4 x 2 rooms of 10 m x 10 m
	aps := [][2]float64{{5, 5}, {20, 2}, {35, 5}, {5, 15}, {22, 18}, {35, 15}}
	header := []string{"device", "ap1_dbm", "ap2_dbm", "ap3_dbm", "ap4_dbm", "ap5_dbm", "ap6_dbm", "room"}
	return Dataset{
		Path: "classification/wifi-localization.csv",
		Description: "Indoor localization from Wi-Fi signal strength. An office floor of 40 m x 20 m has eight rooms (R1–R8, a 4 x 2 grid) and six access points. " +
			"The received signal strength follows a log-distance path loss model (-35 dBm at 1 m, exponent 2.8), loses 6 dB per wall crossed, " +
			"varies with log-normal shadowing (4 dB) and differs by up to ±4 dB between the five device models. Signals below -95 dBm are reported as -100. " +
			"The device column must be ignored.",
		Task:     nn.Classification,
		Rows:     20_000,
		Header:   header,
		Target:   "room",
		Ignore:   []string{"device"},
		Loadable: true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			offsets := make([]float64, 5)
			for i := range offsets {
				offsets[i] = (rng.Float64()*2 - 1) * 4
			}
			row := make([]string, len(header))
			for range n {
				rx, ry := rng.IntN(4), rng.IntN(2)
				x, y := float64(rx)*10+0.5+rng.Float64()*9, float64(ry)*10+0.5+rng.Float64()*9
				dev := rng.IntN(len(offsets))
				row[0] = "D" + strconv.Itoa(dev+1)
				for i, ap := range aps {
					d := math.Max(1, math.Hypot(x-ap[0], y-ap[1]))
					walls := math.Abs(math.Floor(x/10)-math.Floor(ap[0]/10)) + math.Abs(math.Floor(y/10)-math.Floor(ap[1]/10))
					rssi := -35 - 28*math.Log10(d) - 6*walls + rng.NormFloat64()*4 + offsets[dev]
					if rssi < -95 {
						rssi = -100
					}
					row[1+i] = num(math.Round(rssi))
				}
				row[7] = "R" + strconv.Itoa(ry*4+rx+1)
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func smartMeter() Dataset {
	header := []string{"timestamp", "hour", "weekday", "month", "temperature_c", "load_lag_24h_kw", "load_lag_168h_kw", "anomaly", "load_kw"}
	return Dataset{
		Path: "regression/smart-meter.csv",
		Description: "Hourly electricity consumption of a household with an electric heat pump, as read from a smart meter, for two years. " +
			"Consumption has a base load, morning and evening peaks, more use on weekends, extra heating below 15 °C, random appliance runs and holidays away. " +
			"Inputs are the calendar, the outdoor temperature and the consumption 24 and 168 hours earlier (forecasting the next hour). " +
			"In about 1 % of the hours a fault or tampering multiplies or zeroes the reading; these are flagged in anomaly (1) for anomaly detection. " +
			"The target load_kw is the consumption in kW divided by 8 and clipped to [0, 1]. Ignore timestamp and anomaly when forecasting.",
		Task:     nn.Regression,
		Rows:     2 * 365 * 24,
		Header:   header,
		Target:   "load_kw",
		Ignore:   []string{"timestamp", "anomaly"},
		Loadable: true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			const warmup = 168
			history := make([]float64, 0, n+warmup)
			away := 0
			row := make([]string, len(header))
			for i := range n + warmup {
				t := epoch.Add(time.Duration(i-warmup) * time.Hour)
				hour := float64(t.Hour())
				weekend := t.Weekday() == time.Saturday || t.Weekday() == time.Sunday
				day := float64(t.YearDay())
				temp := 10 - 10*math.Cos((day-15)/365*2*math.Pi) + 4*math.Sin((hour-9)/24*2*math.Pi) + rng.NormFloat64()*2
				if away == 0 && rng.Float64() < 0.0015 {
					away = 72 + rng.IntN(240)
				}
				load := 0.25
				if away > 0 {
					away--
				} else {
					load += 0.9*math.Exp(-sq(hour-7)/2) + 1.4*math.Exp(-sq(hour-19)/4)
					if weekend {
						load += 0.4 * math.Exp(-sq(hour-13)/8)
					}
					if rng.Float64() < 0.04 {
						load += 1 + rng.Float64()*2 // washing machine, oven, ...
					}
				}
				if temp < 15 {
					load += 0.12 * (15 - temp)
				}
				load *= logNormal(rng, 0.1)
				anomaly := 0
				if rng.Float64() < 0.01 {
					anomaly = 1
					if rng.Float64() < 0.5 {
						load *= 3
					} else {
						load = 0
					}
				}
				history = append(history, load)
				if i < warmup {
					continue
				}
				row[0] = stamp(t)
				row[1] = num(hour)
				row[2] = strconv.Itoa(int(t.Weekday()))
				row[3] = strconv.Itoa(int(t.Month()))
				row[4] = num(temp)
				row[5] = num(history[i-24])
				row[6] = num(history[i-168])
				row[7] = strconv.Itoa(anomaly)
				row[8] = num(clamp01(load / 8))
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func sq(x float64) float64 { return x * x }

func clampRange(x, lo, hi float64) float64 { return min(max(x, lo), hi) }

// poisson draws from a Poisson distribution (Knuth's method, fine for small means).
func poisson(rng *rand.Rand, mean float64) int {
	l, k, p := math.Exp(-mean), 0, 1.0
	for {
		p *= rng.Float64()
		if p <= l {
			return k
		}
		k++
	}
}
