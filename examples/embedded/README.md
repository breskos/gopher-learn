# Embedded fault detector

Train a neural network on a computer and run it on an embedded device. See [docs/embedded.md](../../docs/embedded.md) for the background and measurements.

- `train` simulates motor sensor readings (temperature, vibration, current, speed), trains a 4 × 16 × 8 × 2 network to detect faults (99.5 % test accuracy), converts it to float32 and writes `motor.bin` (about 1 KB).
- `device` loads `motor.bin` with only the `nn` package and classifies readings from standard input. The prediction allocates no memory.

```bash
go run ./examples/embedded/train -out motor.bin
```

```bash
GOOS=linux GOARCH=arm GOARM=7 go build -trimpath -ldflags="-s -w" -o device ./examples/embedded/device
```

Copy `device` and `motor.bin` to the device and run:

```bash
printf '95,7,9,1500\n60,2,7,1500\n' | ./device -model motor.bin
```

```
fault
ok
```
