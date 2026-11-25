# Contributing to gopher-learn

Thanks for your interest in gopher-learn! Contributions of all sizes are welcome: bug reports, documentation fixes, tests and new features.

## Scope

gopher-learn aims to be **the simplest pure-Go library for tabular data and online learning — no cgo, embeddable in any Go service.** Before working on a larger feature, check the [roadmap](ROADMAP.md) and open an issue to discuss it. Features outside this scope (for example deep-learning layers) will likely be declined.

Issues labeled [`good first issue`](https://github.com/breskos/gopher-learn/labels/good%20first%20issue) are a good place to start.

## Ground rules

- **Standard library only.** The library must not add dependencies or require cgo.
- **Errors, not panics.** Library code returns errors and never calls `log.Fatal`, `os.Exit` or `panic` on bad input.
- **Reproducible.** Anything random takes a seed or a `*rand.Rand`.
- **Tested.** New code comes with tests. CI fails if total coverage drops below 90 %.
- **Documented.** Exported identifiers have doc comments. User-visible changes go into [CHANGELOG.md](CHANGELOG.md) under "Unreleased".

## Development

Requires Go 1.24 or newer and, for linting, [golangci-lint](https://golangci-lint.run/).

```
make vet     # go vet
make test    # all tests
make race    # tests with the race detector
make lint    # golangci-lint
make cover   # coverage report
make bench   # benchmarks of the nn package
```

The loaders are fuzz tested. To fuzz locally:

```
go test -run='^$' -fuzz='^FuzzReadCSV$' -fuzztime=30s ./learn/
```

Quality tests train real models on synthetic data and take a few seconds; `go test -short ./...` skips them. If a change lowers the measured quality, the test fails with the value it reached.

Synthetic data sets with known structure come from `internal/synth`. To write them as files, for example for benchmarks:

```
go run ./internal/cmd/gendata -out ../gopher-learn-data/synthetic
```

Example programs live in [examples](examples); their data is in [gopher-learn-data](https://github.com/breskos/gopher-learn-data).

## Pull requests

1. Fork the repository and create a branch from `master`.
2. Keep the change focused: one feature or fix per pull request.
3. Run `make vet test lint` before pushing.
4. Describe what changed and why in the pull request.

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
