.PHONY: all test race vet lint bench cover

all: vet test

test:
	go test ./...

race:
	go test -race ./...

vet:
	go vet ./...

lint:
	golangci-lint run

bench:
	go test -run=^$$ -bench=. -benchmem ./nn/

cover:
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out
