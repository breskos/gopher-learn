# N-grams

Classifies German questions by the type of answer they expect (`FACT`, `PARAGRAPH`, `LIST`) using the n-gram string encoder. The encoder is saved and loaded again to show persistence. See [docs/encoders.md](../../docs/encoders.md) for details on encoders.

Data: [gopher-learn-data/ngram](https://github.com/breskos/gopher-learn-data/tree/master/ngram)

```
go run . -data path/to/data.phrase
```
