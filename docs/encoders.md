# Encoders

Encoders turn raw input such as strings or large float slices into feature vectors a network can process.
An `Encoder` holds one model per named input dimension, so mixed inputs (for example a text field and a list of measurements) can be encoded with one encoder.

## Workflow

1. Collect sample data of the input.
2. `Scan` decides which model to use. With `encoders.Automatic` the decision is made from the data; otherwise the given type is used.
3. `Fit` learns the model from the sample data. If the dimension was not scanned before, `Fit` scans automatically.
4. `Encode`, `EncodeString` or `EncodeFloats` turn new values into feature vectors.

```go
enc := encoders.New("questions")
in := encoders.NewInput(encoders.String)
for _, q := range questions {
	in.AddString(q)
}
if err := enc.Fit("question", in); err != nil { // automatic scan
	return err
}
_ = enc.Explain(os.Stdout)

vector, err := enc.EncodeString("question", "Was ist ein Haus?")
```

Encoders are saved and loaded with `persist.SaveEncoder` and `persist.LoadEncoder`.

## Models

| Type | Input | Encoding |
| --- | --- | --- |
| `StringDictionary` | string | One-hot over the distinct normalized strings |
| `StringSplitDictionary` | string | Multi-hot over the distinct tokens |
| `StringNGrams` | string | Multi-hot over the most frequent character n-grams |
| `FloatExact` | floats | Passed through unchanged |
| `FloatReducer` | floats | Drops constant dimensions and dimensions highly rank-correlated (Spearman) with an earlier kept one |

Strings are normalized before encoding: they are lowercased and every run of characters that are not letters or digits becomes a single space. Unicode letters such as umlauts are kept. Unknown strings, tokens or n-grams encode to zeros.

## Automatic selection

For float input `Automatic` selects `FloatExact`. For strings the heuristics in `Config` decide:

1. `StringDictionary` if there are at most `DictionaryMaxEntries` distinct strings with at most `DictionaryMaxTokens` tokens each,
2. `StringSplitDictionary` if there are fewer than `SplitDictionaryMaxEntries` distinct tokens,
3. `StringNGrams` if no string is longer than `NGramMaxLength` characters,
4. `DefaultStringEncoder` otherwise.

## Config

`encoders.New` applies `encoders.DefaultConfig()`. Change fields before calling `Scan` and `Fit`:

```go
enc := encoders.New("test")
enc.Config.DictionaryMaxEntries = 300
enc.Config.NGramSize = 4
```
