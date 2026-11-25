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
| `StringHashing` | string | Hashing trick: every token adds ±1 to one of `HashDimensions` dimensions. No vocabulary, so new words in a stream are encoded too |
| `StringTFIDF` | string | Token counts weighted by inverse document frequency, unit length; keeps the `TFIDFMaxTerms` terms occurring in most texts |
| `FloatCategories` | floats | One-hot per dimension for numbers that are category codes (at most `CategoriesMax` per dimension) |

Strings are normalized before encoding: they are lowercased and every run of characters that are not letters or digits becomes a single space. Unicode letters such as umlauts are kept. Unknown strings, tokens, n-grams or category codes encode to zeros.

For streams, `StringHashing` is the safest choice: its size never changes and it needs no refitting when new words appear. `Automatic` never picks the hashing, TF-IDF or category models; choose them with `Scan`.

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
