# Models

A `model.Model` takes you from raw records to predictions. It bundles everything that a trained network needs in production:

- the **columns** it reads and how each one is turned into features,
- the **encoders** for text columns (categories, tags, short phrases),
- the **scaler** for numeric columns, fitted on the training data,
- for regression, the **target scaler** that maps predictions back to the original units,
- the trained **network**.

All of it is saved and loaded as one file, so the code that serves predictions never has to repeat the preprocessing.

## Train

```go
records, err := model.LoadRecords("machines.csv") // first row names the columns
if err != nil {
	return err
}
m, history, err := model.Train(ctx, records, model.Config{
	Target: "status",
	Ignore: []string{"machine_id"},
})
if err != nil {
	return err
}
log.Printf("trained %d epochs, stopped: %s", len(history.Epochs), history.Stop)

if err := persist.SaveModel("model.json", m); err != nil {
	return err
}
```

`Train` does the following:

1. **Columns:** every column except the target and the ignored ones becomes an input. A column whose values are all numbers is `Numeric`; any other column is `Categorical`. Override this with `Config.Kinds`, for example to treat a numeric postal code as a category.
2. **Split:** `Config.ValidationRatio` (default 0.2) of the records are held out for validation, stratified by class.
3. **Fit the preprocessing on the training part only:** an encoder per categorical column (the `encoders` package picks dictionary, split dictionary or n-grams from the data, or use `Config.Encodings`) and a scaler for the numeric columns (`Config.Scale`, default `learn.Standardize`; `Config.NoScaling` turns it off). Regression targets are standardized as well.
4. **Train** a network with `nn.Fit`: one hidden layer of 32 neurons by default (`Config.Hidden`), Softmax output for classification, Linear output for regression, Adam, and early stopping with a patience of 10 epochs that keeps the weights of the best epoch. `Config.Fit` passes every other `nn.FitConfig` option through.

Set `Config.Task: nn.Regression` to predict a number. `Config.Rand` makes runs reproducible.

## Predict

```go
m, err := persist.LoadModel("model.json")
if err != nil {
	return err
}
p, err := m.Predict(model.Record{"temperature": "91.5", "vibration": "2.1", "vendor": "acme"})
if err != nil {
	return err
}
fmt.Println(p.Label, p.Scores) // classification
fmt.Println(p.Value)           // regression, in the units of the target
```

A `model.Record` maps column names to raw values, exactly as they appear in the CSV file or arrive in a request. Columns the model does not use, including the target, are ignored.

| Input | Result |
| --- | --- |
| numeric column missing or empty | error, `errors.Is(err, model.ErrMissing)`, names the column |
| numeric column not a number, `NaN` or infinite | error, `errors.Is(err, model.ErrInvalid)` |
| categorical value never seen in training | allowed, encodes to zeros |

`m.Encode(record)` returns the feature vector the network sees, which helps when debugging.

## Evaluate

```go
test, _ := model.LoadRecords("test.csv")
eval, err := m.Evaluate(test, 0) // threshold: largest regression error that counts as correct
if err != nil {
	return err
}
_ = eval.WriteSummaries(os.Stdout)
fmt.Printf("macro AUC %.3f\n", eval.MacroAUC())
```

## File format

Models are stored with `persist.SaveModel` / `persist.WriteModel` as a JSON document of kind `model` in format version 2. Version 1 files (networks, data sets, online learners and encoders written by gopher-learn 0.4) stay readable. `persist.LoadModel` validates that encoders, scaler and network fit together and rejects inconsistent files.
