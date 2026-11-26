# Online service

The [`service`](../../service) package in a realistic setting: a model is trained offline on the history of an event log and then runs as a live service. Its feedback arrives minutes to days after the prediction, sometimes never, is sometimes wrong or corrupt and gets corrected later, and after 60 % of the stream the world changes.

The example replays the event log in time order (predictions at `t`, feedback at `t + feedback_delay`, corrections at `t + feedback_delay + correction_delay`) and compares the learning service with the frozen offline model.

```
go run . -log offers       # will the customer accept the offer? (classification)
go run . -log deliveries   # how many minutes will the delivery take? (regression)
```

Without `-data` the event log is generated in memory; it is the same data as [service/offers.csv and service/deliveries.csv](https://github.com/breskos/gopher-learn-data/tree/main/synthetic/service) of gopher-learn-data:

```
go run . -log deliveries -data path/to/gopher-learn-data/synthetic/service/deliveries.csv
```

## Results

| Event log | After the drift | Learning service | Frozen offline model |
| --- | --- | ---: | ---: |
| offers (40,000 events) | accuracy | 89.9 % | 63.7 % |
| deliveries (40,000 events) | mean absolute error | 4.2 min | 13.5 min |

Before the drift both are equally good; the service then notices the drift, raises its learning rate and is back at its old quality within a few hundred feedbacks, while the frozen model stays wrong. On the deliveries, the outlier filter rejects 394 corrupt values (seconds instead of minutes, typos); most of them are later corrected and then trained on.

```
  events   hours concept  live MAE   frozen MAE version drifts rollbacks rejected corrections pending
   16000    69.3       1  2.463      2.498          128      0        10      185           4     730
   19200    82.9       2  11.437     13.420         134      3        32      226           6     798
   22400    96.7       2  2.518      13.374         152      3        42      272           7     923
```

## What the example shows

- `main.go`: training on the history (cleaned: only plausible values, corrected labels), scalers for inputs and targets, the service configuration (rolling holdout set as guardrail, replay buffer large enough for corrections that come days later) and the replay.
- `server.go`: the same service as an HTTP API:

```
go run . -log offers -serve :8080

curl -X POST localhost:8080/predict  -d '{"id": "order-17", "features": [1.1, 3, 42.5, 19, 10]}'
curl -X POST localhost:8080/feedback -d '{"id": "order-17", "label": "no"}'
curl localhost:8080/stats
```

Feature order: offers `price_ratio, visits, basket_eur, hour, discount_pct`; deliveries `distance_km, items, traffic, hour, rain` (feedback with `"value"`). Implausible feedback answers 422, unknown or expired IDs 404, a duplicate ID 409.

See [docs/service.md](../../docs/service.md) for every option.
