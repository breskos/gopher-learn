# Production online learning

The [`service`](../service) package runs a network as a service that answers live requests and keeps learning from what happens afterwards. It is made for the situation of a real system:

- the true label or value arrives **later** than the prediction (a click, a payment, a measurement), sometimes much later, out of order or not at all;
- a label can turn out to be **wrong** and be corrected;
- feedback can be **corrupt** (a broken upstream system, a unit error, an attacker);
- the world **changes** (concept drift);
- the live model must **never** get worse unnoticed, and predictions must **never** wait for training.

[`online`](online.md) is the learning core for a clean stream of labeled samples; `service` puts the guardrails of a production system around it.

## Data flow

```
Predict(x) ──► event ID + live prediction ──► x parked in the ID store
                                                    │
Feedback(id, label) ◄───────── later, any order ────┘
   │
   ├─ outlier filter        implausible feedback is rejected, the event stays pending
   ├─ drift check           error of the live prediction → ADWIN / Page-Hinkley
   ├─ replay buffer         labeled sample stored under its ID
   ├─ online update         new sample + replay mini-batch, clipped, on a shadow copy
   │
   └─ every PublishEvery updates: guardrail on a reference set
          ├─ as good as live ──► atomic swap: the shadow copy becomes the new version
          └─ worse           ──► rollback: the shadow copy is reset to the live version
```

## Example and data

[examples/online-service](../examples/online-service) is a complete program: it trains a model offline on the history of an event log, starts the service, replays the rest of the log with delayed, missing, wrong and corrupt feedback, corrections and a concept drift, and compares the result with the frozen offline model. With `-serve :8080` it offers the service as an HTTP API.

The event logs are [service/offers.csv](https://github.com/breskos/gopher-learn-data/tree/main/synthetic/service) (will a customer accept an offer?) and service/deliveries.csv (delivery time in minutes) of gopher-learn-data; the example also generates them in memory. Every row is a request with its features, the true outcome, when its feedback arrives (`feedback_delay`), what the feedback says (`feedback`, sometimes wrong or corrupt) and when a correction follows (`correction_delay`).

| Event log | After the drift | Learning service | Frozen offline model |
| --- | --- | ---: | ---: |
| offers | accuracy | 89.9 % | 63.7 % |
| deliveries | mean absolute error | 4.2 min | 13.5 min |

## Quick start

```go
cfg := service.DefaultConfig()
cfg.Reference = referenceSet // labeled samples the live model must keep doing well on
s, err := service.New(trainedNet, nn.Classification, cfg)

// in the request handler
p, err := s.Predict(features)
respond(p.Label, p.ID)

// when the outcome is known, e.g. in a webhook or a queue consumer
u, err := s.Feedback(p.ID, "bought")
```

For regression, use `s.FeedbackValue(id, value)`. With `Scaler` (and `TargetScaler` for regression), the service takes raw features and values and answers in raw units; use the scalers the network was trained with. `Inputs`, `Values` and `Reference` are raw too. `PredictID(id, x)` uses your own ID, e.g. a request or order ID, instead of a generated one. Everything is safe for concurrent use.

## The requirements and how they are met

### FR-01 ID-based delayed feedback

`Predict` returns a `Prediction` with a unique `ID` and parks a copy of the input vector together with the live output. `Feedback(id, …)` finds it, builds the labeled sample and trains on it.

- **TTL and capacity:** a prediction waits at most `PendingTTL` (default 24 h). When `PendingSize` (default 100,000) predictions are waiting, the oldest is dropped. Feedback for an unknown, expired or dropped ID returns `ErrUnknownEvent`. `Stats` counts `Expired` and `Evicted`.
- **Memory:** one pending prediction holds the input vector and the output, about 8·(inputs + outputs) bytes plus about 150 bytes of bookkeeping. 100,000 predictions with 20 inputs take about 35 MB; lower `PendingSize` on small devices.
- **Order:** feedback can arrive in any order. The test `TestDelayedFeedback` delays it by up to 50 events and shuffles it; the network still goes from 96.6 % to 100 %.
- Generated IDs are a random per-service prefix plus a counter, so IDs of a restarted service do not collide with old ones.

### FR-02 Later correction

Feedback for an event that already has one is a **correction**: the stored sample in the replay buffer gets the new label or value and is trained on again (`Update.Corrected`). The error of a correction is not fed to the drift detector, because the live prediction was already scored once. A correction is possible as long as the sample is in the replay buffer (`ReplaySize`, default 5000 samples) or the holdout set; size the buffer for the latest corrections you expect (the example keeps 20,000 events, about five days). After a drift, the buffer forgets the old concept, and corrections for those events return `ErrUnknownEvent`.

A rejected feedback (see NFR-01) leaves the event pending, so a corrected feedback can still arrive as the first one.

`TestCorrectionsRepair` sends 2000 events with 40 % wrong labels and then corrects all 800 of them; the accuracy goes from 98.9 % to 99.7 %.

### FR-03 Experience replay

Every labeled sample is stored in a ring buffer of `ReplaySize` samples, addressed by its event ID. Each update trains the new sample **together with `ReplayBatch` (default 15) random samples from the buffer** in one mini-batch. This keeps what older data taught when the recent feedback comes from only one part of the input space.

`TestReplayPreventsForgetting` trains on two input regions and then for 3000 events only on one of them. Accuracy on the region that no longer appears:

| | Accuracy on the old region |
| --- | ---: |
| without replay (`ReplayBatch: -1`) | 75.1 % |
| with replay (default) | 98.7 % |

### FR-04 Concept drift detection

The error of every live prediction (0/1 for classification, absolute error for regression) feeds a drift detector, ADWIN by default (`Drift: online.PageHinkley` or `online.NoDrift` are the alternatives). Only a rising error counts. On a drift:

1. **Warning:** `OnDrift` is called with a `DriftEvent` and a warning is logged; `Update.Drift` is set. Further detections within `DriftCooldown` (200) feedbacks are not reported again.
2. **Learning rate:** it is multiplied by `DriftBoost` (default 3) and fades back to the base rate over `DriftBoostUpdates` (200) updates. `Update.LearningRate` and `Stats.LearningRate` show the current value.
3. **Forgetting:** the replay buffer keeps only the samples ADWIN attributes to the new concept, so replay does not train the old concept back in.

`TestDriftBoostRecovers` flips the concept after 2000 events: with the boost, the live model is back at 90 % after 100 feedbacks, without it after 150.

### FR-05 Exploration vs. exploitation

With `Epsilon` > 0 (classification), a prediction returns a random **other** class with that probability instead of the best one (ε-greedy). `Prediction.Explored` marks it, and `Prediction.Output` still holds the scores of the model. Use it when you only learn the outcome of the action you took, e.g. which offer was shown: without exploration, the system never learns whether the other actions would have been better. The drift detector and the statistics always use the model's best class, so exploration does not look like errors.

### NFR-01 Outlier filter and plausibility

Before feedback trains, it passes these checks (rejected: `ErrOutlier`, counted in `Stats.Rejected`, nothing is trained, the event stays pending):

- **Finite values:** NaN and infinite features or targets are always rejected.
- **Ranges:** `Inputs` bounds every feature (checked already in `Predict`), `Values` bounds regression targets.
- **Robust z-score** (regression): a value further than `OutlierZ` (default 6) robust standard deviations from the recent values is rejected, after `OutlierWarmup` (50) values. Location and scale follow the last ~200 values, and every value, also a rejected one, enters them **clamped to the bound**. A single spike therefore barely moves the statistics, but a lasting level shift is accepted after a while: in `TestOutlierFilter`, corrupt values of 1e6, −5e4 and 400 leave the model bit for bit unchanged, while a real shift of the target by 30 (13 standard deviations) is accepted after 19 rejections.
- **Custom check:** `Check func(*learn.Sample) error` for domain rules.

Regression targets do not have to be scaled to [0, 1]: use a linear output layer (`nn.Linear`) and pass the target scaler as `TargetScaler`; the filter then works on raw values, and the drift detector sees standardized errors.

### NFR-02 Gradient clipping

Every update rescales the mean gradient to an L2 norm of at most `ClipNorm` (default 1), so one SGD step changes the weights by at most `LearningRate · ClipNorm` (0.05 by default) in L2 norm, however extreme a sample is. `TestClipping` measures exactly 0.05 for a target of 10⁶; without clipping the same step moves the weights by 1.2·10⁵. Steps with a non-finite gradient are skipped (`Stats.Skipped`).

### NFR-03 Snapshots and auto-rollback

Training changes only a **shadow copy**. Every `PublishEvery` (100) updates, the guardrail decides:

- The shadow copy and the live version are both scored on the reference set with `Metric` (default accuracy, RMSE for regression). If the shadow copy is at most `Tolerance` (2 %, relative) worse, it is published as a new **version**.
- Otherwise it is reset to the live version, the optimizer state is discarded, `OnRollback` is called with both scores and a warning is logged. Weights that are no longer finite always cause a rollback.

The reference set comes from one of:

- `Reference`: a fixed, trusted set. After a real concept change it must be replaced with `SetReference`, or the guardrail keeps the old concept live.
- `Holdout`: a rolling holdout set built from the feedback. Every `HoldoutEvery`-th (10th) feedback is kept aside instead of trained on. Because it follows the stream, it also follows a drift. It protects against a broken training step, not against feedback that is poisoned as a whole.
- Neither: versions are published without a quality check; only non-finite weights are caught.

The last `History` (5) versions are kept. `Rollback()` puts the previous one live by hand, `History()` lists them, `Publish()` runs the guardrail immediately. With `SnapshotDir`, every version is saved atomically as `model-000042.json` (readable with `persist.LoadNetwork`), and only the newest `History` files are kept.

`TestGuardrailRollback` first trains on clean feedback (99.6 %) and then sends 1000 events with inverted labels. The guardrail rolls back 11 times, and the live model stays at 99.6 % on the true concept.

### NFR-04 Decoupled inference and training

The live version is an immutable copy of the network behind an `atomic.Pointer`. `Predict` loads the pointer and predicts without any lock on the network; it only takes a short lock to park the input. Training holds a separate lock and changes only the shadow copy. Publishing clones the shadow copy and swaps the pointer, so a prediction always sees exactly one complete version (`Prediction.Version`). `TestConcurrentPredictions` checks under the race detector that every answer of four predicting goroutines equals the output of the version it names, while feedback trains and publishes.

## Speed

Apple M1, network 2-8-2, default configuration:

| Operation | Time | Allocations |
| --- | ---: | ---: |
| `Predict` (incl. parking and eviction) | 0.40 µs | 5 |
| `Predict`, 8 goroutines | 0.52 µs | 5 |
| `Feedback` (update with 16 samples, guardrail on 200 samples every 100 updates) | 5.7 µs | 11 |

With such a small network, the lock that parks the input dominates, so parallel predictions do not scale; with larger networks the prediction itself dominates and runs in parallel without a lock.

## Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `PendingTTL`, `PendingSize` | 24 h, 100,000 | ID store |
| `ReplaySize`, `ReplayBatch` | 5000, 15 | replay buffer, samples per update (−1: off) |
| `LearningRate`, `Steps`, `Optimizer` | 0.05, 1, SGD | online update |
| `ClipNorm` | 1 | gradient clipping (−1: off) |
| `Drift`, `DriftDelta`, `DriftCooldown` | ADWIN, 0.002, 200 | drift detection |
| `DriftBoost`, `DriftBoostUpdates` | 3, 200 | learning rate after a drift |
| `OnDrift` | – | drift warning |
| `Scaler`, `TargetScaler` | – | raw features and values in and out |
| `Inputs`, `Values`, `Check` | – | plausibility |
| `OutlierZ`, `OutlierWarmup` | 6, 50 | robust z-score of regression values (−1: off) |
| `Reference`, `Holdout`, `HoldoutEvery`, `MinHoldout` | –, 0, 10, 50 | guardrail data |
| `Metric`, `Tolerance`, `Threshold` | accuracy / RMSE, 0.02, 0.05 | guardrail decision |
| `PublishEvery`, `History`, `SnapshotDir` | 100, 5, – | versions |
| `OnPublish`, `OnRollback` | – | callbacks |
| `Epsilon` | 0 | exploration |
| `Seed`, `Now`, `Logger` | – | reproducibility, clock, logging |

Zero values select the defaults, so `service.Config{}` works.

## Limits

- The state of the service (pending predictions, replay buffer, statistics) lives in memory. Snapshots save the published networks, not the pending events; after a restart, feedback for old IDs returns `ErrUnknownEvent`.
- Feedback is supervised: it must be the true label or value. For bandit problems where only a reward for the chosen action is known, map the reward to a label yourself (e.g. "the shown action was right / wrong") and use `Epsilon` to keep exploring.
- One service trains on one goroutine at a time. Feedback is serialized; at about 6 µs per update for small networks, that is well above 100,000 feedbacks per second.
