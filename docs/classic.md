# Classic models

Not every problem needs a neural network. The classic models train in milliseconds, have no hyperparameters worth tuning, give the same result on every run and are easy to explain. Use them as the **baseline** every project should start with, and keep them when they are good enough.

| Package | Model | Use it for |
| --- | --- | --- |
| [`linear`](../linear) | linear regression (with ridge) | predicting numbers; interpretable coefficients |
| [`linear`](../linear) | logistic regression (multinomial) | classification; calibrated probabilities |
| [`cluster`](../cluster) | k-means | finding groups in unlabeled data |
| [`pca`](../pca) | principal component analysis | compressing correlated features, visualizing data |

All of them are pure Go, deterministic, safe for concurrent prediction, serialize to JSON (`persist.SaveRegression`, `SaveLogistic`, `SaveKMeans`, `SavePCA`) and validate what they load.

## Linear regression

```go
reg, err := linear.FitRegression(set, linear.RegressionConfig{L2: 0.1})
price, err := reg.Predict(x)
fmt.Println(reg.Features, reg.Weights, reg.Bias)
```

- Solves the least-squares problem **exactly** with a Cholesky decomposition of the normal equations; there is no learning rate and no epoch count.
- **Coefficients in original units:** the features are standardized internally, but `Weights[j]` is the change of the prediction per unit of feature `j`, whatever its scale. On synthetic data generated from `3·a − 2·b + 0.005·c + 7` with `c` in the thousands, the fit recovers the coefficients to nine digits.
- `L2` (ridge) shrinks the standardized weights. It helps with correlated features and many features relative to samples; 0 gives plain least squares. Duplicated features without a penalty still work: they share the weight.
- `reg.Evaluate(set, threshold)` returns the usual evaluation with MAE, RMSE and R².

## Logistic regression

```go
clf, err := linear.FitLogistic(set, linear.LogisticConfig{})
label, err := clf.Classify(x)
p, err := clf.Probabilities(x) // one probability per class, summing to 1
```

- Multinomial (softmax) logistic regression for any number of classes; with two classes it is the classic logistic regression.
- Minimizes the cross-entropy with L-BFGS. The problem is convex, so the result does not depend on a random start or a learning rate.
- `L2` defaults to 1e-4, which keeps the weights finite when the classes are separable; larger values regularize more, a negative value turns the penalty off.
- `Weights` holds one row per class in original units. With two classes, the difference of the two rows is the classic logit coefficient: on data generated with logit `0.002·x1 − 1.5·x2`, the fit gives 0.0020 and −1.41.
- `ProbabilitiesInto(dst, x)` predicts without allocating (90 ns for 20 features and 4 classes).

## k-means

```go
m, labels, err := cluster.Fit(points, cluster.Config{K: 4})
group, err := m.Predict(newPoint)
score, err := cluster.Silhouette(points, labels)
```

- k-means++ starting centroids and Lloyd iterations until no point changes its cluster; `Restarts` (default 10) runs keep the best result (lowest `Inertia`).
- **Choose K** by comparing `cluster.Silhouette` for several values (higher is better; it is O(n²), so use a sample of a few thousand points). On three synthetic groups, K = 3 scores 0.87 and K = 2, 4, 5 score 0.70, 0.67, 0.48.
- k-means uses Euclidean distances: **standardize** features on different scales first (`learn.FitScaler`).
- Empty clusters are refilled with the farthest point, so `Fit` always returns K clusters.

## PCA

```go
p, err := pca.Fit(points, pca.Config{MinExplained: 0.95, Standardize: true})
fmt.Println(p.ExplainedRatio())
z, err := p.Transform(x)   // coordinates along the components
back, err := p.Inverse(z)  // reconstruction
reduced, err := p.TransformSet(set)
```

- Finds the orthogonal directions of largest variance from the eigen decomposition of the covariance matrix (Householder tridiagonalization and implicit QL, O(d³)).
- Keep a fixed number of `Components` or the fewest that explain `MinExplained` of the variance.
- `Standardize` gives every feature the same weight; without it, features on large scales dominate. `Whiten` scales the coordinates to unit variance.
- `TransformInto(dst, x)` projects without allocating.

On the synthetic `classification/high-dim` data (200 features: 10 informative, 20 noisy copies of them, 170 noise), the first 10 components stand out clearly, and a logistic regression on **20 components reaches 94.8 % accuracy, the same as on all 200 features (94.9 %)**, with a tenth of the inputs.

## Speed

Apple M1, 8 cores:

| Operation | Data | Time |
| --- | --- | ---: |
| `linear.FitRegression` | 10,000 samples × 3 features | 1.3 ms |
| `linear.FitLogistic` | 10,000 samples × 20 features, 4 classes | 13 ms |
| `Logistic.ProbabilitiesInto` | 20 features, 4 classes | 90 ns, no allocation |
| `cluster.Fit` (10 restarts) | 10,000 points × 10 features, K = 8 | 17 ms |
| `KMeans.Predict` | 10 features, K = 8 | 45 ns, no allocation |
| `pca.Fit` | 10,000 points × 100 features | 16 ms |
| `pca.Fit` | 2,000 points × 784 features | 1.3 s |
| `PCA.TransformInto` | 100 features, 10 components | 1 µs, no allocation |

Large inputs are processed on all cores, split into fixed chunks that are combined in order, so the results are bit for bit the same on any machine.

## Baseline first

The quality tests compare the classic models with networks on the synthetic data:

| Data | Classic model | Result | Network |
| --- | --- | ---: | ---: |
| `classification/blobs` (4 Gaussian clusters) | logistic regression | 100 % accuracy | – |
| `classification/imbalanced` | logistic regression | macro AUC 0.992 | – |
| `regression/friedman1` (nonlinear) | linear regression | R² 0.72 | R² 0.94 |
| `classification/blobs` without labels | k-means | 100 % purity | – |

When the classes or the relation are (nearly) linear, the classic model is as good as a network and much cheaper. When it is clearly nonlinear, as in Friedman #1, the network wins, and the baseline shows by how much.
