# Neural Network From Scratch (C++)

This project is a simple and compact machine learning "library" implemented in modern C++.
- A matrix engine for linear algebra and some more operations.
- Fully connected layers with multiple activation functions.
- A lightweight `LinearRegression` model using the same core building blocks.
- Backpropagation training with MSE and Cross-Entropy options in function of the activation functions.
- Full-batch and mini-batch gradient descent.
- Optional per-epoch training metrics logging.
- Built-in metrics utilities (`MSE`, `MAE`, `RMSE`, `R2`, `accuracy`, `cross-entropy`).
- CSV data loading utilities.
- A standardization utility (`StandardScaler`).

> The project is still **in progress**, but due to upcoming classes and exams, development may slow down or be temporarily on hold.

---

## Requirements

- C++17 compiler (for example, `g++`)
- Linux/macOS shell (or equivalent command-line setup)

No external libraries are required.

---

## Build and Run

Build with Makefile:

```bash
make
```

```bash
make run
```

Build with a custom entry point file:

```bash
make MAIN=your_main.cc
```

Useful targets:

- `make debug`
- `make release`
- `make clean`

Your output depends on the model architecture, dataset, and training settings defined in your entry point.

---

## User Guide (Quick Start)

### 1) Load data from CSV

```cpp
auto [X, Y] = DataLoader::loadDataset("dataset.csv");
```

Useful variants:

- Last column as label (default):

```cpp
auto [X, Y] = DataLoader::loadDataset("dataset.csv", -1, 1);
```

- Multi-class labels with one-hot encoding:

```cpp
auto [X, Y] = DataLoader::loadDataset("dataset.csv", -1, numClasses);
```

### 2) (Optional) standardize features

```cpp
StandardScaler scaler;
Matrix XScaled = scaler.fitTransform(X);
```

### 3) Train a neural network

```cpp
NeuralNetwork model;
model.addLayer(2, 16, RELU);
model.addLayer(16, 1, SIGMOID);

int epochs = 5000;
double lr = 0.01;
Loss loss = AUTO;
int batchSize = 32;    // 0 => full-batch
int shuffleSeed = 42;  // -1 => random each run
bool logMetrics = true;
int metricsEvery = 100;

model.train(X, Y, epochs, lr, loss, batchSize, shuffleSeed, logMetrics, metricsEvery);
Matrix pred = model.predict(X);
```

### 4) Train a linear regression model

```cpp
LinearRegression model(2, 1);

int epochs = 3000;
double lr = 0.01;
int batchSize = 32;    // 0 => full-batch
int shuffleSeed = 42;  // -1 => random each run
bool logMetrics = true;
int metricsEvery = 100;

model.train(X, Y, epochs, lr, batchSize, shuffleSeed, logMetrics, metricsEvery);
Matrix pred = model.predict(X);
```

### 5) Compute metrics

```cpp
#include "Metrics.h"

double mse = Metrics::mse(yTrue, yPred);
double mae = Metrics::mae(yTrue, yPred);
double rmse = Metrics::rmse(yTrue, yPred);
double r2 = Metrics::r2Score(yTrue, yPred);
double acc = Metrics::accuracy(yTrue, yPred);
double ce = Metrics::crossEntropy(yTrue, yPred);
```

### 6) Save and load models

```cpp
model.save("model.bin");
model.load("model.bin");
```

### 7) Common parameter rules

- `batchSize = 0` => full-batch training.
- `batchSize > N` => throws an error.
- If `N % batchSize != 0`, the last mini-batch is processed with the remaining samples.
- `shuffleSeed = -1` => non-deterministic shuffling.
- `shuffleSeed >= 0` => deterministic/reproducible shuffling.
- `logMetrics = true` => prints training metrics to stdout.
- `metricsEvery = k` => logs every `k` epochs.

---

## Input File Format

CSV format expected by `DataLoader::loadDataset`:

- First columns: input features (`X`)
- Label column (default: last): target (`Y`)

`DataLoader::loadDataset(path, labelCol, numClasses)` behavior:

- `labelCol = -1` means the last column is used as label.
- If `numClasses > 1`, labels are one-hot encoded.

---

## Project Architecture

```text
neural-network-from-scratch/
├── include/
│   ├── Matrix.h
│   ├── Layer.h
│   ├── Metrics.h
│   ├── NeuralNetwork.h
│   ├── DataLoader.h
│   ├── StandardScaler.h
│   └── LinearRegression.h
├── src/
│   ├── Matrix.cc
│   ├── Layer.cc
│   ├── Metrics.cc
│   ├── NeuralNetwork.cc
│   ├── DataLoader.cc
│   ├── StandardScaler.cc
│   └── LinearRegression.cc
├── Makefile
└── README.md
```

### Main modules

- `Matrix`: matrix operations (`dot`, transpose, element-wise ops, slicing, softmax, one-hot, concatenation, `sum/mean/var/std` reductions, `exp/log`).
- `Layer`: fully connected layer (`W`, `b`) + activation and gradient propagation.
- `NeuralNetwork`: stack of layers, forward pass, backpropagation, training loop, save/load.
- `DataLoader`: CSV parser and dataset split utilities.
- `StandardScaler`: feature standardization (`fit`, `transform`, `inverseTransform`).
- `LinearRegression`: single linear layer model trained with gradient descent.
- `Metrics`: reusable error/score functions for regression and classification.

---

## How the Training Works

### 1) Forward pass

For each layer $L$:

$$
Z^{[L]} = A^{[L-1]} W^{[L]} + b^{[L]}, \quad A^{[L]} = g^{[L]}(Z^{[L]})
$$

Supported activations:

- `RELU`
- `LEAKY_RELU`
- `SIGMOID`
- `TANH`
- `SOFTMAX`
- `NONE`

### 2) Loss selection

`NeuralNetwork::train(..., Loss loss = AUTO)` chooses:

- `CROSS_ENTROPY` when output activation is `SOFTMAX`
- `MSE` otherwise

`train(..., batchSize, shuffleSeed)` behavior:

- `batchSize = 0` uses full-batch training.
- `batchSize > 0` enables mini-batch training.
- `batchSize > N` throws an error.
- Rows are shuffled at each epoch before batching (same permutation for `X` and `Y`).
- Last batch uses the remaining samples when `N` is not divisible by `batchSize`.
- `shuffleSeed = -1` uses non-deterministic randomness.
- `shuffleSeed >= 0` enables deterministic/reproducible shuffling.

Optional logging in both `NeuralNetwork::train` and `LinearRegression::train`:

- `logMetrics = true` enables epoch-level logs.
- `metricsEvery` controls logging frequency.

### 3) Output gradient

- MSE:

$$
\nabla_{\hat{Y}} = \frac{2}{N}(\hat{Y} - Y)
$$

- Softmax + Cross-Entropy canonical form:

$$
\nabla_{Z^{[L]}} = \frac{1}{N}(\hat{Y} - Y)
$$

### 4) Backpropagation and update

Each layer computes gradients for weights, bias and previous activations, then applies gradient descent:

$$
W \leftarrow W - \eta \nabla_W, \quad b \leftarrow b - \eta \nabla_b
$$

where $\eta$ is the learning rate.

---

## Serialization

Serialization in this project is model-specific (there is no single shared format for all models).

For `NeuralNetwork`, `NeuralNetwork::save(path)` stores:

- Number of layers
- Per-layer dimensions
- Activation enum value
- Weight matrix values
- Bias vector values

`NeuralNetwork::load(path)` reconstructs the full model from the same binary format.

`LinearRegression` has its own `save/load` implementation, and future ML models are expected to define their own serialization format as needed.

---

## Minimal Usage Example

```cpp
#include "NeuralNetwork.h"
#include "DataLoader.h"

int main() {
	auto [X, Y] = DataLoader::loadDataset("dataset.csv");

	NeuralNetwork model;
	model.addLayer(2, 8, RELU);
	model.addLayer(8, 1, SIGMOID);

	model.train(X, Y, 10000, 0.03, AUTO, 32, 42);
	Matrix pred = model.predict(X);

	model.save("model.bin");
	return 0;
}
```

---

## Current Limitations

- No advanced optimizers (Adam, RMSProp, momentum).
- No regularization layers (dropout, batch normalization).
- No automatic train/validation splitting.
- Training metrics logging is console-based (there is no structured history object yet).
