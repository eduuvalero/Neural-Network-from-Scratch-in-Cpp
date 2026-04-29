# Neural Network From Scratch (C++)

This project is a simple and compact machine learning library implemented in C++. The codebase is designed to be lightweight and easy to navigate, focusing on clear implementations rather than complex abstractions.

The aim of the project is for me to learn and understand the basics of AI and machine learning from the ground up. By coding these algorithms manually, I am learning the mathematical foundations and the internal logic that drives neural networks.

For now, the project includes:
- A matrix engine for linear algebra and some more operations.
- A multiple layer perceptron (MLP) neural netowrk totally customizable allowing the creation of multiple layers and the use of the activation functions: `ReLU`, `Leaky ReLU`,`Sigmoid`, `Tanh`, `Sigmoid`, `Softmax`.
- A lightweight `LinearRegression` model using the same core building blocks.
- Backpropagation training with MSE and Cross-Entropy options in function of the activation functions.
- Pluggable optimizers (`SGD`, `Adam`) with full-batch and mini-batch training.
- Dropout regularization during training.
- Optional per-epoch training metrics logging.
- Built-in metrics utilities (`MSE`, `MAE`, `RMSE`, `R2`, `acppuracy`, `cross-entropy`).
- Versioned model serialization (dropout + optimizer state).
- Checkpoint saving during training (.ckpy + .model).
- CSV data loading utilities.
- A standardization utility (`StandardScaler`).
> The project is still **in progress**, but due to upcoming classes and exams, development may slow down or be temporarily on hold.

---

## Requirements

- C++17 compiler (for example, `g++`)

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
make MAIN=your_main.cpp
```

```bash
make MAIN=your_main.cpp run
```

Useful targets:

- `make debug`
- `make release`
- `make clean`

Your output depends on the model architecture, dataset, and training settings defined in your entry point.

## Input File Format

CSV format expected by `DataLoader::loadDataset`:
> It espects a rectangular or square matrix full of numerical values
- Firs  columns as (default: input features (`X`)
- Label column (default: last): target (`Y`)
> But the label column can be the first, can be in the middle and it could even have more than one column if labels are one-hot encoded.

`DataLoader::loadDataset(path, labelCol, numClasses)` behavior in a $m \cdot n$ dataset:

- labelCol = $x$ means the label is in the column $[x]$.
- labelCol = $-x$ means the label is in the column $[n - x]$.
- If `numClasses > 1`, labels are one-hot encoded.

---

## Project Architecture

```text
Neural-Network-from-Scratch-in-Cpp/
├── include/
│   ├── Matrix.h
│   ├── Layer.h
│   ├── Optimizer.h
│   ├── Metrics.h
│   ├── NeuralNetwork.h
│   ├── DataLoader.h
│   ├── StandardScaler.h
│   └── LinearRegression.h
├── src/
│   ├── utils/
│   │   ├── Random.cpp
│   │   └── TrainingUtils.cpp
│   ├── Matrix.cpp
│   ├── Layer.cpp
│   ├── Optimizer.cpp
│   ├── Metrics.cpp
│   ├── NeuralNetwork.cpp
│   ├── DataLoader.cpp
│   ├── StandardScaler.cpp
│   └── LinearRegression.cpp
├── Makefile
└── README.md
```

### Main modules

- `Matrix`: matrix operations (`dot`, transpose, element-wise ops, slicing, softmax, one-hot, concatenation, `sum/mean/var/std` reductions, `exp/log`).
- `Layer`: fully connected layer (`W`, `b`) + activation, dropout, and gradient propagation.
- `Optimizer`: optimizer interface with `SGD` and `Adam` implementations.
- `NeuralNetwork`: stack of layers, forward pass, backpropagation, training loop, save/load.
- `Dense`: simple layer descriptor for `NeuralNetwork::add(...)` in Sequential-style definitions.
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

`NeuralNetwork::compile(..., Loss loss = AUTO_LOSS)` and `NeuralNetwork::train(..., Loss loss = AUTO_LOSS)` choose:

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

`fit(X, Y, epochs)` uses the parameters previously configured with `compile(...)`.

Optional logging in both `NeuralNetwork::compile/fit` and `LinearRegression::compile/fit`:

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

### 4) Backpropagation and optimizer update

Each layer computes gradients for weights, bias and previous activations, then the selected optimizer applies the parameter update.
For `SGD` the update is:

$$
W \leftarrow W - \eta \nabla_W, \quad b \leftarrow b - \eta \nabla_b
$$

where $\eta$ is the learning rate. `Adam` applies adaptive per-parameter scaling with bias-corrected first and second moments.

### 5) Dropout (training only)

When a layer uses dropout, activations are randomly zeroed during training using inverted dropout:

$$
	ilde{A} = A \odot M, \quad M_{ij} \in \{0, \tfrac{1}{1-p}\}
$$

where $p$ is the dropout rate. During inference, dropout is disabled.

---

## Serialization

Serialization in this project is model-specific (there is no single shared format for all models).

For `NeuralNetwork`, `NeuralNetwork::save(path)` stores:

- The number of layers
- For each layer:
  - The number of inputs, outputs, the activation function and the inicialization method
  - All its weights
  - All its biases 
  
|	                   |Layer 1			   |*Weights 1* |*Biases 1*|$...$|*Layer N*  |*Weights N*|*Biases N*|
|:--------------------:|:-----------------:|:----------:|:--------:|:---:|:---------:|:--:|:--:|
|***Nº* of layers**    |*Nº* Inputs 	   |$W_1$	    | $b_1$	   ||*Nº* Inputs|$W_1$	    | $b_1$	   |
|				       |*Nº* Outputs	   |$W_2$	    | $b_2$	   ||*Nº* Outputs        |$W_2$	    | $b_2$	   |
|				   	   |Activation function|$W_3$	    | $b_3$	   ||Activation function|$W_3$	    | $b_3$	   |
|				       |Inicialization     |$...$		| $...$	   ||Inicialization     |$...$	    | $...$	   |
| | |$...$|$B_{outputs}$| | |$...$|	$B_{outputs}$ |	
| | | $W_{inputs * outputs}$ | | | | $W_{inputs * outputs}$



`NeuralNetwork::load(path)` reconstructs the full model from the same binary format.

---

For `linearRegression`, `LinearRegression::save(path)` stores:

- The number of inputs
- All its weights
- Its bias 
  
|	  		    |*Weights*   |			|
|:-------------:|:----------:|:--------:|
|***Nº* Inputs**|$W_1$	     | **Bias** |
| 			    |$W_2$	     |          |
|     		    |$...$	     |  	    |
|               |$W_{inputs}$|



`LinearRegression::load(path)` reconstructs the full model from the same binary format.

> Future ML models are expected to define their own serialization format as needed.

---

## Minimal Usage Example

```cpp
#include "NeuralNetwork.h"
#include "DataLoader.h"

int main() {
	auto [X, Y] = DataLoader::loadDataset("dataset.csv");

	NeuralNetwork model;
	model.input(X.getCols())
	     .add(Dense(8, RELU, HE))
	     .add(Dense(1, SIGMOID, XAVIER));

	model.compile(AUTO_LOSS, 0.03, 32, 42, false, 1);
	model.fit(X, Y, 10000);
	Matrix pred = model.predict(X);

	model.save("model.bin");
	return 0;
}
```

---

## Current Limitations

- No optimizers beyond SGD/Adam (e.g., RMSProp, momentum).
- No regularization layers beyond dropout (e.g., batch normalization).
- No automatic train/validation splitting.
- Training metrics logging is console-based (there is no structured history object yet).
