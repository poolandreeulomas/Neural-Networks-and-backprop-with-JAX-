# 389.204 MLA - Exercise 3 - Neural Networks, Backprop

**Date:** 15. May 2026

---

# Problem 3.1 (25%)

Consider the simple two layer neural network with three neurons in total depicted in Figure 3.1.1. Here, the input consists of the vector:

$$
x = [x_0, x_1]^T
]

while the output is given by the scalar:

$$
\hat{y}
]

For each neuron (j) in layer (i), the activation is computed according to the following affine transformation:

$$
a_j^{(i)} = w_j^{(i)} z^{(i-1)} + b_j^{(i)}
\tag{3.1.1}
]

before the activation function:

$$
\sigma_j^{(i)}(\cdot)
]

is applied to form the final output:

[
z_j^{(i)} = \sigma_j^{(i)}(a_j^{(i)})
]

In the following, we will assume linear or relu activation functions. Hence:

[
\sigma_j^{(i)}(a_j^{(i)})
]

is either:

[
a_j^{(i)}
]

or:

[
\max(0, a_j^{(i)})
]

---

## 3.1.1

Find a configuration of the network in Figure 3.1.1 that approximates a logical XOR without any error. I.e., manually select activation functions (relu or linear) and weights and biases for which the input output relation mimics a logical XOR.

Report your configuration and fill out Table 3.1.1, specifying the respective activations (a) and outputs (z) of each neuron.

**Hint:** Note, that the configuration is not unique.

### Table 3.1.1: Network output and activations

| (x^\top) | (y) | (a_0^{(0)}) | (z_0^{(0)}) | (a_1^{(0)}) | (z_1^{(0)}) | (a_0^{(1)}) | (\hat{y} = z_0^{(1)}) |
| -------- | --- | ----------- | ----------- | ----------- | ----------- | ----------- | --------------------- |
| [0,0]    | 0   |             |             |             |             |             |                       |
| [0,1]    | 1   |             |             |             |             |             |                       |
| [1,0]    | 1   |             |             |             |             |             |                       |
| [1,1]    | 0   |             |             |             |             |             |                       |

---

## 3.1.2

Use the chain rule to find analytical expressions for the derivatives of the loss:

[
L(w,b) = (y - \hat{y})^2
]

with respect to:

[
w_{0,0}^{(0)},; w_{0,1}^{(0)},; b_0^{(0)}
]

and:

[
w_{0,0}^{(1)},; w_{0,1}^{(1)},; b_0^{(1)}
]

for an arbitrary choice of (y) and (x).

Assume that each neuron has a linear activation function, but highlight the factors of the gradient that would change for a nonlinear one.

Briefly describe the flow of gradient information throughout the network during optimization — why can backpropagation be implemented efficiently?

---

## 3.1.3

Use JAX to implement the neural network from Figure 3.1.1 from scratch.

For that, adapt the file:

```text
problem_skeleton_logical_and.py
```

provided on TUWEL.

Use a sigmoid activation function for the second layer and train the network to approximate a logical AND.

Finally, extract the obtained weights and biases from the trained network and again fill out Table 3.1.1.

**Hint:** You are free in your choice of the loss function, but you should use automatic differentiation to compute gradients (e.g. `jax.grad`). Note, that it might be helpful to increase the learning rate.

---

# Problem 3.2 (25%)

Learning curves are particularly helpful to analyse the training process of a neural network.

In the following, you will implement the networks from Figures 3.2.1 and 3.2.2 using JAX and Flax, and evaluate the obtained learning curves.

Note that both networks consist of three densely connected hidden layers and an output layer of dimension 3.

The model in Figure 3.2.2 differs from the one in Figure 3.2.1 by the inclusion of dropout layers.

## Implementation guideline (mandatory)

Use the provided Flax NNX skeleton file:

```text
problem_skeleton_flax_keras_like.py
```

(available on TUWEL) and structure your solution using the following components:

* `DataConfig` (Python dataclass): stores all data-related settings (e.g. random seed, batch size, test split, shuffling, whether to drop the last incomplete batch, and an optional test batch size).

* `Config` (Python dataclass): stores training and model hyperparameters (e.g. epochs, learning rate, logging/evaluation frequency, layer sizes, dropout rate) and contains a `data` field of type `DataConfig`.

* `DataModule` class: provides `train_iter()` and `test_iter()` yielding mini-batches according to `cfg.data`.

* `Network` (`flax.nnx.Module`): defines the forward pass.

* `Model` wrapper: provides `init()`, `train_step()`, `eval_step()`, `fit()`, and `predict()`.

Use `nnx.value_and_grad` for gradient computation and `optax.adam` for optimization.

In Flax NNX, parameters and other model state are stored inside the `nnx.Module` object and are updated through an `nnx.Optimizer` (no explicit `TrainState` is required).

Dropout is controlled via a `train` flag and `deterministic=not train` when calling the module; randomness is provided by `nnx.Rngs` during model construction and is managed automatically by NNX.

---

## 3.2.1

Implement both models from Figures 3.2.1 and 3.2.2 in Flax using relu activation functions for the hidden layers and softmax for the output.

Set the dropout rate to 0.5 for the model in Figure 3.2.2.

Print a model overview using `Module.tabulate(...)` (or implement an equivalent summary yourself).

Explain how the number of trainable parameters is computed for each dense layer, and make explicit why dropout layers do not add trainable parameters.

---

## 3.2.2

Download the file:

```text
classification.csv
```

from TUWEL.

Load the data, transform the targets (y) into a one-hot encoding, and split the dataset into:

[
T_{train}
]

and:

[
T_{test}
]

using a test size of 30%.

Use:

```python
sklearn.model_selection.train_test_split
```

with:

```python
random_state=0
```

Implement a `DataModule` that yields mini-batches of size 64 from:

[
T_{train}
]

and yields the validation data from:

[
T_{test}
]

(either as a single batch or in mini-batches).

---

## 3.2.3

Train each model for 200 epochs and report the accuracy on:

[
T_{train}
]

and:

[
T_{test}
]

Use the Adam optimizer (via `optax.adam`) with default settings, a batch size of 64, and cross-entropy as the loss function (softmax classification).

Which model achieves a higher accuracy on:

[
T_{test}
]

?

---

## 3.2.4

Extend `Model.fit()` to keep track of loss and accuracy on the training dataset:

[
T_{train}
]

and the validation dataset:

[
T_{test}
]

during training.

Repeat the training from Task 3.2.3 and plot learning curves for loss and accuracy (train vs. test).

Briefly describe what you observe and interpret the curves (e.g. overfitting/underfitting, effect of dropout).

### Hint

Inside `fit()`, create a dict `history` with one list per metric (e.g. `loss`, `val_loss`).

Track epoch-wise means of the metrics returned by `train_step` and `evaluate(...)` and append them to `history`.

A learning curve plots epoch vs. the stored values.

---

## 3.2.5

Implement early stopping for the baseline model (Figure 3.2.1) inside `Model.fit()`.

Monitor the validation accuracy on:

[
T_{test}
]

and stop training shortly after its peak.

Use the hyperparameters:

```text
min_delta
```

and:

```text
patience
```

report the configuration you used, and restore the best model parameters (“best state”) before returning from `fit()`.

Plot the new learning curves (loss and accuracy) and compare to Task 3.2.4.

### Hint

At the end of each epoch, after computing the validation accuracy, compare it to the best validation accuracy seen so far.

If `val_acc` improves by at least `min_delta`, update the best score and store a copy of the current state as the best state; in this case, reset the counter of “epochs without improvement” back to 0.

If there is no such improvement, increase the counter by 1.

If the counter reaches `patience`, stop training.

Finally, set:

```python
state = best_state
```

before returning from `fit()`.

`min_delta` is the minimum improvement required to count as progress; `patience` is the number of consecutive epochs without progress you tolerate before stopping.

---

## 3.2.6

Revisit the California Housing Dataset from the first exercise and implement a neural network that achieves a MAE below 0.35 on:

[
T_{test}
]

Split the dataset using:

```python
train_test_split
```

from sklearn with:

```python
test_size=0.2
random_state=0
```

Use MAE as the evaluation metric on:

[
T_{test}
]

Provide your network configuration (layer sizes, activation functions, optimizer settings) and plot learning curves.

### Hint

It might be beneficial to standardize the input data.

Note that transforming the outputs also changes the reported error metrics.

---

# Problem 3.3 (25%)

It is often valuable to provide a measure of confidence together with the prediction of a neural network.

For regression tasks, we can account for data uncertainty by following a probabilistic interpretation of the network output.

The loss function is given by the negative log-likelihood:

[
J_y(\mu_\theta, \sigma_\theta)
= -\log \mathcal{N}(y; \mu_\theta, I\sigma_\theta)
\tag{3.3.1}
]

As sketched above, this can be achieved by estimating the moments of a Gaussian distribution through a neural network with parameterization:

[
\theta
]

The loss function is then given by the negative log-likelihood of the targets:

[
y
]

given the network outputs:

[
\mu_\theta
]

and:

[
\sigma_\theta
]

After training, we can interpret the mean:

[
\mu_\theta^{(i)}
]

as the network’s estimate for input:

[
i
]

while:

[
\sigma_\theta^{(i)}
]

represents the uncertainty associated with that prediction.

## Implementation guideline (mandatory)

Use your modified version (training history) of the provided Flax NNX skeleton file:

```text
problem_skeleton_flax_keras_like.py
```

and structure your solution using the following components:

* `DataConfig`
* `Config`
* `DataModule` class
* `Network` (`flax.nnx.Module`)
* `Model` wrapper

---

## 3.3.1

Expand the loss function:

[
J_y(\mu_\theta, \sigma_\theta)
]

from (3.3.1) and derive an analytical expression for the gradient with respect to a single network output:

[
\mu_\theta^{(i)}
]

and:

[
\sigma_\theta^{(i)}
]

Briefly interpret the obtained expressions — when are the individual gradients zero?

---

## 3.3.2

Download the `data_uncertainty` regression dataset from TUWEL and generate a 2D scatter plot of training and test data, highlighting the respective target values through colors.

Then, fit a neural network with two hidden layers to the dataset using the provided Flax/NNX skeleton.

Use a scalar deterministic output and implement the MSE loss in `loss_and_metrics`.

Provide learning curves (epoch-wise averages) and visualize the predictions as well as the absolute error for individual samples using scatter plots.

### Hint

Make sure to use:

```text
X_train_coords
X_test_coords
```

for visualization, but the standardized, polynomial expansions:

```text
X_train
X_test
```

as model inputs.

---

## 3.3.3

Extend the above model to a probabilistic formulation by adapting the output layer such that it generates:

[
\mu_\theta
]

and:

[
\sigma_\theta
]

Ensure that:

[
\sigma_\theta
]

is mapped to a valid range using a suitable activation function.

Implement:

[
J_y(\mu_\theta, \sigma_\theta)
]

from (3.3.1) as the training loss inside `loss_and_metrics`, using `nnx.value_and_grad` in the training step.

You may reuse ideas from the reference implementation on TUWEL.

---

## 3.3.4

Fit the probabilistic model variant to the `data_uncertainty` dataset.

Provide learning curves that include:

* the loss
* the MSE of the mean prediction (\mu_\theta)
* the average predicted uncertainty (\sigma_\theta)

as custom metrics.

Then visualise the mean:

[
\mu_\theta
]

and uncertainty:

[
\sigma_\theta
]

outputs through separate scatter plots.

Briefly interpret the results.

### Hint

Custom metrics are implemented analogously to the loss and are returned via the `metrics` dict.

---

## 3.3.5

Revisit the California Housing dataset and apply a neural network with sufficient capacity and the probabilistic treatment from above using the provided Flax/NNX skeleton.

Provide learning curves and compare the obtained MAE to the results from the previous exercises.

Then, select different quantiles of the predicted uncertainty:

[
\sigma_\theta
]

as thresholds and compute the MAE only for predictions fulfilling these upper thresholds.

Further compute the correlation coefficient between the absolute prediction error and:

[
\sigma_\theta
]

### Hint

Use `train_test_split` with:

```python
test_size=0.2
random_state=0
```

Again, it can be helpful to standardize the dataset in advance.

---

# Problem 3.4 (25%)

Ensemble methods are machine learning techniques that combine the predictions of multiple individual models to improve overall predictive accuracy.

In the following, we study the gradient boosting method to construct a strong learner:

[
F_m(\cdot)
]

out of an ensemble of:

[
m \in {1, ..., M}
]

weak learners:

[
h_m(\cdot)
]

Starting from a base model:

[
F_0(\cdot)
]

we update the ensemble under a stagewise procedure with step size:

[
\alpha
]

[
F_m(x_i) = F_{m-1}(x_i) + \alpha h_m(x_i)
\tag{3.4.1}
]

Thereby, we select:

[
h_m(\cdot)
]

such that:

[
F_m(\cdot)
]

improves upon the previous model:

[
F_{m-1}(\cdot)
]

We interpret (3.4.1) as a gradient descent scheme, which yields that the negative gradient:

[
r_i^{(m-1)} = -\nabla_{F_{m-1}} L(y_i, F_{m-1}(x_i))
\tag{3.4.2}
]

is a suitable update direction for:

[
F_{m-1}(\cdot)
]

For each step, the newly added weak learner:

[
h_m(\cdot)
]

is thus fitted to the so-called pseudo residuals:

[
r_i^{(m-1)}
]

from (3.4.2), before the ensemble is updated according to (3.4.1).

---

## 3.4.1

Consider a regression framework with inputs:

[
x
]

and targets:

[
y
]

under a simple squared loss function:

[
L(\cdot) = (y_i - F_{m-1}(x_i))^2
]

Show that the negative gradient from (3.4.2) does indeed resemble the residuals up to a constant factor.

Briefly interpret how the ensemble model:

[
F_m(\cdot)
]

is improved for each added weak learner:

[
h_m(\cdot)
]

---

## 3.4.2

Download the:

```text
gradient_boosting_regression.csv
```

dataset from TUWEL and implement a gradient boosting regressor that can use arbitrary scikit-learn models as the weak learners:

[
h_m(\cdot)
]

Select the constant mean estimator:

[
F_0(\cdot) = \frac{1}{N} \sum_{i=1}^{N} y_i
]

as the base model and iteratively increase:

[
m
]

to:

[
M = 100
]

by fitting:

[
h_m(\cdot)
]

to the residuals:

[
r_i^{(m-1)}
]

using a:

```python
DecisionTreeRegressor(max_depth=1)
```

Visualize the regressor estimates:

[
F_m(x_i)
]

for increasing ensemble sizes:

[
m
]

comparing it to the targets:

[
y_i
]

Additionally, provide learning curves of the MSE over:

[
m
]

### Hint

Select a constant step size of:

[
\alpha = 0.5
]

Consider an object-oriented style following the scikit-learn API.

---

## 3.4.3

Switch the `DecisionTreeRegressor` for a `LinearRegression` scheme and compare the obtained ensemble to the results from Task 3.4.2.

Discuss whether a linear model is a reasonable choice for a boosting algorithm.

The gradient boosting method can also be extended to classification use cases by mapping the regressor output:

[
F_m(x_i)
]

to class probabilities.

For binary classification with the targets:

[
y \in {0,1}
]

we compute:

[
\hat{y}_i^{(m)} = \frac{1}{1 + e^{-F_m(x_i)}} \in (0,1)
\tag{3.4.3}
]

and use the cross-entropy loss function:

[
L(\cdot) = -(y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i))
\tag{3.4.4}
]

---

## 3.4.4

Compute the gradient of the cross-entropy loss with respect to:

[
F_{m-1}
]

and show that the pseudo residuals for the binary classification tasks are given by:

[
r_i^{(m-1)} = y_i - \hat{y}_i^{(m-1)}
]

### Hint

The derivative of the sigmoid function:

[
\sigma(x)
]

can be written as:

[
\frac{\partial \sigma(x)}{\partial x} = \sigma(x)(1-\sigma(x))
]

---

## 3.4.5

Download the:

```text
gradient_boosting_classification.csv
```

dataset from TUWEL and extend your implementation to the binary classification use case.

Select the constant:

[
F_0(\cdot) = 0
]

estimator as the base model and iteratively increase the ensemble size by fitting a weak learner:

[
h_m(\cdot)
]

to the residuals from Task 3.4.4, using a:

```python
DecisionTreeRegressor(max_depth=1)
```

Map the ensemble output to the class probabilities and report the accuracy over increasing ensemble sizes:

[
m
]

with:

[
M = 1000
]

Further, visualize the decision surface through a heatmap plot.

---

## 3.4.6

Again, switch the `DecisionTreeRegressor` for a `LinearRegression` scheme and compare the obtained ensemble to the results from Task 3.4.5.

Briefly discuss how the obtained decision surface differs.
