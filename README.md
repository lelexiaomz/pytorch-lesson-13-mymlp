# PyTorch Lesson 13: MLP Regression

**Author:** lelexiaomz

## Overview

This lesson uses PyTorch to train a small multilayer perceptron (MLP) to approximate the nonlinear function:

$$
y = x^2
$$

The program generates 500 samples over `[-19, 30]`, randomly splits them into training, validation, and test sets, and normalizes inputs and targets using statistics computed from the training set. Training uses mean squared error (MSE), the Adam optimizer, a StepLR scheduler, and early stopping based on validation loss.

## File

- `pytorch lesson 13_mymlp.py` — dataset, MLP model, training/validation/testing, prediction, and plotting code.

The `MyDataset` class generates the data, so no external dataset file is required.

## Requirements

- Python 3
- PyTorch
- Matplotlib

Install dependencies in Windows PowerShell:

```powershell
py -m pip install torch matplotlib
```

If the `py` launcher is unavailable, use pip associated with your active Python environment.

## Run

Open a terminal in the script's directory and run:

```powershell
py "pytorch lesson 13_mymlp.py"
```

The program prints dataset splits, normalization statistics, model architecture, training and validation progress, test loss, and the prediction and relative error for `x = 90`. It also saves and displays the training/validation loss plot.

## Model and training configuration

- Data: 500 evenly spaced `x` values in `[-19, 30]`; targets are `y = x²`.
- Split: 70% training, 15% validation, and the remainder for testing, randomly assigned with `random_split`.
- Normalization: both inputs and targets use z-score statistics calculated from the training set.
- Model: `1 → 64 → 1` with a ReLU hidden activation.
- Loss: mean squared error (`MSELoss`).
- Optimizer: Adam with an initial learning rate of `0.01`.
- Scheduler: `StepLR(step_size=500, gamma=0.5)`. Since `scheduler.step()` is called once per epoch, the learning rate is multiplied by 0.5 every 500 epochs.
- Training limit: 5000 epochs; early-stopping patience: 500 epochs.

## Outputs and notes

- The best model state is saved to `lesson13_best_model.pth`, relative to the process's current working directory.
- The loss plot is saved as `loss_curve.png` and displayed with Matplotlib.
- The example predicts at `x = 90`, outside the training range `[-19, 30]`; this is extrapolation and is for demonstration only. Accuracy is not guaranteed.
- Results may vary with random dataset splitting, model initialization, PyTorch version, and runtime environment. A fixed seed improves reproducibility but does not guarantee bit-for-bit identical results across environments.

