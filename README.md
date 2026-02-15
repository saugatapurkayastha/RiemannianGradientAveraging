# Riemannian Gradient Averaging Optimizer

This repository provides a reference implementation of the
Riemannian Gradient Averaging (RGrad-Avg) optimizer for optimization on
smooth manifolds.

The algorithm inherently uses concepts from differential geometry like exponential retraction map and parallel transport.

## Repository Contents

- `RGD.py`: Implementation of the Riemannian Gradient Averaging optimizer.

## 



## Usage

You can use this optimizer in either of the following three ways.

### Option 1: Installation

To install the package locally, run the following command in the root directory:

```bash
pip install .
```

Once installed, you can import it as 
```bash
from RGD import RiemannianGradientAveraging
```


### Option 2: Download the file and use it locally

1. Download the file `RGD.py`.
2. Place it in the same directory as your Python script.

Import it as:

    from RGD import RiemannianGradientAveraging

This option is suitable for quick experiments or standalone scripts.

### Option 3: Clone the GitHub repository

1. Clone the repository.
2. Add the repository to your Python path.

## Class Arguments

The `RiemannianGradientAveraging` class accepts the following arguments during initialization:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `manifold` | object | **Required** | An object representing the manifold. It must implement:<br>• `project_tangent(x, v)`: Projects vector `v` to tangent space at `x`.<br>• `exp(x, v)`: Exponential map or retraction at `x` along `v`.<br>• `transport(x_from, x_to, v)`: Parallel transports vector `v` from `x_from` to `x_to`. |
| `cost` | callable | **Required** | The cost function to minimize. It takes a point `x` and returns a scalar. |
| `lr` | float | `0.1` | The step size (learning rate) for the gradient updates. |
| `max_iter` | int | `100` | The maximum number of iterations to run the optimization. |
| `tol` | float | `1e-6` | Tolerance parameter (available for custom convergence checks). |

## Example

    optimizer = RiemannianGradientAveraging(
        manifold=manifold,
        cost=cost,
        lr=0.1,
        max_iter=50,
        tol=1e_6
    )

    x_opt, history = optimizer.run(X, y, grad_func, w0)



## Citation

If you find the repository useful, please cite the following work:

```bib
@inproceedings{
purkayastha2025on,
title={On Riemannian Gradient Descent Algorithm using gradient averaging},
author={Saugata Purkayastha and Sukannya Purkayastha},
booktitle={OPT 2025: Optimization for Machine Learning},
year={2025},
url={https://openreview.net/forum?id=GGFhHH96Ze}
}
```

The work is released under the Apache License.