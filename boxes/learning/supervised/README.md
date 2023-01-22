# Learning : supervised

Supervised learning requires *labeled data*, where the correct output for each input is known.

## Gradient descent

1. Explore some *simple* data: [simple data](1_load_simple_data.py)
2. Fit a linear function by taking discrete steps [numerical steps - simple](2_numerical_steps_simple.py)
3. Fit a linear function by descending the *numerical* gradient [numerical gradient - simple](3_numerical_gradient_simple.py)
4. Fit a linear function by descending the *analytical* gradient [analytical gradient - simple](4_analytical_gradient_simple.py)
5. Explore some *complex* data: [complex data](5_load_complex_data.py)
6. Fit a nonlinear function by descending the *numerical* gradient [numerical gradient - complex](6_numerical_gradient_complex.py)
7. Fit a nonlinear function *faster* by descending the *numerical* gradient with momentum [numerical gradient with momentum - complex](7_numerical_gradient_momentum_complex.py)

## Automatic differentiation

Install Jax (Autograd + XLA)

```bash
pip intsall jax
pip install jaxlib
```

8. Fit a nonlinear function using Jax to automatically compute the gradient [Autodiff gradient - complex](8_autodiff_gradient_complex.py)
9. Fit a nonlinear function *faster* using Jax and momentum [Autodiff gradient with momentum - complex](9_autodiff_gradient_momentum_complex.py)

## Neural networks

10. Fit a simple neural network using Jax and Stax [simple network](10_network_autodiff_gradient_momentum.py)

----

### Ground truth

Parameters (simple):
- A = 1.75
- B = -4.22

Parameters (complex):
- A = 0.75
- B = 0.05
- C = -8.5
- D = 0.21
- E = 9.15

----
