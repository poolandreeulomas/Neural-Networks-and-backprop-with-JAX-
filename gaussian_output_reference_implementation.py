'''
- Below you can find one possible implementation of the negative log-likelihood loss which can be passed to model.compile().
- This implementation requires the output to be a concatenation of mu and log_var estimates, which should stem from your own model.
- Note, that the loss function is implemented with jax.numpy functions, such that the gradients of the overall model are available.
'''

import jax.numpy as jnp
from flax import nnx

def loss_and_metrics(
        preds: Array,
        y: Array, 
        mask: Array
):
    mean, log_var = preds[..., 0], preds[..., 1]  # shape (batch_size,)
    # TODO: Ensure y has the same shape as mean/log_var

    # Compute valid batch size for averaging the loss and metrics
    valid_batch_size = jnp.maximum(jnp.sum(mask), 1.0)
  
    # Per Batch average negative log-likelihood loss
    loss = jnp.sum((0.5 * jnp.log(2. * jnp.pi) + 0.5 * log_var + 0.5 * (y - mean) ** 2 * jnp.exp(-log_var)) * mask) / valid_batch_size

    # TODO: Compute per batch MSE and mean std_dev metrics
    mse_metric = ...
    sigma_metric = ...

    # TODO: Undo standardization for mse_metric and sigma_metric if necessary

    return loss, {"loss": loss, "mse": mse_metric, "sigma": sigma_metric}

# The neural network generates a intermediate output from which mu and log_var are computed
# This is an example of how the model's __call__ method can be implemented to produce the required output format for the loss function
def __call__(self, x: Array):
    # ... hidden layers (e.g., self.fc1 (hidden_dim1), self.fc2 (hidden_dim2)) and activations ...
    # Assuming intermediate_output is the output of the last hidden layer 
    mu = self.mu(intermediate_output)          # shape (batch_size, 1)
    log_var = self.log_var(intermediate_output)  # shape (batch_size, 1)
    return jnp.concatenate([mu, log_var], axis=-1)  # shape (batch_size, 2)