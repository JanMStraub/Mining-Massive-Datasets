import os

import dataclasses
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds


def total_loss_all_batches(mat_u, mat_v, dataset, batch_size, lambda_u, lambda_v):
    """Compute mse per batch using vectorized operations
    Returns a list of mse values for all batches as floats
    """
    total_all_batches = []
    for record in tfds.as_numpy(dataset.batch(batch_size)):
        total_loss = total_loss_one_batch(mat_u, mat_v, record, lambda_u, lambda_v)
        total_all_batches.append(total_loss)
    # convert list of arrays to list of floats
    total_all_batches = list(map(float, total_all_batches))
    return total_all_batches


@jax.jit  # Comment out for single-step debugging
def total_loss_one_batch(mat_u, mat_v, record, lambda_u, lambda_v):
    """This colab experiment motivates the implementation:
    https://colab.research.google.com/drive/1c0LpSndbTJaHVoLTatQCbGhlsWbpgvYh?usp&#x3D;sharing=
    """
    rows, columns, ratings = (
        record["movie_id"],
        record["user_id"],
        record["user_rating"],
    )
    estimator = -(mat_u @ mat_v)[(rows, columns)]
    square_err = jnp.square(estimator + ratings)
    mse = jnp.mean(square_err)

    # Adding regularization terms
    reg_loss_u = lambda_u * jnp.sum(jnp.square(mat_u))
    reg_loss_v = lambda_v * jnp.sum(jnp.square(mat_v))

    # Total loss = MSE + regularization
    total_loss = mse + reg_loss_u + reg_loss_v

    return total_loss


def uv_factorization_reg(mat_u, mat_v, train_ds, valid_ds, config, lambda_u=0.01, lambda_v=0.01):
    """Matrix factorization using SGD with regularization
    Fast vectorized implementation using JAX
    """

    @jax.jit  # Comment out for single-step debugging
    def update_uv(mat_u, mat_v, record, lr):
        loss_value, grad = jax.value_and_grad(total_loss_one_batch, argnums=[0, 1])(
            mat_u, mat_v, record, lambda_u, lambda_v
        )
        mat_u = mat_u - lr * grad[0]
        mat_v = mat_v - lr * grad[1]
        return mat_u, mat_v, loss_value

    for epoch in range(config.num_epochs):
        lr = (
            config.fixed_learning_rate
            if config.fixed_learning_rate is not None
            else config.dyn_lr_initial
            * (config.dyn_lr_decay_rate ** (epoch / config.dyn_lr_steps))
        )
        print(
            f"In uv_factorization_vec_no_reg, starting epoch {epoch} with lr={lr:.6f}"
        )
        train_loss = []
        for record in tfds.as_numpy(train_ds.batch(config.batch_size_training)):
            mat_u, mat_v, loss = update_uv(mat_u, mat_v, record, lr)
            train_loss.append(loss)

        train_loss_mean = jnp.mean(jnp.array(train_loss))
        # Compute loss on the validation set
        valid_loss = total_loss_all_batches(
            mat_u,
            mat_v,
            valid_ds,
            config.batch_size_predict_with_mse,
            lambda_u,
            lambda_v,
        )
        valid_loss_mean = jnp.mean(jnp.array(valid_loss))
        print(
            f"Epoch {epoch} finished, ave training loss: {train_loss_mean:.6f}, ave validation loss: {valid_loss_mean:.6f}"
        )
    return mat_u, mat_v