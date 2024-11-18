import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


@jax.jit
def merge_sorted_arrays(a: Array, b: Array) -> Array:
    """
    Merges two sorted arrays into a single sorted array

    Args:
        a (jax.numpy.ndarray): First sorted array.
        b (jax.numpy.ndarray): Second sorted array.

    Returns:
        jax.numpy.ndarray: Merged sorted array.
    """
    la = a.size
    lb = b.size
    total_size = la + lb
    num_less_in_b = jnp.searchsorted(b, a, side="left", method="scan_unrolled")
    positions_a = jnp.arange(la) + num_less_in_b
    num_less_in_a = jnp.searchsorted(a, b, side="right", method="scan_unrolled")
    positions_b = jnp.arange(lb) + num_less_in_a

    # full = jnp.ones(total_size, dtype=bool).at[positions_a].set(False)
    # positions_b = jnp.nonzero(full, size=lb)[0]

    merged = jnp.empty(total_size, dtype=a.dtype)
    merged = merged.at[positions_a].set(a, indices_are_sorted=True, unique_indices=True)
    merged = merged.at[positions_b].set(b, indices_are_sorted=True, unique_indices=True)

    return merged


@jax.jit
def merge_then_sort(a, b):
    merged = jnp.concatenate([a, b])
    return jnp.sort(merged)


if __name__ == "__main__":
    from timeit import default_timer as timer

    # Sample sorted arrays
    a = jnp.array([1, 4, 4, 5, 7, 9])
    b = jnp.array([2, 4, 4, 4, 4, 6, 8])

    # Merge the arrays using the parallel function
    merged_array1 = merge_sorted_arrays(a, b)

    merged_array2 = merge_then_sort(a, b)

    np.testing.assert_array_equal(merged_array1, merged_array2)

    # Benchmark the parallel function
    a = jnp.sort(jax.random.randint(jax.random.PRNGKey(0), (10000,), 0, 100000))
    b = jnp.sort(jax.random.randint(jax.random.PRNGKey(1), (10000,), 0, 100000))

    merge_sorted_arrays(a, b)  # Warmup
    merge_then_sort(a, b)  # Warmup

    start = timer()
    for _ in range(10):
        merged_array = merge_sorted_arrays(a, b)
    merged_array.block_until_ready()
    end = timer()

    print(f"Parallel merge took {end - start:.6f} seconds.")

    # Benchmark the sort-then-merge function
    start = timer()
    for _ in range(10):
        merged_array = merge_then_sort(a, b)
    merged_array.block_until_ready()
    end = timer()

    print(f"Merge-then-sort took {end - start:.6f} seconds.")
