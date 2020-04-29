import jax.numpy as np
from jax import random

# Synthetic dataset
def gen_pinwheel(radial_std=0.3, tangential_std=0.1, num_classes=5,
                 num_per_class=20, rate=0.25, rng=random.PRNGKey(0)):
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = (random.normal(rng, (num_classes * num_per_class, 2)) \
                            * np.array([radial_std, tangential_std]) \
                            + np.array([1., 0.]))

    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    data = np.einsum("ti,tij->tj", features, rotations)
    #data = 2 * random.permutation(rng, data)
    return data, labels

