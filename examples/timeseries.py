"""
Generate synthetic dataset. Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/generate_timeseries.py
"""

import jax
import jax.numpy as jnp


def _get_next_val(t, tmin, tmax, init, final=None):
    """
    Linearly interpolate on the interval (tmin, tmax) between values (init, final).
    """
    if final is None:
        return init
    val = init + (final - init) / (tmax - tmin) * t
    return val


def _gen_sample(timesteps,
                init_freq,
                init_amplitude,
                starting_point,
                final_freq=None,
                final_amplitude=None,
                phi_offset=0.):
    """
    Generate time-series sample.
    """

    tmin = timesteps[0]
    tmax = timesteps[-1]

    data = []
    t_prev = timesteps[0]
    phi = phi_offset
    for t in timesteps:
        dt = t - t_prev
        amp = _get_next_val(t, tmin, tmax, init_amplitude, final_amplitude)
        freq = _get_next_val(t, tmin, tmax, init_freq, final_freq)
        phi = phi + 2 * jnp.pi * freq * dt                                       # integrate to get phase

        y = amp * jnp.sin(phi) + starting_point
        t_prev = t
        data.append([y])
    return jnp.array(data)


def _add_noise(key, samples, noise_weight):
    n_samples, n_tp, n_dims = samples.shape

    # add noise to all the points except the first point
    noise = jax.random.uniform(key, (n_samples, n_tp - 1, n_dims))

    samples = jax.ops.index_update(samples, jax.ops.index[:, 1:], samples[:, 1:] + noise * noise_weight)
    return samples


def _assign_value_or_sample(key, value, sampling_interval=(0., 1.)):
    """
    Return constant, otherwise return uniform random sample in the interval.
    """
    if value is None:
        value = jax.random.uniform(key,
                                   minval=sampling_interval[0],
                                   maxval=sampling_interval[1])
    key,  = jax.random.split(key, num=1)
    return key, value


class Periodic1D:
    """
    Period 1 dimensional data.
    """
    def __init__(self,
                 init_freq=0.3,
                 init_amplitude=1.,
                 final_amplitude=10.,
                 final_freq=1.,
                 z0=0.):
        """
        If (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
        For now, all the time series share the time points and the starting point.
        """
        super(Periodic1D, self).__init__()

        self.init_freq = init_freq
        self.init_amplitude = init_amplitude
        self.final_amplitude = final_amplitude
        self.final_freq = final_freq
        self.z0 = z0

    def sample(self,
               key,
               n_samples=1,
               noise_weight=1.,
               max_t_extrap=5.,
               n_tp=100):
        """
        Sample periodic functions.
        """
        timesteps_extrap = jax.random.uniform(key,
                                              (n_tp - 1, ),
                                              minval=0.,
                                              maxval=max_t_extrap)
        timesteps = jnp.sort(jnp.concatenate((jnp.array([0.]),
                                              timesteps_extrap)))

        def gen_sample(subkey):
            """
            Generate one time-series sample.
            """
            subkey, init_freq = _assign_value_or_sample(subkey, self.init_freq, [0.4, 0.8])
            final_freq = init_freq if self.final_freq is None else self.final_freq
            subkey, init_amplitude = _assign_value_or_sample(subkey, self.init_amplitude, [0., 1.])
            subkey, final_amplitude = _assign_value_or_sample(subkey, self.final_amplitude, [0., 1.])

            z0 = self.z0 + jax.random.normal(subkey) * 0.1

            sample = _gen_sample(timesteps,
                                 init_freq=init_freq,
                                 init_amplitude=init_amplitude,
                                 starting_point=z0,
                                 final_amplitude=final_amplitude,
                                 final_freq=final_freq)
            return sample

        samples = jax.vmap(gen_sample)(jax.random.split(key, num=n_samples))

        samples = _add_noise(key, samples, noise_weight)
        return timesteps, samples
