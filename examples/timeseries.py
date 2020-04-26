"""
Generate synthetic dataset. Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/generate_timeseries.py
"""

import jax
import jax.numpy as jnp

import os
import errno
import tarfile
import pickle


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath
                )


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


class Periodic1DGap:
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
        super(Periodic1DGap, self).__init__()

        self.init_freq = init_freq
        self.init_amplitude = init_amplitude
        self.final_amplitude = final_amplitude
        self.final_freq = final_freq
        self.z0 = z0

    def sample(self,
               key,
               n_samples=1,
               noise_weight=1.,
               max_t_extrap_left=5.,
               min_t_extrap_right=10.,
               max_t_extrap_right=15.,
               n_tp=100):
        """
        Sample periodic functions.
        """
        n_tp_left = n_tp // 2
        n_tp_right = n_tp - n_tp_left
        timesteps_extrap_left = jax.random.uniform(key,
                                                   (n_tp_left - 1, ),
                                                   minval=0.,
                                                   maxval=max_t_extrap_left)
        timesteps_extrap_right = jax.random.uniform(key,
                                                    (n_tp_right - 1, ),
                                                    minval=min_t_extrap_right,
                                                    maxval=max_t_extrap_right)
        timesteps = jnp.sort(jnp.concatenate((jnp.array([0.]),
                                              timesteps_extrap_left,
                                              timesteps_extrap_right)))

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


class PhysioNet:
    """
    PhysioNet Dataset.
    """

    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self,
                 root,
                 train=True,
                 download=False,
                 quantization=0.1,
                 n_samples=None):

        self.root = root
        self.train = train
        self.reduce = "average"
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        infile = open(os.path.join(self.processed_folder, data_file), 'rb')
        self.data = pickle.load(infile)
        infile.close()

        infile = open(os.path.join(self.processed_folder, self.label_file), 'rb')
        self.labels = pickle.load(infile)
        infile.close()

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]


    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename)
            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], jnp.array(l[1:]).astype(float)
                    outcomes[record_id] = labels

                outfile = open(os.path.join(self.processed_folder, filename.split('.')[0] + '.pt'), 'wb')
                pickle.dump(labels, outfile)
                outfile.close()

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [jnp.zeros(len(self.params))]
                    mask = [jnp.zeros(len(self.params))]
                    nobs = [jnp.zeros(len(self.params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        time = round(time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(jnp.zeros(len(self.params)))
                            mask.append(jnp.zeros(len(self.params)))
                            nobs.append(jnp.zeros(len(self.params)))
                            prev_time = time

                        if param in self.params_dict:
                            #vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                tt = jnp.array(tt)
                vals = jnp.stack(vals)
                mask = jnp.stack(mask)

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]

                patients.append((record_id, tt, vals, mask, labels))

            outfile = open(os.path.join(self.processed_folder,
                                        filename.split('.')[0] + "_" + str(self.quantization) + '.pt'), 'wb')
            pickle.dump(patients, outfile)
            outfile.close()

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                os.path.join(self.processed_folder,
                    filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def label_file(self):
        return 'Outcomes-a.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str


def variable_time_collate_fn(batch,
                             args,
                             data_type="train",
                             data_min=None,
                             data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device = device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask,
        att_min = data_min, att_max = data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict


def init_periodic_data(rng, parse_args):
    """
    Initialize toy data. This example is easier since time_points are shared across all examples.
    """
    n_samples = 1000
    noise_weight = 0.01

    timesteps, samples = Periodic1DGap(init_freq=None,
                                    init_amplitude=1.,
                                    final_amplitude=1.,
                                    final_freq=None,
                                    z0=1.).sample(rng,
                                                  n_samples=n_samples,
                                                  noise_weight=noise_weight)

    def _split_train_test(data, train_frac=0.8):
        data_train = data[:int(n_samples * train_frac)]
        data_test = data[int(n_samples * train_frac):]
        return data_train, data_test

    train_y, test_y = _split_train_test(samples)

    num_train = len(train_y)
    assert num_train % parse_args.batch_size == 0
    num_train_batches = num_train // parse_args.batch_size

    assert num_train % parse_args.test_batch_size == 0
    num_test_batches = num_train // parse_args.test_batch_size

    # make sure we always save the model on the last iteration
    assert num_train_batches * parse_args.nepochs % parse_args.save_freq == 0

    def gen_data(batch_size, shuffle=True, subsample=None):
        """
        Generator for train data.
        """
        key = rng
        num_batches = num_train // batch_size
        inds = jnp.arange(num_train)

        def swor(subkey, w, k):
            """
            Sample k items from collection of n items with weights given by w.
            """
            n = len(w)
            g = jax.random.gumbel(subkey, shape=(n,))
            g += jnp.log(w)
            g *= -1
            return jnp.argsort(g)[:k]

        def get_subsample(subkey, sample):
            """
            Subsample timeseries.
            """
            subsample_inds = jnp.sort(swor(subkey, jnp.ones_like(timesteps), subsample))
            return sample[subsample_inds], timesteps[subsample_inds]

        while True:
            if shuffle:
                key, = jax.random.split(key, num=1)
                epoch_inds = jax.random.shuffle(key, inds)
            else:
                epoch_inds = inds
            for i in range(num_batches):
                batch_inds = epoch_inds[i * batch_size: (i + 1) * batch_size]
                if subsample is not None:
                    # TODO: if we want to do proportional subsampling I don't think we can vmap
                    yield jax.vmap(get_subsample)(jax.random.split(key, num=batch_size), train_y[batch_inds])
                else:
                    yield train_y[batch_inds], jnp.repeat(timesteps[None], batch_size, axis=0)

    # TODO: jit these!
    ds_train = gen_data(parse_args.batch_size, subsample=parse_args.subsample)
    ds_test = gen_data(parse_args.test_batch_size, shuffle=False)

    meta = {
        "num_batches": num_train_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test, meta
