# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from sklearn.decomposition import FastICA, PCA, IncrementalPCA, MiniBatchSparsePCA, SparsePCA, KernelPCA
import fbpca
import numpy as np
import itertools
from types import SimpleNamespace

from tqdm import tqdm
import torch, os




# ICA
class ICAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.maxiter = 10000
        self.whiten = True  # ICA: whitening is essential, should not be skipped
        self.transformer = FastICA(n_components, random_state=0, whiten=self.whiten, max_iter=self.maxiter)
        self.batch_support = False
        self.stdev = np.zeros((n_components,))
        self.total_var = 0.0

    def get_param_str(self):
        return "ica_c{}{}".format(self.n_components, '_w' if self.whiten else '')

    def fit(self, X):
        self.transformer.fit(X)
        if self.transformer.n_iter_ >= self.maxiter:
            raise RuntimeError(f'FastICA did not converge (N={X.shape[0]}, it={self.maxiter})')

        # Normalize components
        self.transformer.components_ /= np.sqrt(np.sum(self.transformer.components_ ** 2, axis=-1, keepdims=True))

        # Save variance for later
        self.total_var = X.var(axis=0).sum()

        # Compute projected standard deviations
        self.stdev = np.dot(self.transformer.components_, X.T).std(axis=1)

        # Sort components based on explained variance
        idx = np.argsort(self.stdev)[::-1]
        self.stdev = self.stdev[idx]
        self.transformer.components_[:] = self.transformer.components_[idx]

    def get_components(self):
        var_ratio = self.stdev ** 2 / self.total_var
        return self.transformer.components_, self.stdev, var_ratio  # ICA outputs are not normalized


# Incremental PCA
class IPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(n_components, whiten=self.whiten, batch_size=max(100, 2 * n_components))
        self.batch_support = True

    def get_param_str(self):
        return "ipca_c{}{}".format(self.n_components, '_w' if self.whiten else '')

    def fit(self, X):
        self.transformer.fit(X)

    def fit_partial(self, X):
        try:
            self.transformer.partial_fit(X)
            self.transformer.n_samples_seen_ = \
                self.transformer.n_samples_seen_.astype(np.int64)  # avoid overflow
            return True
        except ValueError as e:
            print(f'\nIPCA error:', e)
            return False

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_)  # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return self.transformer.components_, stdev, var_ratio  # PCA outputs are normalized


# Standard PCA
class PCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.solver = 'full'
        self.transformer = PCA(n_components, svd_solver=self.solver)
        self.batch_support = False

    def get_param_str(self):
        return f"pca-{self.solver}_c{self.n_components}"

    def fit(self, X):
        self.transformer.fit(X)

        # Save variance for later
        self.total_var = X.var(axis=0).sum()

        # Compute projected standard deviations
        self.stdev = np.dot(self.transformer.components_, X.T).std(axis=1)

        # Sort components based on explained variance
        idx = np.argsort(self.stdev)[::-1]
        self.stdev = self.stdev[idx]
        self.transformer.components_[:] = self.transformer.components_[idx]

        # Check orthogonality
        dotps = [np.dot(*self.transformer.components_[[i, j]])
                 for (i, j) in itertools.combinations(range(self.n_components), 2)]
        if not np.allclose(dotps, 0, atol=1e-4):
            print('IPCA components not orghogonal, max dot', np.abs(dotps).max())

        self.transformer.mean_ = X.mean(axis=0, keepdims=True)

    def get_components(self):
        var_ratio = self.stdev ** 2 / self.total_var
        return self.transformer.components_, self.stdev, var_ratio


# Facebook's PCA
# Good default choice: very fast and accurate.
# Very high sample counts won't fit into RAM,
# in which case IncrementalPCA must be used.
class FacebookPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.transformer = SimpleNamespace()
        self.batch_support = False
        self.n_iter = 2
        self.l = 2 * self.n_components

    def get_param_str(self):
        return "fbpca_c{}_it{}_l{}".format(self.n_components, self.n_iter, self.l)

    def fit(self, X):
        U, s, Va = fbpca.pca(X, k=self.n_components, n_iter=self.n_iter, raw=True, l=self.l)
        self.transformer.components_ = Va

        # Save variance for later
        self.total_var = X.var(axis=0).sum()

        # Compute projected standard deviations
        self.stdev = np.dot(self.transformer.components_, X.T).std(axis=1)

        # Sort components based on explained variance
        idx = np.argsort(self.stdev)[::-1]
        self.stdev = self.stdev[idx]
        self.transformer.components_[:] = self.transformer.components_[idx]

        # Check orthogonality
        dotps = [np.dot(*self.transformer.components_[[i, j]])
                 for (i, j) in itertools.combinations(range(self.n_components), 2)]
        if not np.allclose(dotps, 0, atol=1e-4):
            print('FBPCA components not orghogonal, max dot', np.abs(dotps).max())

        self.transformer.mean_ = X.mean(axis=0, keepdims=True)

    def get_components(self):
        var_ratio = self.stdev ** 2 / self.total_var
        return self.transformer.components_, self.stdev, var_ratio


# Sparse PCA
# The algorithm is online along the features direction, not the samples direction
#   => no partial_fit
class SPCAEstimator():
    def __init__(self, n_components, alpha=10.0):
        self.n_components = n_components
        self.whiten = False
        self.alpha = alpha  # higher alpha => sparser components
        # self.transformer = MiniBatchSparsePCA(n_components, alpha=alpha, n_iter=100,
        #    batch_size=max(20, n_components//5), random_state=0, normalize_components=True)
        self.transformer = SparsePCA(n_components, alpha=alpha, ridge_alpha=0.01,
                                     max_iter=100, random_state=0, n_jobs=-1,
                                     normalize_components=True)  # TODO: warm start using PCA result?
        self.batch_support = False  # maybe through memmap and HDD-stored tensor
        self.stdev = np.zeros((n_components,))
        self.total_var = 0.0

    def get_param_str(self):
        return "spca_c{}_a{}{}".format(self.n_components, self.alpha, '_w' if self.whiten else '')

    def fit(self, X):
        self.transformer.fit(X)

        # Save variance for later
        self.total_var = X.var(axis=0).sum()

        # Compute projected standard deviations
        # NB: cannot simply project with dot product!
        self.stdev = self.transformer.transform(X).std(axis=0)  # X = (n_samples, n_features)

        # Sort components based on explained variance
        idx = np.argsort(self.stdev)[::-1]
        self.stdev = self.stdev[idx]
        self.transformer.components_[:] = self.transformer.components_[idx]

        # Check orthogonality
        dotps = [np.dot(*self.transformer.components_[[i, j]])
                 for (i, j) in itertools.combinations(range(self.n_components), 2)]
        if not np.allclose(dotps, 0, atol=1e-4):
            print('SPCA components not orghogonal, max dot', np.abs(dotps).max())

    def get_components(self):
        var_ratio = self.stdev ** 2 / self.total_var
        return self.transformer.components_, self.stdev, var_ratio  # SPCA outputs are normalized


def get_estimator(name, n_components, alpha):
    if name == 'pca':
        return PCAEstimator(n_components)
    if name == 'ipca':
        return IPCAEstimator(n_components)
    elif name == 'fbpca':
        return FacebookPCAEstimator(n_components)
    elif name == 'ica':
        return ICAEstimator(n_components)
    elif name == 'spca':
        return SPCAEstimator(n_components, alpha)
    else:
        raise RuntimeError('Unknown estimator')


def compute_pca(graph, layer_id, estimator='ipca',
                n_components=40,
                batch_size=32, n=1_000_000,
                seed=88,
                dump_path=None,
                ds=None, is_debug=False):
    output_shape = graph.flow.output_shapes[layer_id]
    sample_dims = output_shape[1]*output_shape[2]*output_shape[3]
    transformer = get_estimator('ipca', n_components, 0)
    # Figure out batch size if not provided
    B = batch_size  # or get_max_batch_size(inst, device, layer_key)
    # Divisible by B (ignored in output name)
    N = n // B * B
    # Compute maximum batch size based on RAM + pagefile budget
    target_bytes = 20 * 1_000_000_000  # GB
    feat_size_bytes = sample_dims * np.dtype('float64').itemsize
    N_limit_RAM = np.floor_divide(target_bytes, feat_size_bytes)
    if not transformer.batch_support and N > N_limit_RAM:
        print('WARNING: estimator does not support batching, ' \
              'given config will use {:.1f} GB memory.'.format(feat_size_bytes / 1_000_000_000 * N))

    # 32-bit LAPACK gets very unhappy about huge matrices (in linalg.svd)
    if estimator == 'ica':
        lapack_max_N = np.floor_divide(np.iinfo(np.int32).max // 4, sample_dims)  # 4x extra buffer
        if N > lapack_max_N:
            raise RuntimeError(f'Matrices too large for ICA, please use N <= {lapack_max_N}')


    print('B={}, N={}, dims={}, N/dims={:.1f}'.format(B, N, sample_dims, N / sample_dims), flush=True)


    torch.manual_seed(seed)
    np.random.seed(seed)



    ds_torch = torch.utils.data.DataLoader(ds, B, shuffle=False, drop_last=True)

    n_lat = len(ds_torch) * B

    latents = np.zeros((n_lat, sample_dims), dtype=np.float32)
    print(f'data number for PCA = {n_lat}')

    with torch.no_grad():
        graph.eval()
        for i, data in tqdm(enumerate(ds_torch), total=len(ds_torch), desc='generate z_list'):
            dissec_dict = dict(output_feat=True, output_feat_detach=True)
            x = data['x']
            z, nll, y_logits, z_list = graph.forward(x.cuda(), dissec=dissec_dict)
            latents[i * B:(i + 1) * B] = z_list[layer_id]['z'].reshape(B, -1)
            if i > 40 and is_debug: break

    X = latents  # Use all samples
    X_global_mean = X.mean(axis=0, keepdims=True, dtype=np.float32)  # TODO: activations surely multi-modal...!
    X -= X_global_mean

    transformer.fit(X)
    assert np.all(transformer.transformer.mean_ < 1e-3), 'Mean of normalized data should be zero'
    X_comp, X_stdev, X_var_ratio = transformer.get_components()

    # Measure component sparsity (!= activation sparsity)
    sparsity = np.mean(X_comp == 0)  # percentage of zero values in components
    print(f'Sparsity: {sparsity:.2f}')

    os.makedirs(dump_path.parent, exist_ok=True)
    np.savez_compressed(dump_path, **{
        'act_comp': X_comp.astype(np.float32),
        'act_mean': X_global_mean.astype(np.float32),
        'act_stdev': X_stdev.astype(np.float32),
        'var_ratio': X_var_ratio.astype(np.float32),
    })
    print('saving {}'.format(dump_path))

    return X_comp, X_global_mean, X_stdev