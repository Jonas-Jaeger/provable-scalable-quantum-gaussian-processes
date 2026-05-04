import numpy as np
from scipy.special import factorial
from sklearn.gaussian_process.kernels import Kernel
from pennylane.ops.op_math import LinearCombination
from math import comb


def majo_cov(features_i, features_j, obs_coeffs, n, m):
    """
    Compute the covariance between two Majorana input state (feature) vectors, i.e., coefficients in Majorana basis.
    :param features_i: The feature vector of the input state i.
    :param features_j: The feature vector of the input state j.
    :param obs_coeffs: The coefficients (tensor) of the Majorana observable (linear comb of Majorana operators).
    :param n: The number of qubits.
    :param m: The number of Majorana operators in the observable.
    :return: The covariance between the two random variables.
    """
    features_contraction = np.sum(features_i * features_j)  # contract the feature vectors (general tensors)
    obs_coeffs_contraction = np.sum(obs_coeffs * obs_coeffs)  # contract the observable coefficients (general tensors)
    normalization = factorial(m)/(2*n)**m  # normalization factor
    cov = normalization * features_contraction * obs_coeffs_contraction
    return cov


class MajoQuantumKernel(Kernel):

    def __init__(self, ts, majo_observable, majo_feature_vecs, n, m):
        """
        Provides a kernel for Majorana observables and feature vectors.
        :param ts: The time points/ parameters of the input state.
        :param majo_observable: The majorana observable, a PauliSentence of the coefficients of the observable only.
        :param majo_feature_vecs: The expectation value of the input state with the Majorana basis operators.
        Also known as y tensors.
        :param n: The number of qubits.
        :param m: The number of Majorana operators in the observable.
        """
        self.ts = ts
        self.majo_observable = majo_observable
        self.majo_feature_vecs = majo_feature_vecs
        self.n = n
        self.m = m
        self.is_multi_dim_ts = ts.ndim > 1 if ts is not None else False

        # Sanity checks:
        if ts is None:
            print("Kernel used in index mode")
            num_data = np.prod(self.majo_feature_vecs.shape[:-1])
        else:
            num_data = len(self.ts) if not self.is_multi_dim_ts else np.prod(self.ts.shape[:-1])
            assert (num_data == np.prod(self.majo_feature_vecs.shape[:-1])), \
                "Time points and feature vectors must have the same length."

        are_coeffs_in_frame = (self.majo_feature_vecs.size == (num_data * (2 * n) ** m))
        are_coeffs_in_basis = (self.majo_feature_vecs.size == (num_data * comb(2 * n, m)))

        assert are_coeffs_in_frame or are_coeffs_in_basis, \
            "Feature vectors must have the correct shape, i.e., match the size of Majorana basis (2n)^m" \
             "or the number of combinations of Majorana operators (2n choose m)." \
            f"Howewer, got {self.majo_observable.size} instead of {(2 * n)**m} or {comb(2 * n, m)}."

        assert m == 1 or not (are_coeffs_in_frame and are_coeffs_in_basis), \
            f"Cannot determine if coefficients are for frame or basis."

        self.apply_basis_correction = (m != 1 and are_coeffs_in_basis)

        super().__init__()

    def __call__(self, X, Y=None, eval_gradient=False):
        self.ts = np.asarray(self.ts) if self.ts is not None else None
        self.majo_feature_vecs = np.asarray(self.majo_feature_vecs)
        # Extract coefficients from observable
        if isinstance(self.majo_observable, LinearCombination):
            assert np.all(np.isreal(self.majo_observable.coeffs))
            self.obs_coeffs = np.real(np.asarray(self.majo_observable.coeffs))
        else:
            self.obs_coeffs = np.asarray(self.majo_observable)

        if Y is None:
            Y = X
        kernel = np.zeros((len(X), len(Y)))
        for i, t_i in enumerate(X):
            idx_i = self._index_lookup(t_i)
            features_i = self.majo_feature_vecs[idx_i]
            for j, t_j in enumerate(Y):
                idx_j = self._index_lookup(t_j)
                features_j = self.majo_feature_vecs[idx_j]
                # evaluate kernel function:
                kernel[i, j] = majo_cov(features_i, features_j, obs_coeffs=self.obs_coeffs, n=self.n, m=self.m)
        #print(kernel.shape)

        if self.apply_basis_correction:
            kernel *= factorial(self.m)

        if eval_gradient:  # Zero gradient, as no hyperparameters to optimize
            return kernel, np.zeros((len(X), len(Y), 0))

        return kernel

    def is_stationary(self):
        return False

    def diag(self, X):
        return np.diag(self(X))  # Could be implemented more efficient by only evaluating diag elements

    def _index_lookup(self, t):
        """
        Find the index of the closest time point to t.
        :param t: The time point to look up.
        :return: The index of the closest time point.
        """
        if self.ts is None:  # index mode
            return t

        if self.is_multi_dim_ts:
            idx = np.unravel_index((np.linalg.norm(self.ts - t, ord=1, axis=-1)).argmin(), self.ts.shape[:-1])
        else:
            idx = (np.abs(self.ts - t)).argmin()
        return idx


# context manager to switch GP to derivative mode
class gp_derivative_mode:
    def __init__(self, gp, dt, post_process_fn=None):
        self.gp = gp
        self.orig_predict = None
        self.dt = dt
        self.post_process_fn = post_process_fn

    def __enter__(self):
        # Switch to derivative mode by replacing the predict method
        self.orig_predict = self.gp.predict
        #def deriv_predict(X, return_std=False, return_cov=False):
        #    pred = derivative_gp_mean(self.orig_gp, X, self.dt, return_std)
        #
        #     if self.post_process_fn:
        #         if return_std:
        #             pred, std = pred
        #             return self.post_process_fn(pred), std
        #         return self.post_process_fn(pred)
        #     return pred

        self.gp.predict = lambda X, return_std=False, return_cov=False: \
            derivative_gp_mean(self.orig_predict, X, self.dt, return_std=return_std)
        return self.gp

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original predict method
        self.gp.predict = self.orig_predict


#from scipy.signal import convolve2d

# class DerivativeGP:
#     def __init__(self, gp, dt):
#         self.gp = gp
#         self.dt = dt
#
#         super().__init__()
#
#     def predict(self, X, return_std=False, return_cov=False):
#         X = np.squeeze(np.asarray(X))
#         assert X.ndim <= 1, "Only 1D input is supported for derivative prediction."
#         assert np.all(np.isclose(X[1:] - X[:-1], self.dt)), \
#             "Input must have the same uniform spacing as the training data."
#
#         gp_mean, gp_cov = self.gp.predict(X.reshape(-1, 1), return_cov=True)
#         d_gp_mean = np.gradient(gp_mean, self.dt)
#         if return_std or return_cov:
#
#             # todo one kernel for both x and y
#             central_diff_kernel_x = np.array([[1, 0, -1]]) / (2 * self.dt)
#             central_diff_kernel_y = central_diff_kernel_x.T
#
#             # compute the second order derivative of the covariance matrix (using convolution, horizontal and vertical)
#             d_cov = convolve2d(gp_cov, central_diff_kernel_x, mode='same')
#             d_cov = convolve2d(d_cov, central_diff_kernel_y, mode='same')
#
#             if return_cov:
#                 return d_gp_mean, d_cov
#             elif return_std:
#                 # std from cov matrix:
#                 d_gp_std = np.sqrt(np.diag(d_cov))
#                 return d_gp_mean, d_gp_std
#             else:
#                 raise ValueError("Either return_std or return_cov must be True.")
#         else:
#             return d_gp_mean

def derivative_gp_mean(gp, X, dt, return_std=False):
    X = np.asarray(X)
    print(f"derivative for shape {X.shape} with dt={dt}")
    assert X.ndim <= 1 or (X.ndim == 2 and X.shape[1] == 1), "Only 1D input is supported for derivative prediction."
    X = np.tile(X.flatten(), 2)
    X[:len(X) // 2] -= dt
    X[len(X) // 2:] += dt
    if hasattr(gp, 'predict'):
        gp = gp.predict
    means = gp(X.reshape(-1, 1), return_cov=return_std)
    if return_std:
        means, cov = means
    p_m, p_p = means[:len(means) // 2], means[len(means) // 2:]
    deriv = (p_p - p_m) / (2 * dt)  # Central finite difference approximation
    if return_std:
        vars = np.diag(cov)
        vars_m = vars[:len(vars) // 2]
        vars_p = vars[len(vars) // 2:]
        cov_m_p = np.diagonal(cov, offset=len(p_m))  # covariance between p_m and p_p is on the N-th off-diagonal
        stds = np.sqrt(vars_m + vars_p - 2 * cov_m_p) / (2 * dt)  # assuming independence
        #p_m, cov_m = p_m
        #p_p, cov_p = p_p
        #std_m = np.sqrt(np.diag(cov_m))
        #std_p = np.sqrt(np.diag(cov_p))
        #std = np.sqrt(np.diag(cov_m) + np.diag(cov_p)) / (2 * dt)  # assuming independence
        #std_max = (std_p + std_m) / (2 * dt)  # since we do not know correlation
        return deriv, stds  # Central finite difference approximation
    return deriv


class DerivativeKernel(Kernel):
    def __init__(self, kernel, dt):
        """
        Provides a kernel for the derivative of a Gaussian process.
        :param kernel: The kernel of the Gaussian process.
        :param ts: The time points of the input state.
        """
        self.kernel = kernel
        self.dt = dt

        super().__init__()

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise NotImplementedError("Gradient evaluation is not implemented for derivative kernel.")
        if Y is None:
            Y = X
        X = np.asarray(X)
        Y = np.asarray(Y)

        k_pp = self.kernel(X + self.dt, Y + self.dt)
        k_pm = self.kernel(X + self.dt, Y - self.dt)
        k_mp = self.kernel(X - self.dt, Y + self.dt)
        k_mm = self.kernel(X - self.dt, Y - self.dt)

        # finite central difference approximation for the derivative
        # (https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences)
        d_k = (k_pp - k_pm - k_mp + k_mm) / (4 * self.dt ** 2)

        return d_k

    def is_stationary(self):
        return False

    def diag(self, X):
        return np.diag(self(X))

