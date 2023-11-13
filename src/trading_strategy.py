import utils
import numpy as np
import torch
import signatory
import itertools
from utils import timeit
from time import time


class TradingStrategy(object):
    def __init__(self, depth: int, delta: float) -> None:
        """
        This class implements the SigTrading strategy as defined in Signature Trading: A Path-Dependent Extension of the
        Mean-Variance Framework with Exogenous Signals (Owen Futter, Blanka Horvath, and Magnus Wiese, 2023)

        Arguments:
        - depth (int) : truncation level for the signature
        - delta (float) : risk aversion parameter
        """
        self.depth = depth
        self.delta = delta
        self.functionals = None  # functionals l_m for m = 1, ..., d
        self.Z_dimension = None  # dimension of the market factor process Z_t = (t, X_t, f_t), will be set in fit method as d + N + 1
        # self.Z_dimension is also the number of letters in the alphabet used to represent words (for indexing purposes)
        self.mu_sig = None
        self.sigma_sig = None
        self.inverted_sigma_sig = None

    def f(self, m: int) -> int:
        """
        Shift operator f as defined in the paper.
        Note that we represent "letters" as integers to allow for direct indexing.
        """
        return str(self.Z_dimension + m)  # or self.Z_dimension + m + 1 ?

    @timeit
    def compute_mu_sig(self, E_ZZ_LL: torch.Tensor) -> torch.Tensor:
        """
        This method computes the mu_sig vector as defined in the paper.
        E_ZZ_LL needs to be computed to truncated level >=self.depth+1. (see (*))
        """
        mu_i_sig_length = utils.get_number_of_words_leq_k(
            self.depth, self.Z_dimension
        )  # mu_i_sig_length = 1 + Z + Z^2 + ... + Z^depth
        mu_sig = torch.zeros(
            self.d * mu_i_sig_length
        )  # mu_sig = [mu_1_sig, ..., mu_d_sig] flat vector
        words = utils.get_length_leq_k_words(
            self.depth, self.Z_dimension
        )  # list of all words of length <= depth in alphabet of size Z
        words_len = len(words)
        for m in range(self.d):
            shift = self.f(m)
            for word in words:
                word_shifted = word + shift  # word_shifted is a str
                mu_sig_index = m * words_len + utils.word_to_i(
                    word, self.Z_dimension  # word and not word_shifted here!!
                )  # yes, because mu_sig is a flat vector, not a signature (i.e. sum of tensors of size k)
                mu_sig[mu_sig_index] = E_ZZ_LL[
                    utils.word_to_i(
                        word_shifted, 2 * self.Z_dimension
                    )  # but E_ZZ_LL is a signature (i.e. sum of tensors of size k) so we self.word_to_i for indexing
                ]
                # /!\ mu_sig est de dim Z alors que E_ZZ_LL est de dim 2*Z, d'où la différence d'indexing!!

        print("\033[92m" + "mu_sig successfully computed" + "\033[0m")
        # pct = (mu_sig != 0).sum() / len(mu_sig)
        # print(f"Percentage of non-zero elements in mu_sig: {pct}")
        self.mu_sig = mu_sig
        return mu_sig

    @timeit
    def compute_sigma_sig(self, E_ZZ_LL: torch.Tensor) -> torch.Tensor:
        """
        This method computes the mu_sig matrix as defined in the paper.
        """
        sigma_sig_length = self.d * utils.get_number_of_words_leq_k(
            self.depth, self.Z_dimension
        )
        sigma_sig = torch.zeros(
            (sigma_sig_length, sigma_sig_length)
        )  # sigma_sig = [sigma_1_sig, ..., sigma_d_sig] square matrix
        words = utils.get_length_leq_k_words(
            self.depth, self.Z_dimension
        )  # list of all words of length <= depth in alphabet of size Z
        words_len = len(words)
        for m in range(self.d):
            shift_m = self.f(m)
            # print(f"m={m}, there are {len(words)} words to tackle for this value of m")
            for w in words:
                wfm = w + shift_m
                for n in range(self.d):
                    shift_n = self.f(n)
                    for v in words:
                        vfn = v + shift_n
                        # calcul de sigma_sig_{wf(m),vf(n)}

                        # calcul du terme de gauche
                        left_term = 0
                        words_shuffle = utils.shuffle_product(wfm, vfn)
                        # cela génère des mots de longueur <= 2*depth, donc on doit les tronquer!? --> non, il faut juste un E_ZZ_LL tronqué à 2*depth
                        # il y a aussi des doublons potentiels ? si oui, à enlever ?
                        # print(f'words={words}')
                        for word in words_shuffle:
                            left_term += E_ZZ_LL[
                                utils.word_to_i(
                                    word, 2 * self.Z_dimension
                                )  # 2* self.Z_dimension car on index sur E_ZZ_LL de dim 2*self.Z_dimension
                            ]

                        # calcul du terme de droite
                        right_term = (
                            E_ZZ_LL[
                                utils.word_to_i(wfm, 2 * self.Z_dimension)
                            ]  # indice pour E_ZZ_LL de dimension 2*Z (et pas Z)
                            * E_ZZ_LL[utils.word_to_i(vfn, 2 * self.Z_dimension)]
                        )

                        sigma_sig_value = left_term - right_term
                        # print(
                        #     f"computing sigma_sig_value={left_term}-{right_term}={sigma_sig_value}"
                        # )
                        index_i = m * words_len + utils.word_to_i(
                            w, self.Z_dimension
                        )  # w and not wfm!
                        index_j = n * words_len + utils.word_to_i(
                            v, self.Z_dimension
                        )  # v and not vfm!
                        index = (index_i, index_j)
                        # print(f"setting sigma_sig[{index}] = {sigma_sig_value}")
                        sigma_sig[index] = sigma_sig_value

        print("\033[92m" + "sigma_sig successfully computed" + "\033[0m")
        self.sigma_sig = sigma_sig
        return sigma_sig

    def compute_functionals(self) -> torch.Tensor:
        """
        This method computes the functionals l_m for m = 1, ..., d as defined in the paper.
        """
        inv_sigma_sig = torch.inverse(self.sigma_sig)
        inv_sigma_sig_times_vector = torch.matmul(inv_sigma_sig, self.mu_sig)
        words = utils.get_length_leq_k_words(self.depth, self.Z_dimension)
        words_len = len(words)
        self.functionals = []
        for m in range(self.d):
            l_m = torch.zeros(words_len)
            for w in words:
                wfm = w + self.f(m)
                l_m[utils.word_to_i(w, self.Z_dimension)] = inv_sigma_sig_times_vector[
                    m * words_len + utils.word_to_i(w, self.Z_dimension)
                ] / (2 * self.lambda_)
            self.functionals.append(l_m)

    def fit(self, X: torch.Tensor, f: torch.Tensor) -> None:
        """
        This method fits the trading strategy to the data.

        Arguments:
        - X (np.ndarray) : tradable asset's price paths
        - f (np.ndarray) : non-tradable factor's paths
        """
        assert (
            X.shape[0] == f.shape[0]
        ), "Number of price paths and factor paths must be the same"
        assert X.shape[1] == f.shape[1], "X and f must have same length"

        # 0. retrieve dimensions
        M = X.shape[0]  # number of paths considered
        T = X.shape[1]  # number of time steps for each path
        d = X.shape[2]  # number of tradable assets (i.e. dimension of the price paths)
        N = f.shape[
            2
        ]  # number of non-tradable factors (i.e. dimension of the factor paths)

        # for later use
        self.d = d
        self.N = N
        self.Z_dimension = d + N + 1

        # 1. aggregate price and factor paths into market factor process Z_t = (t, X_t, f_t) (no time component)
        Z = torch.zeros((M, T, self.Z_dimension))
        # # time component t
        Z[:, :, 0] = torch.arange(T)  # time is defined with t_i = i
        # price component X_t
        Z[:, :, 1 : d + 1] = X
        # factor component f_t
        Z[:, :, d + 1 :] = f
        # Z has shape (M, T, Z) where Z = 1 + d + N

        # 2. compute the lead-lag transform Z^LL_t of each market factor process Z_t
        Z_LL = utils.compute_lead_lag_transform(Z)
        # Z_LL has shape (M, T, 2*Z)

        # 3. compute the N-truncated signature ZZ^^LL_t of each lead-lag transform Z^LL_t
        # ZZ_LL = utils.compute_signature(Z_LL, self.depth+1) # we'll need 2*(depth+1) to compute sigma later on
        # yes, now is the time for 2*(depth+1) to compute sigma
        ZZ_LL = utils.compute_signature(Z_LL, 2 * (self.depth + 1))
        # ZZ_LL has shape (M, T, K) with K = 1 + (2*Z) + (2*Z)^2 + ... + (2*Z)^(depth+1)

        # 4. compute the expected N-truncated signature E[ZZ^^LL_t] using the empirical mean
        E_ZZ_LL = torch.mean(ZZ_LL, axis=0)
        # E_ZZ_LL has shape (T, K)

        # 5. compute the mu_sig vector as defined in the paper
        mu_sig = self.compute_mu_sig(E_ZZ_LL)
        # mu_sig has shape (d, mu_i_length) with mu_i_length = 1 + Z + Z^2 + ... + Z^depth

        # 6. compute the sigma_sig matrix as defined in the paper
        sigma_sig = self.compute_sigma_sig(E_ZZ_LL)

        # 7. compute lambda the variance-scaling parameter
        self.compute_lambda()

        # 7. now we can finally compute the functionals l_m for m = 1, ..., d
        self.compute_functionals()

        # print 'Fitting successful' in green
        print("\033[92m" + "Fitting successful" + "\033[0m")

    @timeit
    def compute_lambda(self) -> float:
        """
        Computes the variance-scaling parameter lambda as defined in the paper. (using mu_sig and sigma_sig)
        """
        inv_sigma_sig = torch.inverse(self.sigma_sig)
        inv_sigma_sig_times_vector = torch.matmul(inv_sigma_sig, self.mu_sig)
        words = utils.get_length_leq_k_words(self.depth, self.Z_dimension)
        words_len = len(words)
        s = 0
        for m in range(self.d):
            for n in range(self.d):
                for w in words:
                    wfm = w + self.f(m)
                    index_wfm = m * words_len + utils.word_to_i(w, self.Z_dimension)
                    for v in words:
                        vfn = v + self.f(n)
                        index_vfn = n * words_len + utils.word_to_i(v, self.Z_dimension)

                        # compute all three terms
                        wfm_term = inv_sigma_sig_times_vector[index_wfm]
                        vfn_term = inv_sigma_sig_times_vector[index_vfn]
                        sigma_sig_term = self.sigma_sig[(index_wfm, index_vfn)]
                        # print(
                        #     f"wfm_term={wfm_term}, vfn_term={vfn_term}, sigma_sig_term={sigma_sig_term}"
                        # )
                        # their product is the sum increment
                        s_incr = wfm_term * vfn_term * sigma_sig_term

                        s += s_incr

        # print lambda successufully computed in green
        print("\033[92m" + "Lambda successfully computed" + "\033[0m")
        print(f"computing self.lambda_ = 2* sqrt({s}/{self.delta})")
        self.lambda_ = 2 * np.sqrt(s / self.delta)

    def trade(self, X: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        This method implements the trading strategy given new unseen data, after the model has been fitted.

        Arguments:
        - X (np.ndarray) : tradable asset's price paths
        - f (np.ndarray) : non-tradable factor's paths

        Returns:
        - np.ndarray : trading strategy's paths
        """
        assert (
            self.functionals is not None
        ), "The model must be fitted before calling the trade method"

        assert X.shape[0] == f.shape[0], "X and f must have the same length"

        # retrieve dimensions
        T = X.shape[0]  # number of time steps
        d = X.shape[1]  # number of tradable assets (i.e. dimension of the price paths)
        N = f.shape[
            1
        ]  # number of non-tradable factors (i.e. dimension of the factor paths)

        # aggregate price and factor paths into market factor process Z_t = (t, X_t, f_t)
        Z = np.zeros((T, d + N + 1))
        # time component t
        Z[:, 0] = np.arange(T)
        # price component X_t
        Z[:, 1 : d + 1] = X
        # factor component f_t
        Z[:, d + 1 :] = f

        # for t in range(T), compute the signature of each market factor process Z_{0 to t}
        len_signatures = utils.get_number_of_words_leq_k(
            self.depth, self.Z_dimension
        )  # we have t signatures and each have this length
        ZZ = torch.zeros((T, len_signatures))
        xi = torch.zeros(
            (T, self.d)
        )  # xi has shape (T, d) i.e. one row per time step, one column per tradable asset (so T rows and d columns)
        for t in range(T):
            for m in range(self.d):
                xi[t, m] = torch.dot(self.functionals[m], ZZ[t, :])

        return xi

    def compute_pnl(self, X: torch.Tensor, xi: torch.Tensor) -> float:
        """
        This method computes the PnL of the trading strategy when prices follow path X and the trading strategy is given by xi.

        Arguments:
        - X (np.ndarray) : tradable asset's price paths
        - xi (np.ndarray) : trading strategy

        Returns:
        - float : trading strategy's PnL
        """
        assert (
            X.shape[0] == xi.shape[0] + 1
        ), "xi must have exactly one less time step than X"
        assert X.shape[1] == xi.shape[1], "X and xi must have same dimension"

        # retrieve dimensions
        T = X.shape[0]
        d = X.shape[1]

        pnl = 0
        for m in range(d):
            # compute the PnL for asset m as the sum of xi_t^m * (X_(t+1)^m-X_t^m)
            pnl_m = np.sum(xi[:, m] * (X[1:, m] - X[:-1, m]))
            pnl += pnl_m

        return pnl
