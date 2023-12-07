import utils
import numpy as np
import torch
import signatory
import itertools
from utils import timeit
from time import time
from typing import List


class TradingStrategy(object):

    """
    This class implements the SigTrading strategy as defined in Signature Trading: A Path-Dependent Extension of the
    Mean-Variance Framework with Exogenous Signals (Owen Futter, Blanka Horvath, and Magnus Wiese, 2023)

    Arguments:
    - depth (int) : truncation level for the signature
    - delta (float) : risk aversion parameter
    """

    def __init__(self, depth: int, delta: float) -> None:
        self.depth = depth
        self.delta = delta
        self.functionals = None  # functionals l_m for m = 1, ..., d
        self.d = None  # number of tradable assets, will be set in fit method
        self.N = None  # number of non-tradable factors, will be set in fit method
        self.Z_dimension = None  # dimension of the market factor process Z_t = (t, X_t, f_t), will be set in fit method as d + N + 1
        # self.Z_dimension is also the number of letters in the alphabet used to represent words (for indexing purposes)
        self.mu_sig = None
        self.sigma_sig = None

    def f(self, m: int) -> List[int]:
        """
        Shift operator f as defined in the paper.
        Note that we represent "letters" as integers to allow for direct indexing. A word is consequently a list of integers.
        """
        return [
            self.Z_dimension + m + 1
        ]  # even a single letter is a word, so it is represented as a list

    @timeit
    def compute_mu_sig(self, E_ZZ_LL: torch.Tensor) -> None:
        """
        This method computes the mu_sig vector as defined in the paper.
        Note that E_ZZ_LL must be truncated to level >=self.depth+1 in order for this method to work. (see (*))
        """

        # 1. create empty mu_sig
        mu_i_sig_length = utils.get_number_of_words_leq_k(
            self.depth, self.Z_dimension
        )  # mu_i_sig_length = 1 + Z + Z^2 + ... + Z^depth
        mu_sig = torch.zeros(
            self.d * mu_i_sig_length
        )  # mu_sig = [mu_1_sig, ..., mu_d_sig] flat vector with d vectors of length mu_i_sig_length

        # 2. create the list of all words of length <= depth in alphabet of size Z, which we need to index mu_sig
        words = utils.get_length_leq_k_words(self.depth, self.Z_dimension)
        words_len = len(words)

        # 3. populate mu_sig
        for m in range(self.d):  # iterate over m
            fm = self.f(m)  # represents the letter f(m) in the paper
            for word in words:
                # let's populate mu_sig[wfm] using (3.2) in the paper

                word_shifted = word + fm  # represents the word wf(m) in the paper

                # find the integer index of word_shifted in mu_sig
                mu_sig_index = m * words_len + utils.word_to_i(
                    word, self.Z_dimension  # word and not word_shifted here!
                )

                # find the integer index of word_shifted in E_ZZ_LL
                E_ZZ_LL_index = utils.word_to_i(
                    word_shifted, 2 * self.Z_dimension
                )  # 2*self.Z_dimension because ZZ_LL has dimension 2*self.Z_dimension

                # now populate the appropriate cell in mu_sig with the appropriate cell in E_ZZ_LL
                mu_sig[mu_sig_index] = E_ZZ_LL[
                    E_ZZ_LL_index
                ]  # (*): we need E_ZZ_LL to be truncated to level >=self.depth+1 for this to work

        print("\033[92m" + "mu_sig successfully computed" + "\033[0m")
        self.mu_sig = mu_sig

    @timeit
    def compute_sigma_sig(self, E_ZZ_LL: torch.Tensor) -> None:
        """
        This method computes the mu_sig matrix as defined in the paper.
        Note that E_ZZ_LL must be truncated to level >=2*(self.depth+1) in order for this method to work. (see (*))
        """

        # 1. create empty sigma_sig
        sigma_sig_length = self.d * utils.get_number_of_words_leq_k(
            self.depth, self.Z_dimension
        )
        sigma_sig = torch.zeros(
            (sigma_sig_length, sigma_sig_length)
        )  # sigma_sig = [sigma_1_sig, ..., sigma_d_sig] square matrix

        # 2. create the list of all words of length <= depth in alphabet of size Z, which we need to index sigma_sig
        words = utils.get_length_leq_k_words(self.depth, self.Z_dimension)
        words_len = len(words)

        for m in range(self.d):  # iterate over m
            fm = self.f(m)  # represents the letter f(m) in the paper
            for w in words:
                wfm = w + fm  # represents the word wf(m) in the paper
                for n in range(self.d):  # iterate over n
                    fn = self.f(n)  # represents the letter f(n) in the paper
                    for v in words:
                        # let's populate sigma_sig[wfm, vfn] using (3.3) in the paper

                        vfn = v + fn  # represents the word vf(n) in the paper

                        # computing the left term, which is a sum over a shuffle product of words
                        left_term = 0
                        words_shuffle = utils.shuffle_product(wfm, vfn)
                        for word in words_shuffle:
                            E_ZZ_LL_index = utils.word_to_i(word, 2 * self.Z_dimension)
                            left_term += E_ZZ_LL[E_ZZ_LL_index]

                        # computing the right term, which is a product of two values
                        index_E_ZZ_LL_wfm = utils.word_to_i(wfm, 2 * self.Z_dimension)
                        index_E_ZZ_LL_vfn = utils.word_to_i(vfn, 2 * self.Z_dimension)
                        right_term = (
                            E_ZZ_LL[index_E_ZZ_LL_wfm] * E_ZZ_LL[index_E_ZZ_LL_vfn]
                        )

                        # computing the final value
                        sigma_sig_value = left_term - right_term

                        # now populate the appropriate cell in sigma_sig with the appropriate value
                        index_i = m * words_len + utils.word_to_i(
                            w, self.Z_dimension
                        )  # w and not wfm!
                        index_j = n * words_len + utils.word_to_i(
                            v, self.Z_dimension
                        )  # v and not vfm!
                        index = (index_i, index_j)
                        sigma_sig[index] = sigma_sig_value

        print("\033[92m" + "sigma_sig successfully computed" + "\033[0m")
        self.sigma_sig = sigma_sig

    def compute_functionals(self) -> None:
        """
        This method computes the functionals l_m for m = 1, ..., d as defined in the paper.
        Remember that (3.1) allows us to compute l_m as a function of mu_sig and sigma_sig. (plus a rescaling factor for variance)
        """

        # 1. create empty functionals
        self.functionals = []

        # 2. compute sigma_sig inverse times mu_sig
        inv_sigma_sig = torch.inverse(self.sigma_sig)
        inv_sigma_sig_times_vector = torch.matmul(inv_sigma_sig, self.mu_sig)

        # 3. create the list of all words of length <= depth in alphabet of size Z, which we need for indexing purposes
        words = utils.get_length_leq_k_words(self.depth, self.Z_dimension)
        words_len = len(words)

        for m in range(self.d):  # iterate over m
            # let's populate l_m using (3.1) in the paper
            l_m = torch.zeros(words_len)
            for w in words:
                fm = self.f(m)  # represents the letter f(m) in the paper
                wfm = w + fm  # represents the word wf(m) in the paper
                l_m_index = utils.word_to_i(w, self.Z_dimension)
                inv_sigma_sig_times_vector_index = m * words_len + utils.word_to_i(
                    w, self.Z_dimension
                )

                # use (3.1)
                l_m[l_m_index] = inv_sigma_sig_times_vector[
                    inv_sigma_sig_times_vector_index
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

        # time component t
        # Z[:, :, 0] = torch.arange(T)  # time is defined with t_i = i
        # NEW : let's use normalized time instead (so that t has less importance)
        Z[:, :, 0] = torch.arange(T) / T

        # price component X_t
        Z[:, :, 1 : d + 1] = X
        # factor component f_t
        Z[:, :, d + 1 :] = f
        # Z has shape (M, T, Z) where Z = 1 + d + N

        # 2. compute the lead-lag transform Z^LL_t of each market factor process Z_t
        Z_LL = utils.compute_lead_lag_transform(Z)
        # Z_LL has shape (M, T, 2*Z)

        # 3. compute the N-truncated signature ZZ^^LL_t of each lead-lag transform Z^LL_t
        # /!\ note that we need to truncate to level >=2*(self.depth+1) for this to work because of (3.3) for computing sigma_sig
        # this can create computational bottlenecks if self.depth is large...
        ZZ_LL = utils.compute_signature(Z_LL, 2 * (self.depth + 1))
        # ZZ_LL has shape (M, T, K) with K = 1 + (2*Z) + (2*Z)^2 + ... + (2*Z)^(2*(depth+1))

        # 4. compute the expected N-truncated signature E[ZZ^^LL_t] using the empirical mean
        E_ZZ_LL = torch.mean(ZZ_LL, axis=0)
        # E_ZZ_LL has shape (T, K)

        # 5. compute the mu_sig vector as defined in the paper
        self.compute_mu_sig(E_ZZ_LL)
        # mu_sig has shape (d, mu_i_length) with mu_i_length = 1 + Z + Z^2 + ... + Z^depth

        # 6. compute the sigma_sig matrix as defined in the paper
        self.compute_sigma_sig(E_ZZ_LL)

        # 7. compute lambda the variance-scaling parameter
        self.compute_lambda()

        # 7. now we can finally compute the functionals l_m for m = 1, ..., d
        self.compute_functionals()

        # print 'Fitting successful' in green and bold
        print("\033[1m" + "\033[92m" + "Fitting successful.\n" + "\033[0m" + "\033[0m")

    @timeit
    def compute_lambda(self) -> float:
        """
        Computes the variance-scaling parameter lambda as defined in the paper. (using mu_sig and sigma_sig)
        """
        inv_sigma_sig = torch.inverse(self.sigma_sig)
        inv_sigma_sig_times_vector = torch.matmul(inv_sigma_sig, self.mu_sig)

        # create the list of all words of length <= depth in alphabet of size Z, which we need for indexing purposes
        words = utils.get_length_leq_k_words(self.depth, self.Z_dimension)
        words_len = len(words)

        s = 0  # lambda is computed as a sum so we need to initialize it

        for m in range(self.d):  # iterate over m
            for n in range(self.d):  # iterate over n
                for w in words:  # iterate over words w of length <= depth
                    wfm = w + self.f(m)  # represents the word wf(m) in the paper
                    index_wfm = m * words_len + utils.word_to_i(
                        w, self.Z_dimension
                    )  # wrong!?
                    for v in words:  # iterate over words v of length <= depth
                        vfn = v + self.f(n)  # represents the word vf(n) in the paper
                        index_vfn = n * words_len + utils.word_to_i(v, self.Z_dimension)

                        # compute all three terms
                        wfm_term = inv_sigma_sig_times_vector[index_wfm]  # term 1
                        vfn_term = inv_sigma_sig_times_vector[index_vfn]  # term 2
                        sigma_sig_index = (index_wfm, index_vfn)
                        sigma_sig_term = self.sigma_sig[sigma_sig_index]  # term 3

                        # their product is the sum increment
                        s_incr = wfm_term * vfn_term * sigma_sig_term

                        # add it to the sum
                        s += s_incr

        # print lambda successufully computed in green
        print("\033[92m" + "lambda successfully computed" + "\033[0m")
        self.lambda_ = 0.5 * np.sqrt(s / self.delta)

    def trade(
        self, X: torch.Tensor, f: torch.Tensor, min_steps: int = 5
    ) -> torch.Tensor:
        """
        This method implements the trading strategy given data, after the model has been fitted.

        Arguments:
        - X (np.ndarray) : tradable asset's price paths
        - f (np.ndarray) : non-tradable factor's paths
        - min_steps (int) : minimum number of time steps to consider before trading

        Returns:
        - np.ndarray : trading strategy's paths (i.e. list of positions xi_t^m for each tradable asset m at each time step t)
        """
        assert (
            self.functionals is not None
        ), "The model must be fitted before calling the trade method"

        assert X.shape[0] == f.shape[0], "X and f must have the same length"

        # 0. retrieve dimensions
        T = X.shape[0]  # number of time steps
        d = X.shape[1]  # number of tradable assets (i.e. dimension of the price paths)
        N = f.shape[
            1
        ]  # number of non-tradable factors (i.e. dimension of the factor paths)

        # 1. aggregate price and factor paths into market factor process Z_t = (t, X_t, f_t)
        Z = torch.zeros((T, d + N + 1))
        # time component t
        Z[:, 0] = torch.arange(T) / T  # normalized time for stability
        # price component X_t
        Z[:, 1 : d + 1] = X
        # factor component f_t
        Z[:, d + 1 :] = f

        # 2. initialize xi
        xi = torch.zeros(
            (T, self.d)
        )  # xi has shape (T, d) i.e. one row per time step, one column per tradable asset (so T rows and d columns)

        # 3. compute vector xi_t for t = min_steps, ..., T-1
        for t in range(min_steps, T):
            Z_t = Z[:t, :]  # we only look at information up to now
            ZZ_t = utils.compute_signature(Z_t, self.depth, no_batch=True)
            for m in range(self.d):
                xi[t, m] = torch.dot(self.functionals[m], ZZ_t)

        return xi

    def compute_pnl(self, X: torch.Tensor, xi: torch.Tensor) -> torch.tensor:
        """
        This method computes the PnL of the trading strategy when prices follow path X and the trading strategy is given by xi.

        Arguments:
        - X (np.ndarray) : tradable asset's price paths
        - xi (np.ndarray) : trading strategy

        Returns:
        - float : trading strategy's PnL
        """
        assert X.shape[0] == xi.shape[0], "xi and X must have the same number of steps"
        assert X.shape[1] == xi.shape[1], "X and xi must have same dimension"

        # 0. retrieve dimensions
        T = X.shape[0]
        d = X.shape[1]

        # 1. compute daily PnL
        daily_pnl = xi[:-1, :] * (
            X[1:, :] - X[:-1, :]
        )  # daily_pnl has shape (T-1, d) and it's the pnl for each asset for each "trading day" (ie. each time step)

        return daily_pnl

    def print_functionals(self) -> None:
        """
        Displays the functionals l_m for m = 1, ..., d in a pretty way.
        """
        for i, func in enumerate(self.functionals):
            print("\033[91m" + "L_" + str(i + 1) + "\033[0m")
            utils.print_signature(func.flatten(), self.Z_dimension, self.depth)
