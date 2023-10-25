import utils
import numpy as np
import torch
import signatory
import itertools
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

    def init_bijection(self, channels:int):
        # we need a bijection between words and integers for indexing purposes
        # we need to go up to 2*(depth+1) because of the shift operator f and the LL transform
        start = time()
        words = []
        for i in range(0, 2*(self.depth+1)):
            words += list(itertools.product(range(2*channels), repeat=i))
        self.i_to_word = words
        self.word_to_i = {word: i for i, word in enumerate(words)}
        end = time()
        print(f'bijection init took {end-start} seconds')

    def f(self, m:int) -> int:
        """
        Shift operator f as defined in the paper.
        Note that we represent "letters" as integers to allow for direct indexing.
        """
        return self.Z_dimension + m # or self.Z_dimension + m + 1 ?

    def compute_mu_sig(self, E_ZZ_LL: torch.Tensor) -> torch.Tensor:
        """
        This method computes the mu_sig vector as defined in the paper.
        E_ZZ_LL needs to be computed to truncated level >=self.depth+1. (see (*))
        """
        mu_i_length = signatory.signature_channels(self.Z_dimension, self.depth) # mu_i_length = 1 + Z + Z^2 + ... + Z^depth
        mu_sig = torch.zeros((self.d,mu_i_length)) # mu_sig = [mu_1_sig, ..., mu_d_sig]
        for m in range(self.d):
            shift = self.f(m)
            print(f'using shift f({m})={shift}')
            words = signatory.all_words(2*self.Z_dimension, self.depth)
            # add empty word
            words = [()] + words
            # compute mu_i_sig
            mu_i_sig = torch.zeros(mu_i_length)
            print(f'E_ZZ_LL.shape={E_ZZ_LL.shape}')
            for word_index, word in enumerate(words):
                word_shifted = word + (shift,)
                print(f'setting mu_i_sig[{word}] = E_ZZ_LL[{word_shifted}]')
                mu_i_sig[word_index] = E_ZZ_LL[self.word_to_i[word_shifted]]
            mu_sig[m] = mu_i_sig
        return mu_sig


    def compute_sigma_sig(self, E_ZZ_LL: torch.Tensor) -> torch.Tensor:
        """
        This method computes the mu_sig matrix as defined in the paper.
        """
        pass

    def compute_functionals(
        self, mu_sig: torch.Tensor, sigma_sig: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method computes the functionals l_m for m = 1, ..., d as defined in the paper.
        """
        pass

    def compute_xi(self, Z_LL: torch.Tensor) -> torch.Tensor:
        """
        This method computes the trading strategy xi_t as defined in the paper.
        """
        pass

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

        self.init_bijection(self.Z_dimension) # set the bijection between words and integers (for indexing purposes)

        # 1. aggregate price and factor paths into market factor process Z_t = (t, X_t, f_t) (no time component)
        Z = torch.zeros((M, T, self.Z_dimension))
        # # time component t
        Z[:, :, 0] = torch.arange(T)  # time is defined with t_i = i
        # price component X_t
        Z[:,:,1:d+1] = X
        # factor component f_t
        Z[:,:,d+1:] = f
        # Z has shape (M, T, Z) where Z = 1 + d + N

        # 2. compute the lead-lag transform Z^LL_t of each market factor process Z_t
        Z_LL = utils.compute_lead_lag_transform(Z)
        # Z_LL has shape (M, T, 2*Z)
        
        # 3. compute the N-truncated signature ZZ^^LL_t of each lead-lag transform Z^LL_t
        ZZ_LL = utils.compute_signature(Z_LL, self.depth+1) # we'll need 2*(depth+1) to compute sigma later on
        # ZZ_LL has shape (M, T, K) with K = 1 + (2*Z) + (2*Z)^2 + ... + (2*Z)^(depth+1)
        
        # 4. compute the expected N-truncated signature E[ZZ^^LL_t] using the empirical mean
        E_ZZ_LL = torch.mean(ZZ_LL, axis=0)
        # E_ZZ_LL has shape (T, K)
        
        # 5. compute the mu_sig vector as defined in the paper
        mu_sig = self.compute_mu_sig(E_ZZ_LL)
        # mu_sig has shape (d, mu_i_length) with mu_i_length = 1 + Z + Z^2 + ... + Z^depth

        # 6. compute the simga_sig matrix as defined in the paper
        sigma_sig = self.compute_sigma_sig(E_ZZ_LL)

        # 7. now we can finally compute the functionals l_m for m = 1, ..., d
        self.functionals = self.compute_functionals(mu_sig, sigma_sig)

        # print 'Fitting successful' in green
        print("\033[92m" + "Fitting successful" + "\033[0m")

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

        # compute the lead-lag transform Z^LL_t of each market factor process Z_t
        Z_LL = utils.compute_lead_lag_transform(Z)

        xi = self.compute_xi(Z_LL)

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
