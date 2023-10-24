import utils
import numpy as np
import torch
import signatory


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
        mu_i_length = signatory.signature_channels(self.Z_dimension, self.depth)
        mu_sig = torch.zeros((self.d,mu_i_length)) # mu_sig = [mu_1_sig, ..., mu_d_sig]
        for m in range(self.d):
            shift = self.f(m)
            words = signatory.all_words(self.Z_dimension, self.depth)
            # compute mu_i_sig
            mu_i_sig = torch.zeros(mu_i_length)
            for k in range(self.depth):
                # get all words of length k
                words_k = utils.get_level_k_words(k, self.Z_dimension)
                # get (k+1)-th term of the signature since we have words of length k+1 (because of the shift operator)
                level_k_signature = signatory.extract_signature_term(E_ZZ_LL, 2*self.Z_dimension, k+1) # (*)
                # compute mu_i_sig for each word of length k
                for word in words_k:
                    mu_i_sig[word] = level_k_signature[word+shift]
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

        # 1. aggregate price and factor paths into market factor process Z_t = (t, X_t, f_t)
        Z = torch.zeros((M, T, self.Z_dimension))
        # time component t
        Z[:, :, 0] = torch.arange(T)  # time is defined with t_i = i
        # price component X_t
        Z[:, :, 1 : d + 1] = X
        # factor component f_t
        Z[:, :, d + 1 :] = f

        # 2. compute the lead-lag transform Z^LL_t of each market factor process Z_t
        Z_LL = utils.compute_lead_lag_transform(Z)

        # 3. compute the N-truncated signature ZZ^^LL_t of each lead-lag transform Z^LL_t
        ZZ_LL = utils.compute_signature(Z_LL, self.N)

        # 4. compute the expected N-truncated signature E[ZZ^^LL_t] using the empirical mean
        E_ZZ_LL = torch.mean(ZZ_LL, axis=0)

        # 5. compute the mu_sig vector as defined in the paper
        mu_sig = self.compute_mu_sig(E_ZZ_LL)

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
