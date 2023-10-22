import utils
import numpy as np


class TradingStrategy(object):
    def __init__(self, N: int, delta: float) -> None:
        """
        This class implements the SigTrading strategy as defined in Signature Trading: A Path-Dependent Extension of the
        Mean-Variance Framework with Exogenous Signals (Owen Futter, Blanka Horvath, and Magnus Wiese, 2023)

        Arguments:
        - N (int) : truncation level for the signature
        - delta (float) : risk aversion parameter
        """
        self.N = N
        self.delta = delta
        self.functionals = None  # functionals l_m for m = 1, ..., d

    def compute_mu_sig(self, E_ZZ_LL: np.ndarray) -> np.ndarray:
        """
        This method computes the mu_sig vector as defined in the paper.
        """
        pass

    def compute_sigma_sig(self, E_ZZ_LL: np.ndarray) -> np.ndarray:
        """
        This method computes the mu_sig matrix as defined in the paper.
        """
        pass

    def compute_functionals(
        self, mu_sig: np.ndarray, sigma_sig: np.ndarray
    ) -> np.ndarray:
        """
        This method computes the functionals l_m for m = 1, ..., d as defined in the paper.
        """
        pass

    def compute_xi(self, Z_LL: np.ndarray) -> np.ndarray:
        """
        This method computes the trading strategy xi_t as defined in the paper.
        """
        pass

    def fit(self, X: np.ndarray, f: np.ndarray) -> None:
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

        # 1. aggregate price and factor paths into market factor process Z_t = (t, X_t, f_t)
        Z = np.zeros((M, T, d + N + 1))
        # time component t
        Z[:, :, 0] = np.arange(T)  # time is defined with t_i = i
        # price component X_t
        Z[:, :, 1 : d + 1] = X
        # factor component f_t
        Z[:, :, d + 1 :] = f

        # 2. compute the lead-lag transform Z^LL_t of each market factor process Z_t
        Z_LL = utils.compute_lead_lag_transform(Z)

        # 3. compute the N-truncated signature ZZ^^LL_t of each lead-lag transform Z^LL_t
        ZZ_LL = utils.compute_signature(Z_LL, self.N)

        # 4. compute the expected N-truncated signature E[ZZ^^LL_t] using the empirical mean
        E_ZZ_LL = np.mean(ZZ_LL, axis=0)

        # 5. compute the mu_sig vector as defined in the paper
        mu_sig = self.compute_mu_sig(E_ZZ_LL)

        # 6. compute the simga_sig matrix as defined in the paper
        sigma_sig = self.compute_sigma_sig(E_ZZ_LL)

        # 7. now we can finally compute the functionals l_m for m = 1, ..., d
        self.functionals = self.compute_functionals(mu_sig, sigma_sig)

        # print 'Fitting successful' in green
        print("\033[92m" + "Fitting successful" + "\033[0m")

    def trade(self, X: np.ndarray, f: np.ndarray) -> np.ndarray:
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

    def compute_pnl(self, X: np.ndarray, xi: np.ndarray) -> float:
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
