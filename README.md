# SigTrading

Implementing state-of-the-art Signature Trading method, as proposed in _Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals_. (Futter, Owen &amp; Horvath, Blanka &amp; Wiese, Magnus, 2023).

**/src** folder contains

- <code>trading_strategy.py</code> class, which is the core implementation of the Sig Trader
- <code>utils.py</code> file used for <code>trading_strategy</code>
- 3 Jupyter notebooks each illustrating the performance of the Sig Trader in different contexts

<h2 style="color:red">Important</h2>

The Sig Trader uses <code>signatory</code> to compute signatures, itself using <code>torch</code> as backend. However, most triplet versions <code>python/torch/signatory</code> are <b>not</b> compatible.

We have found that Python 3.8.0, <code>torch==1.7.1</code> and <code>signatory==1.2.4.1.7.1</code> works for our implementation.