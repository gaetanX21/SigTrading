# SigTrading

Implementing state-of-the-art <b>Signature Trading method</b>, as proposed in _Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals_. (Futter, Owen &amp; Horvath, Blanka &amp; Wiese, Magnus, 2023).

The **/src** folder contains
- <code>trading_strategy.py</code> class, which is the core implementation of the Sig Trader
- <code>utils.py</code> file used for <code>trading_strategy</code>
- 4 Jupyter notebooks each illustrating the performance of the Sig Trader in different contexts

In addition,**original_paper.pdf** contains the original paper while **report.pdf** contains the project report.

<h2 style="color:red">Important</h2>

The Sig Trader uses <code>signatory</code> to compute signatures, itself using <code>torch</code> as backend. However, most triplet versions <code>python/torch/signatory</code> are <b>not</b> compatible.

We have found that <code>Python 3.8.0</code>, <code>torch==1.7.1</code> and <code>signatory==1.2.4.1.7.1</code> works for our implementation.
