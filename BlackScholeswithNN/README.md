This project follows [Robert Culkin, Sanjiv R. Das, Machine Learning in Finance: The Case of Deep Learning for Option Pricing (2017)](https://srdas.github.io/Papers/BlackScholesNN.pdf).

Here firstly, a NN is trained on synthetic option pricing data with the following features:
- $S_0$: Stock Price
- $K$: Exercise Price
- $(T-t)$: Time to Maturity, where T is Exercise Date
- $\sigma$: Underlying Volatility (a standard deviation of log returns)
- $r$: Risk-free Interest Rate (i.e., T-bill Rate)

This trained model is validated on real world option data which results in interesting observations.
Black Scholes tends to misprice calls that are both deeply ITM or OTM, which is due to the implied volatility smile.
