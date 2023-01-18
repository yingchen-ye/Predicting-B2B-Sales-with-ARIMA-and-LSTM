# Predicting-B2B-Sales-with-ARIMA-and-LSTM
* Time series models AR, ARIMA, SARIMA, LSTMs were used to forecast B2B sales with seasonality and trend.
* This project is the base of the master thesis in collaboration with a commercial company, thus all the outputs were removed for the sake for data security.
* "Pipelines" is the diagram of the experiment.
# Preprocessing
* Libraries: ``pandas``, ``numpy``, ``matplotlib.pyplot``
* Steps: datetime data processing, stationarity check (autocorrelation) with ACF, PACF, ADF test, STL decomposition.

# AM, ARIMA, SARIMA
* Libraries: ``statsmodels``, ``scipy``, ``pmdarima``, ``statsmodels.tsa.ar_model``, ``sklearn.metrics``
* Steps: time series split, hyperparameter tuning with walk forward validation and grid search for ARIMA and SARIMA, evaluation the generalization with the residual diagnostics and regression metrics

# LSTM
* Libraries: ``TimeseriesGenerator`` in ``keras``, ``keras.models``, `` keras.layers``, ``keras.callbacks``, ``keras.regularizers``, ``keras.optimizers``, ``keras_tuner``
* Involved models: univariate and multivariate LSTM, unidirectional and bidirectional LSTM
* The tuned hyperparameters can be seen in the <b> Hyperparameters <b>