# Geoelectric Field data forecast model: 2023

This model has been made to predict the Geoelectric Field data for the next 6 hours. It uses real-time data
(or historical) and interpolates missing data, cleans the data by reformatting the dataframe, and offers a forecast
set by turning the hypermeters. Prophet was used to create the model with reason being that it preforms incredibly
well with time series data. It is compared to libraries such as ARIMA, SARIMA, Tensorflow, and others. The model is
able to predict the data with a 95% confidence interval.

### Quick start: How to use this model

This will be the function that will be called when the program is run. It will call all the other functions that are
specified in the program.

For example:

    def prediction(observatory_name="BOU"):
    # Get the data from the USGS Geomagnetic Observatory Network
    df = get_observatory_data(observatory_name=observatory_name)
    # Make copy of the data
    dfnan = df
    # Clean up the data
    df, dfnan = cleanup_dataframe(df, dfnan)
    # Merge the data and make the predictions
    merge_y, merge = prophet_model(df, dfnan)
    # Concatenate the data
    df = prophet_model(df, dfnan)[1]
    # Make the predictions
    m1 = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01).fit(df)
    future_1 = m1.make_future_dataframe(periods=6, freq='h', include_history=True)
    forecast_1 = m1.predict(future_1)
    best_params = tune_hyperparameters(m1)
    print(best_params)

    return m1, forecast_1

> Call the main function: forecast_1 = prediction()
> > This is the final dataframe that will be used to plot the data or whatever you want to do with it.

![My image](/Users/gabriel/Desktop/Data-project/geoelectric_fields/AGU/prophet_6hr_forcast.png)


## Details: How does it work and building the model

### Gathering the data: Real-time or Historical

Generate a function that will pull the data from https://lasp.colorado.edu/space-weather-portal/latis/dap/ and
return a dataframe with the data.

With this function, you should be able to pull in the data for the observatory of your choice.
> Observatories: BOU, BRW, CMO, DHT, FDT, FRD, GUA, HON, NEW, SHU, SIT, TUC.

Ideally, the functions should convert the date into a CSV, then read the CSV with pandas.
Pandas is a great starting point for data analysis and manipulation.
> The dataframe will be in variables of X, Y, and Z.
> > **X**: Geographic North, **Y**: Geographic East, **Z**: Geographic Down
> > > The measurements are in *__nT (nanoTesla)__*, which is a measure of magnetic field strength.

### Clean up the data: IMPORTANT

Prophet requires the data to be in a specific format. The data must be in a dataframe with two columns,
*__ds__* and *__y__*. *__ds__* is the date and *__y__* is the value. The data must also be sorted by date.
> *__NOTE__*: Prophet will not work if the data is not in this format.

Generate a function that will clean the data and set it up for prophet.
> *__NOTE__*: This function will also interpolate the data to fill in any missing data.

    dfnan = df[df['X'].isna()]
    dfnan['ds'] = pd.DatetimeIndex(dfnan['Time'])
    dfnan = dfnan.drop(['Time', var_of_interest, 'Z'], axis=1)
    dfnan.columns = ['y', 'ds']
    df['ds'] = pd.DatetimeIndex(df['Time'])
    df = df.drop(['Time', var_of_interest, 'Z'], axis=1)
    df.columns = ['y', 'ds']

This will return a dataframe with the data in the correct format, as well as a dataframe with the missing data.
> *__NOTE__*: The missing data will be used later to compare the model to the actual data.

### Create the Prophet model

Prophet is a great library for time series data. It can be tricky to get the model to work, but once it does, it
allows for great insight into the data. Prophet works well with other libraries such as Pandas, Numpy, and Matplotlib.
> *__NOTE__*: Prophet is a great library, but it is not perfect. It is important to check the model to make sure
> it is working correctly. This can be done by comparing the model to the actual data via plots.

Prophet works under the following parameters that are used in the model:

> *__change_point_prior_scale__*: This is the flexibility of the model.
> The higher the value, the more flexible the model.
>
> *__seasonality_prior_scale__*: This is the flexibility of the seasonality.
> The higher the value, the more flexible the seasonality.
>
> *__future__*: This is the number of hours/days/years you want to forecast.
>
> *__forecast__*: This is the dataframe that will be used to forecast the data.
> > *__NOTE__*: must use m.predict(forecast) to get the forecasted data.
>
> *__m__*: This is the model that will be used to forecast the data.
> > **IMPORTANT**: *m* must be used to call certain functions in the library.

In this model *merge* is used to merge the dataframes together. This is done because we want *y* and *yhat* to be
in the same dataframe. These two columns come from the *__forecast = m.predict(future)__* variable. *y* is the actual
data and *yhat* is the forecasted data. *yhat* is the forecasted data with the confidence interval.

> *__TIP__*: If you want to see the forecasted data without the confidence interval, use *__m.plot(forecast)__*.
> This will give you a plot of the forecasted data without the confidence interval.

### Hyperparameters tuning: what works best?

The hyperparameters are found by using the *__cross_validation__* function and *__performance_metrics__* function.
> *__NOTE__*: This is a very time-consuming process. Use a small dataset to test the model.

The *__cross_validation__* function will take the model and run it through the data. It will then return a dataframe
with the data. The *__performance_metrics__* function will take the dataframe from *__cross_validation__* and return
a dataframe with the metrics. The metrics are used to determine which hyperparameters work best.

> *__cross_validation__* takes the following arguments:
> > *__m__*, *__horizon__*, *__period__*, *__initial__*, and *__parallel__*

For the sake of time and not over-working the computer, you can customize the arguments to fit based on how
large the dataset is. This will come by testing the model with different arguments within the *__cross_validation__*.

> *__performance_metrics__* takes the following arguments:
> > *__df_cv (cross_validation dataframe)__*, *__metrics__*, and *__rolling_window__*

> *__metrics__* takes the following arguments:
> * mae (mean absolute error)
> * mse (mean squared error)
> * rmse (root mean squared error)
> * mape (mean absolute percentage error)
> * mdape (median absolute percentage error)
> * coverage (prediction coverage)
>
> __NOTE__*: The default is mape, rmse, and coverage.

### Results: run a for loop to get the data for all the observatories

This will run a for loop to get the data for all the observatories. The dataframe will have filled all NaN values with
the forecasted data and then generate an additive forecast for each observatory. Depending on the parameters set and
how much data you are using, the time it takes to run the program will vary.

> The for loop should be able to generate plots for each observatory. This will allow you to see how the model is
> working for each observatory.

### Conclusion: what can be done to improve the model?

The model can be improved by using more data. The more data you use, the better the model will be.
> *__NOTE__*: The more data you use, the longer it will take to run the program.
>
> *__TIP__*: Use a small dataset to test the model. Once the model is working, use a larger dataset or apply model in
> a cloud environment.

> Import the libraries needed for the model:

    from datetime import datetime, timedelta
    import pandas as pd
    import matplotlib.pyplot as plt
    from prophet import Prophet
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics

Documentation for libraries used:
> Datetime: https://docs.python.org/3/library/datetime.html

> Pandas: https://pandas.pydata.org/docs/

> Matplotlib: https://matplotlib.org/3.3.3/contents.html

> Prophet:*__NOTE-depending on what platform you use, installing will be different
__* https://facebook.github.io/prophet/docs/quick_start.html#python-api
> > prophet.diagnostics: https://facebook.github.io/prophet/docs/diagnostics.html
>
> > cross_validation: https://facebook.github.io/prophet/docs/diagnostics.html#cross-validation
>
> > performance_metrics: https://facebook.github.io/prophet/docs/diagnostics.html#performance-metrics

> import argparse: https://docs.python.org/3/library/argparse.html








