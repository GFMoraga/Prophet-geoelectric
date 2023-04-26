''' This model is used to predict the Geoelectric field for the next 6 hours.
The model uses date from real time data from the USGS Geomagnetic Observatory Network.
Prophet is used to make the predictions and has been modified to fit the data
Author: Gabriel Moraga, Greg Lucas, Maxine Hartnett, Matthew Bourque '''

import argparse
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.plot import performance_metrics

'''The function below is used to get the data from the USGS Geomagnetic Observatory Network. 
The output is a Pandas DataFrame with the data from the observatory of choice, as well as how far back
you want to go. The default is 7 days back. The data is in the form of a csv file.'''


def get_observatory_data(observatory_name="CMO", start_time=None, end_time=None):
    '''
    Parameters
    ----------
    observatory_name : str
        The name of the observatory to get data from. Default is "BOU".
        3 - letter observatory code.
    start_time : str
        The start time of the data to get. Default is 7 days ago.
        date formatted string: ("%Y-%m-%dT%H:%M:%SZ")
    end_time : str
    ...
    Returns
    -------
    df : Pandas DataFrame with the data from the observatory of choice, as well as how far back
    you want to go. The default is 7 days back. The data is in the form of a csv file.

    '''
    dt_format: str = "%Y-%m-%dT%H:%M:%SZ"
    if end_time is None:
        end_time = datetime.utcnow()
    elif isinstance(end_time, str):
        try:
            end_time = datetime.strptime(dt_format)
        except ValueError:
            raise ValueError(f"end_time must be of the format {dt_format}")

    if start_time is None:
        # Go 7 days back
        start_time = end_time - timedelta(days=7)
    else:
        try:
            start_time = datetime.strptime(start_time, dt_format)
        except ValueError:
            raise ValueError(f'start_time must be of the format {dt_format}')

    base_url = "https://lasp.colorado.edu/space-weather-portal/latis/dap/"
    data_type = "variation"
    if end_time < datetime(2017, 1, 1, 1, 1):
        # The definitive data is only available after 2017
        data_type = "definitive"
    observatory_url = f"usgs_geomag_{observatory_name.lower()}_{data_type}.csv"
    query_params = (f"?time,X,Y,Z&time>={start_time.strftime(dt_format)}"
                    f"&time<={end_time.strftime(dt_format)}"
                    "&formatTime(yyyy-MM-dd'T'HH:mm:ss)")
    # Query_params is used instead of request.GET.
    # Reason is that documentation recommends using query_params instead of request.GET.
    url = base_url + observatory_url + query_params
    df = pd.read_csv(url, parse_dates=["Time"], na_values="99999.00",
                     header=0, names=["Time", "X", "Y", "Z"])
    return df


'''
Returns
-------
df : Pandas DataFrame with the data from the observatory of choice, as well as how far back
you want to go. The default is 7 days back. The data is in the form of a csv file.'''

'''This function is used to clean up the data and make it ready for the Prophet model. 
The output is two dataframes, one with the data and one with the NaN values '''


def cleanup_dataframe(df, dfnan, var_of_interest="X", var_second="Z"):
    '''
    PARAMETERS
    ----------
    df : Pandas DataFrame
    dfnan : Pandas DataFrame with NaN values
    var_of_interest : str
        The variable of interest to predict. Default is "X".
    ...
    Returns
    -------
    df : Cleaned up Pandas DataFrame
    dfnan : Cleaned up Pandas DataFrame with NaN values

    '''
    dfnan = df[df[var_of_interest].isna()]
    dfnan['ds'] = pd.DatetimeIndex(dfnan['Time'])
    dfnan = dfnan.drop(['Time', var_of_interest, var_second], axis=1)
    dfnan.columns = ['y', 'ds']
    df['ds'] = pd.DatetimeIndex(df['Time'])
    df = df.drop(['Time', var_of_interest, var_second], axis=1)
    df.columns = ['y', 'ds']

    return df, dfnan


'''
Returns
-------
df : Cleaned up Pandas DataFrame
dfnan : Cleaned up Pandas DataFrame with NaN values
var_of_interest : str

    '''

''' This function is used to make the predictions using the Prophet model. 
The change point prior scale and seasonality prior scale are used to tune the model and can be changed'''


def prophet_model(df, dfnan):
    '''
    PARAMETERS
    ----------
    df : Pandas DataFrame
    m : Prophet model that fits the data
            m uses the following parameters: changepoint_prior_scale=0.001, seasonality_prior_scale=0.01
    future : dfnan is used to make the predictions with the model
    forecast : uses m.predict(future) to make the predictions, which is then merged with the data
    merge : merges the data with the predictions
    merge_y : fills the NaN values with the predictions
    ...
    Returns
    -------
    merge_y : Pandas DataFrame with the predictions
    merge : Pandas DataFrame with the predictions and the data

    '''
    m = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01).fit(df)
    # The reason for the low prior scale is that the data is not very noisy, so we don't want to overfit the model
    # Change_point_prior_scale and seasonality_prior_scale are used to tune the model and can be adjusted
    future = dfnan
    forecast = m.predict(future)
    merge = df.merge(forecast, on='ds', how='left')
    merge_y = merge['y'].fillna(merge['yhat'])

    return merge_y, merge


''' 
Returns
-------
merge_y : Pandas DataFrame with the predictions
merge : Pandas DataFrame with the predictions and the data
'''

''' This function is used to tune the hyperparameters of the model.
The output prints the best parameters for the model'''


def tune_hyperparameters(m1):
    '''
    PARAMETERS
    ----------
    m1 : Prophet model that fits the data
    df_cv : cross validation of the model
    df_p : performance metrics of the model
    best_params : prints the best parameters for the model
    ...
    Returns
    -------
    best_params : prints the best parameters for the model

    '''
    parm = {'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5], 'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]}
    # Use cross validation to evaluate all combinations of hyperparameters
    df_cv = cross_validation(m1, initial='3 days', period='2 days', horizon='1 days')
    df_p = performance_metrics(df_cv)
    df_p['rmse'] = df_p['rmse'].abs()
    # Find the set of hyperparameters that minimizes the RMSE, but you can use MAPE or MAE
    best_params = df_p.loc[df_p['rmse'].idxmin()]
    # plot results
    # fig = plot_cross_validation_metric(df_cv, metric='rmse')
    # plt.show()

    return best_params


'''
Returns
-------
best_params : prints the best parameters for the model
    '''

'''A main function to call all the functions'''


def prediction(observatory_name="CMO", var_of_interest="Y", var_second="Z"):
    '''
    PARAMETERS
    ----------
    df : Pandas DataFrame
    dfnan : Pandas DataFrame
    merge_y : Pandas DataFrame
    merge : Pandas DataFrame
    ...
    Returns
    -------
    merge_y : Pandas DataFrame with the predictions
    merge : Pandas DataFrame with the predictions and the data

    '''

    # Get the data from the USGS Geomagnetic Observatory Network
    df = get_observatory_data(observatory_name=observatory_name)
    # Make copy of the data
    dfnan = df
    # Clean up the data
    df, dfnan = cleanup_dataframe(df, dfnan, var_of_interest, var_second)
    # Merge the data and make the predictions
    merge_y, merge = prophet_model(df, dfnan)
    # Concatenate the data
    df = prophet_model(df, dfnan)[1]
    # Make the predictions
    m1 = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01).fit(df)
    future_1 = m1.make_future_dataframe(periods=6, freq='h', include_history=True)
    forecast_1 = m1.predict(future_1)
    # call the function to tune the hyperparameters
    # best_params = tune_hyperparameters(m1)

    return m1, forecast_1


'''
Returns
-------
forecast_1 : Pandas DataFrame with the predictions
'''

# Call the main function: defaults with X
forecast_1 = prediction()

''' This function is used to make the predictions for all the observatories in the 
USGS Geomagnetic Observatory Network.'''

# only print the observatories that have data for the last 7 days
for observatory in ["CMO", "BOU", "BRW", "DED", "FRD", "FRN", "GUA", "HON", "NEW", "SHU", "SJG", "TUC", "SIT", "SJT",
                    "SVR", "THD", "TST", "TUC", "WUH", "WUS"]:
    try:
        prediction(observatory_name=observatory)
    except:
        continue
    m1, forecast_1 = prediction(observatory_name=observatory)
    fig1 = m1.plot(forecast_1)
    plt.xlabel('Time at ' + observatory)
    plt.ylabel('Magnetic field ')

    plt.show(block=False)

# Make a command line interface
if __name__ == '__main__':
    # Make the parser and add the arguments
    parser = argparse.ArgumentParser(description='Predict the geomagnetic field')
    parser.add_argument('-o', '--observatory', type=str, help='The observatory of choice')
    parser.add_argument('-d', '--days', type=int, help='The number of days back to go')
    parser.add_argument('-v', '--var', type=str, help='The variable of interest')
    parser.add_argument('-p', '--predict', type=str, help='Make the predictions')
    parser.add_argument('-t', '--tune', type=str, help='Tune the hyperparameters')
    parser.add_argument('-c', '--cross', type=str, help='Cross validation')
    parser.add_argument('-m', '--metrics', type=str, help='Performance metrics')
    parser.add_argument('-b', '--best', type=str, help='Best parameters')
    args = parser.parse_args()
