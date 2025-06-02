import datetime
import os
import pickle

import ndlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


def load_csv(tweet_id):
    # return a panda data frame
    file_name = f'tweet_data/{tweet_id}.csv'
    if not os.path.exists(file_name):
        raise FileNotFoundError(f'Error while loading csv. File {file_name} not found.')
    tweet_data = pd.read_csv(file_name)
    tweet_data['timestamp'] = pd.to_datetime(tweet_data['timestamp'])  # reformat the timestamp
    # sort by date
    tweet_data = tweet_data.sort_values(by='timestamp')
    # Reset the index after sorting
    tweet_data.reset_index(drop=True, inplace=True)

    # add a 'share' count
    tweet_data['share_count'] = tweet_data['retweet_count'] + tweet_data['quote_count']

    return tweet_data


def interpolate_data(tweet_data, frequency='1h'):

    ### add empty rows once an hour after the initial data point ###
    # first and last time stamps
    first_timestamp = tweet_data['timestamp'].min()
    last_timestamp = tweet_data['timestamp'].max()

    # used for deleting old data once it's not needed
    tweet_data['delete'] = True
    tweet_data.loc[0, 'delete'] = False

    new_rows = []
    # Generate new timestamps between the first and last data point
    current_timestamp = first_timestamp
    while current_timestamp <= last_timestamp:
        # Check if the current timestamp is already in the DataFrame
        new_row = {
            'delete': False,
            'retweet_count': np.nan,
            'reply_count': np.nan,
            'like_count': np.nan,
            'quote_count': np.nan,
            'bookmark_count': np.nan,
            'impression_count': np.nan,
            'timestamp': current_timestamp
        }
        if not (current_timestamp in tweet_data['timestamp'].values):
            new_rows.append(new_row)
        # Move to the next hour
        current_timestamp += pd.Timedelta(hours=1)

    # add the new rows with the original DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    interpolated_data = pd.concat([tweet_data, new_rows_df])
    interpolated_data = interpolated_data.sort_values('timestamp')
    # set index
    interpolated_data.set_index('timestamp', inplace=True)
    # fill in empty rows
    interpolated_data = interpolated_data.interpolate('time')

    ### remove original data points, except the first one ###
    interpolated_data = interpolated_data.loc[interpolated_data['delete'] == False]


    # limit data to 3 days
    interpolated_data = interpolated_data.iloc[:3*24]

    # round every column down except for the index column (timestamp)
    interpolated_data = interpolated_data.apply(lambda x: np.round(x).astype(int))

    # Reset the index
    interpolated_data.reset_index(inplace=True)

    # add a share count column
    tweet_data['share_count'] = tweet_data['retweet_count'] + tweet_data['quote_count']


    # add a impression_change column, compute the change as the difference between current and previous value
    interpolated_data['impression_change'] = interpolated_data['impression_count'] - interpolated_data['impression_count'].shift(1)
    interpolated_data.loc[0, 'impression_change'] = 0
    interpolated_data['impression_change'] = interpolated_data['impression_change'].astype(int)

    return interpolated_data


def plot_metric_over_time(tweet_data, metric, title=None):
    """
    Plots the number of likes and impressions over time for a tweet using two y-axes.

    Parameters:
    tweet_data (pd.DataFrame): DataFrame containing tweet data with at least
                               'timestamp', 'like_count', and 'impression_count' columns.

    Returns:
    None
    """
    if ('timestamp' not in tweet_data.columns or metric not in tweet_data.columns or 'impression_count' not in tweet_data.columns):
        raise ValueError(f'The DataFrame must contain "timestamp", "{metric}", and "impression_count" columns.')

    # tweet_data['timestamp'] = pd.to_datetime(tweet_data['timestamp'])
    tweet_data = tweet_data.sort_values('timestamp')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot like_count on the left y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel(f'{metric}', color='tab:blue')
    ax1.plot(tweet_data['timestamp'],
             tweet_data[f'{metric}'],
             marker='o', linestyle='-', color='tab:blue', label=f'{metric}')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # # Create a second y-axis for impression_count (number of people that have been shown the tweet)
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Number of Impressions', color='tab:orange')
    # ax2.plot(tweet_data['timestamp'],
    #          tweet_data['impression_count'],
    #          marker='o', linestyle='-', color='tab:orange', label='Impressions')
    # ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))

    # fig.tight_layout()
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric} Over Time')

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=15))
    ax1.grid(True)
    plt.show()


def plot_ratio_over_time(tweet_data):
    """
    Plots the ratio of likes to impressions over time for a tweet.

    Parameters:
    tweet_data (pd.DataFrame): DataFrame containing tweet data with at least
                               'timestamp', 'like_count', and 'impression_count' columns.

    Returns:
    None
    """
    if 'timestamp' not in tweet_data.columns or 'like_count' not in tweet_data.columns or 'impression_count' not in tweet_data.columns:
        raise ValueError('The DataFrame must contain "timestamp", "like_count", and "impression_count" columns.')

    # tweet_data['timestamp'] = pd.to_datetime(tweet_data['timestamp'])
    tweet_data = tweet_data.sort_values('timestamp')

    # Calculate the ratio of likes to impressions
    tweet_data['ratio'] = tweet_data['like_count'] / tweet_data['impression_count']

    plt.figure(figsize=(10, 6))
    plt.plot(tweet_data['timestamp'],
             tweet_data['ratio'],
             marker='o', linestyle='-', label='Likes/Impressions Ratio')
    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))
    plt.xlabel('Time')
    plt.ylabel('Ratio of Likes to Impressions')
    plt.title('Ratio of Likes to Impressions Over Time')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15))
    # set the minimum and maximum values for Y-axis
    # plt.ylim(0, 0.1)
    plt.grid(True)
    plt.show()


def fit_polynomial(tweet_data, metric, degree=3):
    # tweet_data['timestamp'] = pd.to_datetime(tweet_data['timestamp'])
    tweet_data = tweet_data.sort_values('timestamp')

    # fit polynomial regression
    x = np.array(tweet_data['timestamp'].astype('int64') // 10 ** 9)
    y = np.array(tweet_data[metric])
    fit = Polynomial.fit(x, y, degree)
    prediction = fit(x)
    r2 = r2_score(y, prediction)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(tweet_data['timestamp'], y, marker='o', linestyle='-', label='Original Data')
    plt.plot(tweet_data['timestamp'], prediction, linestyle='--', color='r', label=f'Polynomial Fit\ndegree: {degree}, R² = {r2:.2f}')
    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))
    plt.xlabel('Time')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Over Time')
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=15))
    plt.legend()
    plt.grid(True)
    plt.show()


def fit_exponential(tweet_data, metric):
    # Normalize timestamps to prevent overflow with exp()
    x = (tweet_data['timestamp'] - tweet_data['timestamp'].min()).dt.total_seconds()
    y = np.array(tweet_data[metric])

    # Fit a linear model to the log-transformed data
    p = np.polyfit(x, np.log(y + 1e-10), 1)

    # print('p: ', p)

    # Convert the linear model back to exponential form
    a = np.exp(p[1])
    b = p[0]
    x_fitted = np.linspace(np.min(x), np.max(x), len(x))
    y_fitted = a * np.exp(b * x_fitted)

    r2 = r2_score(y, y_fitted)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(tweet_data['timestamp'], y, label='Original Data')
    plt.plot(tweet_data['timestamp'].min() + pd.to_timedelta(x_fitted, unit='s'),
             y_fitted, color='red',
             label=f'Fitted Exponential Curve\ny = {a:.4f} * e^({b:.4f} * x) (R² = {r2:.2f})')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))

    plt.xlabel('Timestamp')
    plt.ylabel(metric)
    plt.title('Exponential fit to tweet data')
    plt.legend()
    plt.grid(True)
    plt.show()

def fit_logarithmic(tweet_data, metric):
    # # Normalize timestamps to prevent overflow with exp()
    x = (tweet_data['timestamp'] - tweet_data['timestamp'].min()).dt.total_seconds()
    # x = np.array(tweet_data['timestamp'].astype('int64') // 10 ** 9)
    y = np.array(tweet_data[metric])

    # Fit a linear model to the log-transformed data
    p = np.polyfit(np.log(x + 0.000000001), y, 1)

    # print('p: ', p)

    # Convert the linear model back to logarithmic form
    a = p[0]
    b = p[1]
    x_fitted = np.linspace(np.min(x), np.max(x), len(x))
    y_fitted = a * np.log(x_fitted + 0.000000001) + b

    r2 = r2_score(y, y_fitted)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(tweet_data['timestamp'], y, label='Original Data')
    plt.plot(tweet_data['timestamp'], y_fitted, color='red',
             label=f'Fitted Logarithmic Curve\ny = {a:.4f} * log(x) + {b:.4f} (R² = {r2:.2f})')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))

    plt.xlabel('Timestamp')
    plt.ylabel(metric)
    plt.title('Logarithmic fit to tweet data')
    plt.legend()
    plt.show()


def power_law_distribution(tweet_data, metric, bin_count=10):
    """
    I don't think this works properly
    :param tweet_data:
    :param metric:
    :param bin_count:
    :return:
    """

    # get the minimum and maximum values of the metric
    p_min = tweet_data[metric].min()
    if p_min <= 0:
        p_min = 1
    p_max = tweet_data[metric].max()

    # generate logarithmically spaced bins
    # bin_edges = np.logspace(np.log10(p_min), np.log10(p_max+1), num=bin_count + 1)

    # evenly spaced bins
    bin_edges = np.linspace(p_min, p_max, num=bin_count + 1)

    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Count the number of values in each bin
    bins, _ = np.histogram(tweet_data[metric], bins=bin_edges)

    # # # Print the bin edges, centers, and counts
    # print("Bin Edges:", bin_edges)
    # print("Bin Centers:", bin_centers)
    # print("Bin Counts:", bins)

    # x and y values for the power law fit
    x = np.arange(1, bin_count + 1)
    y = bins
    y_log = np.log(y + 1e-10)  # avoid log(0) by adding a small value
    x_log = np.log(x)

    # fit a power law curve to the data
    p = np.polyfit(x_log, y_log, 1)

    # get the parameters for the power law, (y = c * x^k)
    c = np.exp(p[1])
    k = p[0]

    # calculate the fitted values
    x_fitted = np.linspace(np.min(x), np.max(x), len(x))
    y_fitted = c * x_fitted ** k

    # calculate the R² value
    r2 = r2_score(y_log, np.log(y_fitted + 1e-10))

    # # Plot the histogram and the power law fit
    # plt.figure(figsize=(12, 8))
    # plt.bar(x, y, width=0.8, align='center', alpha=0.6, label='Data')
    # ax2 = plt.gca().twinx()
    # ax2.plot(x, y_fitted, 'r-',
    #          label=f'Power law fit\ny = {c:.4f} * x^{k:.4f} (R² = {r2:.4f})')
    # plt.xlabel('Bins')
    # plt.ylabel('Tweet frequency')
    # ax2.set_ylabel('Power Law Fit')
    # plt.title(f'Distribution of {metric} for Finnish tweets')
    # plt.xticks(x, [f'{round(bin_edges[i])} - {round(bin_edges[i + 1])}' for i in range(len(bin_centers))], rotation=45)
    # plt.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot histogram on ax1
    ax1.bar(x, y, width=0.8, align='center', alpha=0.6, label='Data')
    ax1.set_xlabel('Bins')
    ax1.set_ylabel('Tweet frequency')
    ax1.set_title(f'Distribution of {metric} for Finnish tweets')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{round(bin_edges[i])} - {round(bin_edges[i + 1])}' for i in range(len(bin_centers))],
                        rotation=45)

    # Plot power law fit on ax2
    ax2 = ax1.twinx()
    ax2.plot(x_fitted, y_fitted, 'r-', label=f'Power law fit\ny = {c:.4f} * x^{k:.4f} (R² = {r2:.4f})')
    ax2.set_ylabel('Power Law Fit')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()
    plt.tight_layout()
    plt.show()


def plot_comparison(original_data, interpolated_data, metric='impression_count'):
    """
    Plots the original and interpolated data for comparison.

    Parameters:
    - original_df: The original DataFrame with time series data.
    - resampled_df: The resampled DataFrame with time series data.
    - title: Title of the plot.
    """

    # align the first data point
    offset = original_data['timestamp'][0] - interpolated_data['timestamp'][0]

    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(original_data['timestamp'], original_data[metric], label=f'Original {metric}', marker='o',
             linestyle='-', color='blue')

    # Plot resampled data
    plt.plot((interpolated_data['timestamp'] + offset), interpolated_data[metric], label=f'Interpolated {metric}',
             marker='x', linestyle='', color='red')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))
    # plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    # Adding labels and title
    plt.title(f'{metric.replace("_", " ").title()}: Original vs Interpolated Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()
    # plt.xticks(rotation=45)

    # Set y-axis to only show integer values
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_infection(trends, title):
    healthy = trends[0]['trends']['node_count'][0]
    infected = trends[0]['trends']['node_count'][1]

    plt.plot(healthy, label='Healthy')
    plt.plot(infected, label='Infected')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Nodes')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_infection_to_data(trends, tweet_data, data_type, plot=True, title=None):
    # calculates the mean squared error
    """
    [{'trends': {
    'node_count': {
    0: [950, 947, 943, 937, 929, 923, 915, 908, 902, 893],  # healthy
    1: [50, 53, 57, 63, 71, 77, 85, 92, 98, 107]},  # infected
     'status_delta': {
    0: [0, -3, -4, -6, -8, -6, -8, -7, -6, -9],
    1: [0, 3, 4, 6, 8, 6, 8, 7, 6, 9]}}}]
    """
    healthy = trends[0]['trends']['node_count'][0]
    infected = trends[0]['trends']['node_count'][1]

    score = mean_squared_error(tweet_data[data_type], infected)

    if not plot:
        return score

    plt.figure(figsize=(10, 6))
    # plt.plot(healthy, label='Healthy')
    plt.plot(infected, label='Infected')
    plt.plot(tweet_data[data_type], label=data_type)
    plt.xlabel('Hours')
    plt.ylabel('Number of Nodes')
    # plt.text(20, 0, f'Mean Squared Error: {error:2f}', bbox=dict(facecolor='white', alpha=0.5))
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return score

def get_barabasi_albert_graph(name, nodes, connections, seed):
    file_name = f'saved_graphs/barabasi_albert_graph_{name}_{nodes}_{connections}_{seed}.pkl'
    g = load_graph(file_name)
    if not g:
        print('Generating a new Barabási-Albert graph')
        g = nx.barabasi_albert_graph(nodes, connections, seed)
        save_graph(g, file_name)
    return g

def get_erdos_renyi_graph(name, nodes, probability, seed):
    file_name = f'saved_graphs/erdos_renyi_graph_{name}_{nodes}_{(str)(probability).replace(".", "")}_{seed}.pkl'
    g = load_graph(file_name)
    if not g:
        print('Generating a new Erdos-Renyi graph')
        g = nx.erdos_renyi_graph(nodes, probability, seed)
        save_graph(g, file_name)
    return g

def load_graph(file_name):
    """
    This function loads a graph from a .pkl file.
    """
    try:
        with open(file_name, 'rb') as f:
            g = pickle.load(f)
        return g
    except FileNotFoundError as e:
        return None


def save_graph(graph, file_name):
    """
    This function saves a graph as a .pkl file.
    """
    if not os.path.exists(file_name):
        # save to a file
        with open(file_name, 'wb') as f:
            pickle.dump(graph, f)
        print(f'Saved the graph as a file: {file_name}')
        return True
    else:
        print(f'File with the name {file_name} already exists.')
        # graph already exists
        return False


def run_SIS_simulation(graph, tweet_data, beta, lambda_, initial_fraction_infected, data_type, plot=False, return_MSE=True):
    # Model configuration
    model = ep.SISModel(graph)
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('lambda', lambda_)
    config.add_model_parameter('fraction_infected', initial_fraction_infected)  # initial fraction of infected
    model.set_initial_status(config)

    title = f'SIS simulation of {data_type} with parameters b={beta:.3f}, l={lambda_:.2f}'

    return run_simulation_placeholder(model, tweet_data, data_type, title, plot, return_MSE)

def run_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type, plot=False, return_MSE=True):
    # Model configuration
    model = ep.SIRModel(graph)
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_parameter('fraction_infected', initial_fraction_infected)  # initial fraction of infected
    model.set_initial_status(config)

    title = f'SIR simulation of {data_type} with parameters b={beta:.3f}, l={gamma:.2f}'

    return run_simulation_placeholder(model, tweet_data, data_type, title, plot, return_MSE)


def run_simulation_placeholder(model, tweet_data, data_type, title, plot=False, return_MSE=True):
    # todo: remove run_simulation()
    # Run the simulation
    iterations = model.iteration_bunch(len(tweet_data))
    trends = model.build_trends(iterations)

    if return_MSE == "trends":  # im really sorry about this messy code
        return trends

    # Compare with real data and calculate similarity
    similarity = compare_infection_to_data(trends, tweet_data, plot=plot, data_type=data_type, title=title)

    if return_MSE:
        return similarity

    # return the sum of all nodes infected
    return sum(trends[0]['trends']['node_count'][1])


# def run_simulation(graph, tweet_data, beta, lambda_, gamma, initial_fraction_infected, infection_model, data_type, plot=False, return_MSE=True):
#     # TODO: replace this function with run_simulation_placeholder()
#     # Model configuration
#     config = mc.Configuration()
#     config.add_model_parameter('beta', beta)
#
#
#     config.add_model_parameter('fraction_infected', initial_fraction_infected)  # initial fraction of infected
#
#     if infection_model.upper() == 'SIR':
#         config.add_model_parameter('gamma', gamma)
#         model = ep.SIRModel(graph)
#
#     elif infection_model.upper() == 'SIS':
#         config.add_model_parameter('lambda', lambda_)
#         model = ep.SISModel(graph)
#
#     model.set_initial_status(config)
#
#     # Run the simulation
#     iterations = model.iteration_bunch(len(tweet_data))
#     trends = model.build_trends(iterations)
#
#     # Compare with real data and calculate similarity
#     similarity = compare_infection_to_data(trends, tweet_data, plot=plot, data_type=data_type, title=f'{infection_model} simulation of {data_type} with parameters b={beta}, l={lambda_}, g={gamma}')
#
#     if return_MSE:
#         return similarity
#
#     # return the sum of all nodes infected
#     return sum(trends[0]['trends']['node_count'][1])



def run_time_dependent_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type, plot=False, return_MSE=True):
    """
    SIR simulation where the infection rate is lower during the night
    :param graph:
    :param tweet_data:
    :param beta:
    :param gamma:
    :param initial_fraction_infected:
    :param data_type:
    :param plot:
    :param return_MSE:
    :return:
    """
    night_time_multiplier = 0.1

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_parameter('fraction_infected', initial_fraction_infected)  # initial fraction of infected
    model = ep.SIRModel(graph)
    model.set_initial_status(config)


    # get the starting time for the simulation
    current_time = pd.to_datetime(tweet_data['timestamp'].min())
    increment_time = pd.Timedelta(hours=1)

    # defining the beta (infection rate) function
    def get_beta():
        # lower infection rate at night
        if current_time.hour >= 22 or current_time.hour <= 5:
            return beta * night_time_multiplier
        # normal infection rate during the day
        return beta


    iterations = []
    # Run the simulation
    for _ in range(len(tweet_data)):
        # directly modify the beta value of the model object
        model.params["model"]["beta"] = get_beta()
        iterations.append(model.iteration())
        current_time += increment_time

    # iterations = model.iteration_bunch(len(tweet_data))
    trends = model.build_trends(iterations)

    title = f'Time dependent SIR simulation of {data_type} with parameters b={beta:.2f}, b (night)={beta*night_time_multiplier:.3f}, g={gamma:.3f}'

    # Compare with real data and calculate similarity
    similarity = compare_infection_to_data(trends, tweet_data, plot=plot, data_type=data_type, title=title)

    if return_MSE:
        return similarity

    # return the sum of all nodes infected
    return sum(trends[0]['trends']['node_count'][1])

def find_best_SIR_parameters(graph, tweet_data, tweet_id, beta_range, gamma_range, initial_fraction_infected, data_type, save_results=True, display_results=False):
    ### simulate all parameter combinations ###
    # print(f'{datetime.datetime.now().strftime("%H:%M")}:'
    # f' Starting simulation with {len(beta_range) * len(gamma_range)} combinations.')
    param_combinations = []
    for i, beta in enumerate(beta_range):
        for gamma in gamma_range:
            # simulate the parameters

            """running a normal simulation"""
            score = run_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type, plot=False)
            """running the simulation on time dependant beta"""
            # score = run_time_dependent_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type, plot=False)

            param_combinations.append((beta, gamma, (int)(score)))
        # print(f'{datetime.datetime.now().strftime("%H:%M")}: Simulation {((i+1)/len(beta_range))*100:.0f} % complete.')

    df = pd.DataFrame(param_combinations, columns=['beta', 'gamma', 'score'])
    if save_results:
        ### save the results to a CSV file ###
        now = datetime.datetime.now().strftime('%d_%H_%M')
        id = tweet_id[-4:]
        file_name = f'simulation_results/simulation_results_SIR_{id}_{now}.csv'
        df.to_csv(file_name, index=False)
        print(f'Results saved to {file_name}')

    # Find the row with the minimum (best) score
    best_row = df.loc[df['score'].idxmin()]
    # Extract the parameters
    beta, gamma, score = best_row['beta'], best_row['gamma'], best_row['score']
    print(f'MSE of the best SIR simulation with parameters b,g=({beta:.3f}, {gamma:.3f}): ', score)

    if display_results:
        # display the graph of the best simulation
        score = run_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type, plot=True)

    return (beta, gamma)


def find_best_SIS_parameters(graph, tweet_data, tweet_id, beta_range, lambda_range, initial_fraction_infected, data_type, save_results=True, display_results=False):
    ### simulate all parameter combinations ###
    # print(f'{datetime.datetime.now().strftime("%H:%M")}:'
    #       f' Starting simulation with {len(beta_range) * len(lambda_range)} combinations.')
    param_combinations = []
    for i, beta in enumerate(beta_range):
        for lambda_ in lambda_range:
            # simulate the parameters
            score = run_SIS_simulation(graph, tweet_data, beta, lambda_, initial_fraction_infected, data_type, plot=False, return_MSE=True)
            param_combinations.append((beta, lambda_, (int)(score)))
        # print(f"{datetime.datetime.now().strftime('%H:%M')}: Simulation {((i + 1) / len(beta_range)) * 100:.0f} % complete.")

    df = pd.DataFrame(param_combinations, columns=['beta', 'lambda', 'score'])
    if save_results:
        ### save the results to a CSV file ###
        now = datetime.datetime.now().strftime('%d_%H_%M')
        id = tweet_id[-4:]
        file_name = f'simulation_results/simulation_results_SIS_{id}_{now}.csv'
        df.to_csv(file_name, index=False)
        print(f'Results saved to {file_name}')

    # Find the row with the minimum score
    best_row = df.loc[df['score'].idxmin()]
    # Extract the parameters
    beta, lambda_, score = best_row['beta'], best_row['lambda'], best_row['score']
    print(f'MSE of the best simulation with parameters b,g,l=({beta:.3f}, {lambda_:.3f}): ', score)

    if display_results:
        ### display the result of the best simulation ###

        # score = run_simulation(graph, tweet_data, beta, lambda_, gamma, initial_fraction_infected, plot=True, infection_model=model, data_type=data_type)
        score = run_SIS_simulation(graph, tweet_data, beta, lambda_, initial_fraction_infected, data_type,
                                   plot=True, return_MSE=True)
        # print(f'MSE of the best simulation with parameters b,g,l=({beta}, {lambda_}, {gamma}): ', score)

    return (beta, lambda_)


# def model_infection(graph, tweet_data, tweet_id, beta_range, lambda_range, gamma_range, initial_fraction_infected, model, data_type, save_results=True, display_results=False):
#     """
#     :param graph:
#     :param tweet_data:
#     :param tweet_id:
#     :param beta_range: np.arange(0.1, 0.05, -0.01)  # Infection
#     :param lambda_range: np.arange(0.1, 0.05, -0.01)  # Recovery
#     :param gamma_range: np.arange(0.1, 0.05, -0.05)  # Removal
#     :param initial_fraction_infected:
#     """
#
#     ### simulate all parameter combinations ###
#     param_combinations = []
#     if model.upper() == 'SIR':
#         # print(f'{datetime.datetime.now().strftime("%H:%M")}:'
#               # f' Starting simulation with {len(beta_range) * len(gamma_range)} combinations.')
#         for i, beta in enumerate(beta_range):
#             for gamma in gamma_range:
#                 # simulate the parameters
#
#                 """running a normal simulation"""
#                 score = run_simulation(graph, tweet_data, beta, 0, gamma, initial_fraction_infected, plot=False, infection_model=model, data_type=data_type)
#                 """running the simulation on time dependant beta"""
#                 # score = run_time_dependent_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type, plot=False)
#
#
#                 param_combinations.append((beta, 0, gamma, (int)(score)))
#             # print(f'{datetime.datetime.now().strftime("%H:%M")}: Simulation {((i+1)/len(beta_range))*100:.0f} % complete.')
#     elif model.upper() == 'SIS':
#         print(f'{datetime.datetime.now().strftime("%H:%M")}:'
#               f' Starting simulation with {len(beta_range) * len(lambda_range)} combinations.')
#         for i, beta in enumerate(beta_range):
#             for lambda_ in lambda_range:
#                 # simulate the parameters
#                 score = run_simulation(graph, tweet_data, beta, lambda_, 0, initial_fraction_infected, plot=False, infection_model=model, data_type=data_type)
#                 param_combinations.append((beta, lambda_, 0, (int)(score)))
#             print(f"{datetime.datetime.now().strftime('%H:%M')}: Simulation {((i+1)/len(beta_range))*100:.0f} % complete.")
#
#     df = pd.DataFrame(param_combinations, columns=['beta', 'lambda', 'gamma', 'score'])
#     if save_results:
#         ### save the results to a CSV file ###
#         now = datetime.datetime.now().strftime('%d_%H_%M')
#         id = tweet_id[-4:]
#         file_name = f'simulation_results/simulation_results_{model}_{id}_{now}.csv'
#         df.to_csv(file_name, index=False)
#         print(f'Results saved to {file_name}')
#
#     # Find the row with the minimum score
#     best_row = df.loc[df['score'].idxmin()]
#     # Extract the parameters
#     beta, lambda_, gamma, score = best_row['beta'], best_row['lambda'], best_row['gamma'], best_row['score']
#     print(f'MSE of the best simulation with parameters b,g,l=({beta:.2f}, {lambda_:.2f}, {gamma:.2f}): ', score)
#
#     if display_results:
#         ### display the result of the best simulation ###
#
#         # score = run_simulation(graph, tweet_data, beta, lambda_, gamma, initial_fraction_infected, plot=True, infection_model=model, data_type=data_type)
#         score = run_SIR_simulation(graph, tweet_data, beta, gamma, initial_fraction_infected, data_type=data_type, plot=True)
#         # print(f'MSE of the best simulation with parameters b,g,l=({beta}, {lambda_}, {gamma}): ', score)
#
#     return (beta, lambda_, gamma)




def get_max_values(data_type, tweets):
    # power law distribution of impression count for all tweets
    interpolated_data_all = []
    for id in tweets:
        # load data
        d = interpolate_data(load_csv(id), '1h')
        # save result
        interpolated_data_all.append(d)

    max_values = pd.DataFrame({data_type: [0] * len(interpolated_data_all)})
    # create a list of the largest values for each tweet
    for i, d in enumerate(interpolated_data_all):
        max_value = d[data_type].max()
        max_values.loc[i, data_type] = max_value

    max_values.reset_index(drop=True, inplace=True)
    print(max_values)

    return max_values


def main():
    # TODO: clean this mess of a main function

    # all_tweets = [1922689448447300082, 1923052071332217205, 1923080523703779830,
    #               1923352006707581422, 1923342619909701903, 1923477098141802909,
    #               1923743558680428995, 1923754485874119007, 1923793436068487574,
    #               1924088518910808080, 1924428162257047752, 1924437434521010605]

    # all_tweets = [1922689448447300082, 1923052071332217205, 1923080523703779830,
    #               1923352006707581422, 1923342619909701903, 1923477098141802909,
    #               1923743558680428995, 1923754485874119007,
    #               1924088518910808080, 1924428162257047752, 1924437434521010605]

    # usa_tweets = [1923080523703779830, 1923342619909701903, 1923477098141802909, # exclude trump because numbers too high
    #               1923743558680428995, 1923754485874119007]

    fi_tweets = [1924437434521010605, 1926997808369840418, 1927003736154554560,  # gaza, 0-5
                 1927013798415585709, 1927429107190551035, 1927475122765430833,

                 1922689448447300082, 1923052071332217205, 1923352006707581422, # non gaza 6-12
                 1924088518910808080, 1924428162257047752, 1927292506988740922,
                 1927278416195170648]


    gaza_tweets = [1924437434521010605, 1926997808369840418, 1927003736154554560,
                   1927013798415585709, 1927429107190551035, 1927475122765430833]

    non_gaza_tweets = [1922689448447300082, 1923052071332217205, 1923352006707581422,
                       1924088518910808080, 1924428162257047752, 1927292506988740922,
                       1927278416195170648]
    # tweet_id = '1923080523703779830'  # Sen, Ted Cruz (R-Tex.) has been a major proponent of the idea
    # tweet_id = '1923342619909701903'  # Chip Roy: Why there’s a problem.
    # tweet_id = '1923477098141802909'  # Mike Collins: So it’s legal for a president to ship millions of illegal aliens into
    # tweet_id = '1923743558680428995'  # Claudia Tenney: The numbers don't lie, President Trump's America First economic plan is working!
    # tweet_id = '1923754485874119007'  # Bernie Sanders: 68,000 Americans already die every year because they don’t have access to the health care...
    # tweet_id = '1923793436068487574'  # Trump: 🇦🇪🇺🇸
    # tweet_id = '1924127364780261454'  # Alice Weidel: Bielefeld: Ein Syrer verletzt 5 Menschen zum Teil schwer.

    tweet_id = '1924437434521010605'  # POrpo: Gaza Orpo: Gazan siviilien kärsimyksen on loputtava. Suomi vaatii Israelia...
    # tweet_id = '1926997808369840418'  # VHonk: Pal
    # tweet_id = '1927003736154554560'  # STynk: Pal
    # tweet_id = '1927013798415585709'  # LiAnd: Pal
    # tweet_id = '1927429107190551035'  # TSamm: Gaza
    # tweet_id = '1927475122765430833'  # AKale: Gaza

    # tweet_id = '1922689448447300082'  # P.TOVERI Länsi ei ole sodassa Venäjän kanssa. Unohdettiin ilmeisesti kertoa...
    # tweet_id = '1923052071332217205'  # Purra: Edustaja Bergbom totesi osuvasti, että kyselytunti oli taas...
    # tweet_id = '1923352006707581422'  # Purra: Monilapsisissa maahanmuuttajaperheissä summa voi nousta lähemmäs viittäkin tonnia
    # tweet_id = '1924088518910808080'  # Minja Koskela: Olin tänään monen muun tavoin marssimassa Epäluottamuslause-mielenosoituksessa
    # tweet_id = '1924428162257047752'  # Haavisto: Maanantaiaamuna oli mahdollisuus käydä korkealaatuista keskustelua nuorten tilanteesta...
    # tweet_id = '1927292506988740922'  # PSuom: mm
    # tweet_id = '1927278416195170648'  # TSamm: mm

    # load and resample data
    tweet_data = load_csv(tweet_id)
    interpolated_data = interpolate_data(tweet_data, '1h')

    # plot_comparison(tweet_data, interpolated_data, 'impression_count')

    ### plotting metrics over time
    # plot_metric_over_time(interpolated_data, 'impression_change')
    # plot_metric_over_time(tweet_data, 'like_count')
    # plot_metric_over_time(interpolated_data, 'impression_count')
    # plot_metric_over_time(tweet_data, 'impression_count', title='Li Andersson, Gaza tweet: Impression Count')

    ### plotting the ratio of likes to impressions
    # plot_ratio_over_time(tweet_data)

    ## plotting the polynomial fits
    # fit_polynomial(interpolated_data, 'impression_count', degree=3)
    # fit_polynomial(interpolated_data, 'share_count', degree=3)
    # fit_polynomial(interpolated_data, 'like_count', degree=3)

    ## plotting the exponential fits
    # fit_exponential(interpolated_data, 'impression_count')

    ### plotting the power law fit
    # power_law_distribution(get_max_values('impression_count', fi_tweets), 'impression_count', bin_count=6)
    # power_law_distribution(get_max_values('share_count', fi_tweets), 'share_count', bin_count=6)
    # power_law_distribution(get_max_values('retweet_count', fi_tweets), 'retweet_count', bin_count=10)
    # follower_counts = {
    # "politician": [
    #     "Riikka Purra (PS)", "Sebastian Tynkkynen (Ps)", "Minja Koskela (Vas)", "Tere Sammallahti (Kok)",
    #     "Veronkia Honkasalo (Vas)", "Tere Sammallahti (Kok)", "Atte Kaleva (Kok)", "Perussuomalaiset",
    #     "Riikka Purra (PS)", "Petteri Orpo (Kok)", "Pekka Toveri (Kok)", "Li Andersson (Vas)", "Pekka Haavisto (Vihr)"
    # ],
    # "follower_count": [
    #     89500, 51400, 24900, 9326, 25900, 9326, 35300, 43600, 89500, 140000, 103400, 164700, 195100
    # ]
    # }
    # follower_counts = pd.DataFrame(follower_counts)
    # print(follower_counts)
    # power_law_distribution(follower_counts, 'follower_count', bin_count=10)



    # Population defined as 100 * the largest number of shares observer in a country
    # Here we will count both retweets and quotes as shares
    population_finland = 324 * 100
    population_usa = 2384 * 100
    minimum_infected = 1 / population_finland



    # Barabási-Albert-graph
    g_fi_ba = get_barabasi_albert_graph('fi', population_finland, 3, seed=2025)
    # g_fi_er = get_erdos_renyi_graph("fi", population_finland, 0.01, seed=2025)
    # g_fi_er = get_erdos_renyi_graph("fi", population_finland, 0.0000926, seed=2025)  # Average degree 3: 3/32399=0.0000926
    # g_fi_er = get_erdos_renyi_graph('fi', population_finland, 0.003, seed=2025)

    """ run many simulations to find optimal parameters """
    # SIR
    # initial_fraction_infected = interpolated_data['impression_change'][1] / population_finland / 4
    # if initial_fraction_infected < minimum_infected:
    #     initial_fraction_infected = minimum_infected
    # beta_range = np.arange(0.002, 0.035, 0.002)  # Infection
    # gamma_range = np.arange(0.05, 0.25, 0.01)  # Removal, SIR
    # beta, gamma = find_best_SIR_parameters(g_fi_ba, interpolated_data, tweet_id, beta_range, gamma_range, initial_fraction_infected, 'impression_change', True, True)
    #
    # print(beta, gamma)
    # return

    # SIS
    # initial_fraction_infected = interpolated_data['share_count'][1] / population_finland / 4
    # if initial_fraction_infected < minimum_infected:
    #     initial_fraction_infected = minimum_infected
    # initial_fraction_infected = 0.005
    # beta_range = np.arange(0.002, 0.024, 0.002)  # Infection
    # lambda_range = np.arange(0.01, 0.15, 0.01)  # Recovery, SIS
    # beta, lambda_ = find_best_SIS_parameters(g_fi_ba, interpolated_data, tweet_id, beta_range, lambda_range, initial_fraction_infected , 'share_count', True, True)

    # beta=0.06  # time dependent sir
    # gamma=0.23
    #
    # # run the simulation with the learnt optimal values
    # result = run_time_dependent_SIR_simulation(g_fi_ba, interpolated_data, beta, gamma,
    #                         initial_fraction_infected, data_type='impression_change', plot=True, return_MSE=False)
    # initial_fraction_infected = 0.1

    # print(result)
    # beta=0.032  # normal sir
    # gamma=0.23
    # initial_fraction_infected = interpolated_data['impression_change'][1] / population_finland / 4
    # # run the simulation with the learnt optimal values
    # result = run_SIR_simulation(g_fi_ba, interpolated_data, beta, gamma,
    #                         initial_fraction_infected, data_type='impression_change', plot=True, return_MSE=False)
    # print(result)

    # beta=0.012
    # lambda_=0.1
    # # initial_fraction_infected = interpolated_data['share_count'][1] / population_finland / 4
    # initial_fraction_infected = 0.005
    #
    # # Get the last row of the DataFrame
    # last_row = interpolated_data.tail(1)
    # # Extend the last row by repeating it 24 times
    # extended_rows = pd.concat([last_row] * 48, ignore_index=True)
    # # Append the extended rows to the original DataFrame
    # d_extended = pd.concat([interpolated_data, extended_rows], ignore_index=True)
    #
    # if initial_fraction_infected < minimum_infected:
    #     initial_fraction_infected = minimum_infected
    # result = run_SIS_simulation(g_fi_ba, d_extended, beta, lambda_,
    #                             initial_fraction_infected, data_type='share_count', plot=True, return_MSE=False)
    # print(result)
    #
    # return

    """running trough the simulations with the optimal average values"""

    # beta=0.02  # normal sir
    # gamma=0.156
    #
    # for id in fi_tweets:
    #     d = interpolate_data(load_csv(id))
    #     result = run_SIR_simulation(g_fi_ba, d, beta, gamma,
    #                             initial_fraction_infected, data_type='impression_change', plot=True, return_MSE=True)
    #     print(id, result)
    # return
    # run the simulation with the learnt optimal values

    """ SIR simulations """

    # beta_range = np.arange(0.002, 0.035, 0.002)  # Infection
    # gamma_range = np.arange(0.05, 0.25, 0.01)  # Removal, SIR
    #
    # datatype = "impression_change"
    # print("Running Normal SIR simulations for data type ", datatype)
    # # impressions gained in the first 15 minutes as a fraction of the total population
    # # initial_fraction_infected = interpolated_data['impression_change'][1] / population_finland / 4
    # # if initial_fraction_infected < minimum_infected:
    # #     initial_fraction_infected = minimum_infected
    #
    # # initial fraction infected is 10 %
    # initial_fraction_infected = 0.1
    #
    #
    # """iterate through all the fi tweets to find an average best values"""
    # optimal_values_fi = []  # (beta, gamma)
    # for id in fi_tweets:
    #     d = interpolate_data(load_csv(id))
    #     print(f"{datetime.datetime.now().strftime('%H:%M')} Staring analysis on tweet: ", id)
    #     beta, gamma = find_best_SIR_parameters(g_fi_ba, d, tweet_id, beta_range, gamma_range, initial_fraction_infected, datatype, True, False)
    #     optimal_values_fi.append((beta, gamma))
    #
    # print("Optimal values for all fi tweets: ", optimal_values_fi)
    # beta_values = [value[0] for value in optimal_values_fi]
    # beta_avg = np.mean(beta_values)
    # gamma_values = [value[1] for value in optimal_values_fi]
    # gamma_avg = np.mean(gamma_values)
    # optimal_avg_values_fi = (beta_avg, gamma_avg)
    #
    # print("Optimal values for ALL finnish tweets: ", optimal_avg_values_fi)
    #
    #
    # """iterate through all the tweets to find an average best values GAZA"""
    # optimal_values_gaza = optimal_values_fi[0:6]
    # beta_values = [value[0] for value in optimal_values_gaza]
    # beta_avg = np.mean(beta_values)
    # gamma_values = [value[1] for value in optimal_values_gaza]
    # gamma_avg = np.mean(gamma_values)
    # optimal_avg_values_gaza = (beta_avg, gamma_avg)
    # print("Optimal average values for gaza tweets: ", optimal_avg_values_gaza)
    #
    # """iterate through all the tweets to find an average best values NON GAZA"""
    # optimal_values_non_gaza = optimal_values_fi[6:]
    # beta_values = [value[0] for value in optimal_values_non_gaza]
    # beta_avg = np.mean(beta_values)
    # gamma_values = [value[1] for value in optimal_values_non_gaza]
    # gamma_avg = np.mean(gamma_values)
    # optimal_avg_values_non_gaza = (beta_avg, gamma_avg)
    # print("Optimal average values for non gaza tweets: ", optimal_avg_values_non_gaza)


    """ SIS Simulations """

    # print("\nbeginning the SIS simulations...\n")
    #
    # beta_range = np.arange(0.001, 0.024, 0.001)  # Infection
    # lambda_range = np.arange(0.01, 0.15, 0.01)  # Recovery, SIS
    #
    # datatype = "share_count"
    # print("Running Normal SIS simulations for data type ", datatype)
    # # # impressions gained in the first 15 minutes as a fraction of the total population
    # # initial_fraction_infected = interpolated_data['share_count'][1] / population_finland / 4
    # # if initial_fraction_infected < minimum_infected:
    # #     initial_fraction_infected = minimum_infected
    # # fraction infected 0.5 %
    # initial_fraction_infected = 0.005
    #
    #
    # """iterate through all the fi tweets to find an average best values"""
    # optimal_values_fi = []  # (beta, gamma)
    # for id in fi_tweets:
    #     d = interpolate_data(load_csv(id))
    #     print(f"{datetime.datetime.now().strftime('%H:%M')} Staring analysis on tweet: ", id)
    #     beta, gamma = find_best_SIS_parameters(g_fi_ba, d, tweet_id, beta_range, lambda_range, initial_fraction_infected, datatype, True, False)
    #     optimal_values_fi.append((beta, gamma))
    #
    # # Formatting the output
    # formatted_values_fi = [(round(float(x), 3), round(float(y), 3)) for x, y in optimal_values_fi]
    # print("Optimal values for all fi tweets: ", formatted_values_fi)
    #
    # beta_values = [value[0] for value in optimal_values_fi]
    # beta_avg = np.mean(beta_values)
    # gamma_values = [value[1] for value in optimal_values_fi]
    # gamma_avg = np.mean(gamma_values)
    # # optimal_avg_values_fi = (beta_avg, gamma_avg)
    # # print("Optimal average values for finnish tweets: ", optimal_avg_values_fi)
    # # Format the average values
    # optimal_avg_values_fi = (round(float(beta_avg), 3), round(float(gamma_avg), 3))
    # print("Average optimal values for fi tweets: ", optimal_avg_values_fi)
    #
    # """iterate through all the tweets to find an average best values GAZA"""
    # optimal_values_gaza = optimal_values_fi[0:6]
    # beta_values = [value[0] for value in optimal_values_gaza]
    # beta_avg = np.mean(beta_values)
    # gamma_values = [value[1] for value in optimal_values_gaza]
    # gamma_avg = np.mean(gamma_values)
    # # optimal_avg_values_gaza = (beta_avg, gamma_avg)
    # # print("Optimal values for gaza tweets: ", optimal_avg_values_gaza)
    # optimal_avg_values_gaza = (round(float(beta_avg), 3), round(float(gamma_avg), 3))
    # print("Average optimal values for gaza tweets: ", optimal_avg_values_gaza)
    #
    # """iterate through all the tweets to find an average best values NON GAZA"""
    # optimal_values_non_gaza = optimal_values_fi[6:]
    # beta_values = [value[0] for value in optimal_values_non_gaza]
    # beta_avg = np.mean(beta_values)
    # gamma_values = [value[1] for value in optimal_values_non_gaza]
    # gamma_avg = np.mean(gamma_values)
    # # optimal_avg_values_non_gaza = (beta_avg, gamma_avg)
    # # print("Optimal values for non gaza tweets: ", optimal_avg_values_non_gaza)
    # optimal_avg_values_non_gaza = (round(float(beta_avg), 3), round(float(gamma_avg), 3))
    # print("Average optimal values for non-gaza tweets: ", optimal_avg_values_non_gaza)

    def compare_trends(trends, title=None):
        """
        Compares a list of trends and displays them in a graph.

        Parameters:
        - trends: List of trends, where each trend is a dictionary containing 'node_count'.
        - plot: Boolean indicating whether to plot the trends.
        - title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))

        for i, trend in enumerate(trends):
            infected = trend[0]['trends']['node_count'][1]

            plt.plot(infected, label=f'Simulation {i + 1}')

        plt.xlabel('Hours')
        plt.ylabel('Number of Nodes Infected')
        plt.title(title if title else 'Trends')
        plt.legend()
        plt.grid(True)
        plt.show()

    """showing the results of 10 SIS simulations in one graph"""
    # trends = []
    # beta = 0.008
    # lambda_ = 0.077
    # fraction_infected = 0.005
    # data_type = 'share_count'
    # for _ in range(10):
    #     trends.append(run_SIS_simulation(g_fi_ba, interpolated_data, beta, lambda_, fraction_infected, data_type, False, "trends"))
    #
    # compare_trends(trends, f'10 SIS simulations with the average best parameters for all tweets b={beta:.3f}, l={lambda_:.2f}')

    """showing the results of 5 SIR simulations in one graph"""
    # trends = []
    # beta = 0.032
    # gamma = 0.23
    # fraction_infected = interpolated_data['impression_change'][1] / population_finland / 4
    # data_type = 'impression_change'
    # for _ in range(5):
    #     trends.append(run_SIR_simulation(g_fi_ba, interpolated_data, beta, gamma, fraction_infected, data_type, False,
    #                                      "trends"))
    # compare_trends(trends, f'5 SIR simulations with the best parameters for Tweet 1 b={beta:.3f}, g={gamma:.2f}')


    """Show the original data for all tweets in one graph"""
    def plot_all_tweets(all_tweets, data_type):
        """
        Plots the values from a specified data_type column across multiple DataFrames.

        Parameters:
        - all_tweets: List of Pandas DataFrames.
        - data_type: The column name to plot values from.
        - plot: Boolean indicating whether to plot the trends.
        - title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))

        for i, df in enumerate(all_tweets):
            if data_type in df.columns:
                # Extract the values from the specified column
                values = df[data_type].values

                # Plot the values for the current DataFrame
                plt.plot(values, label=f'Tweet {i + 1}')

        plt.xlabel('Index')
        plt.ylabel(data_type)
        plt.title(f'Values of {data_type} Across All Tweets')
        plt.legend()
        plt.grid(True)
        plt.show()

    tweets = []
    for tweet in fi_tweets:
        tweets.append(interpolate_data(load_csv(tweet)))
    plot_all_tweets(tweets, 'bookmark_count')


    """some other stuff"""

    # 0.007,0.01,0.0
    # 0.005,0.01,0.0
    # 0.013000000000000001,0.08999999999999998,0.0
    # beta = 0.037  # good SIS values
    # lambda_ = 0.049
    # gamma = 0

    # 0.01, 0.005, 0.019
    # beta = 0.05  # good SIR values
    # lambda_ = 0
    # gamma = 0.09
    ###
    # beta = 0.03  # best SIR values
    # lambda_ = 0
    # gamma = 0.08

    # beta = 0.03 # best SIR values PURRA for impression_change
    # lambda_ = 0
    # gamma = 0.1

    # beta = 0.004 # best SIR values PURRA
    # lambda_ = 0
    # gamma = 0.13

    # beta = 0.03 # best SIR values ORPO
    # lambda_ = 0
    # gamma = 0.19
    #
    # beta = 0.06 # best SIR values TOVERI
    # lambda_ = 0
    # gamma = 0.25
    #
    # beta = 0.13 #
    # lambda_ = 0
    # gamma = 0.17

    # result = run_SIS_simulation(g_fi_ba, interpolated_data, beta, lambda_, initial_fraction_infected,
    #                         data_type='impression_count', plot=True, return_MSE=False)
    #
    # result = run_SIR_simulation(g_fi_ba, interpolated_data, beta, gamma, initial_fraction_infected,
    #                         data_type='impression_change', plot=True, return_MSE=False)
    #
    # print("Total impressions:", interpolated_data['impression_count'].iloc[-1])
    # print("Impressions predicted by model:", result)
    # print(f"Difference:", result - interpolated_data['impression_count'].iloc[-1])

    beta = 0.085 # best time dependent SIR values PURRA, 21-5: 5 % infection rate
    lambda_ = 0
    gamma = 0.12

    # result = run_time_dependent_SIR_simulation(g_fi_ba, interpolated_data, beta, gamma,
    #                         initial_fraction_infected, data_type='impression_change', plot=True, return_MSE=False)
    #
    # print("Total impressions:", interpolated_data['impression_count'].iloc[-1])
    # print("Impressions predicted by model:", result)
    # print(f"Difference:", result - interpolated_data['impression_count'].iloc[-1])


    """
    28,39,289,6,3,3954,2025-05-16 15:58:04; 1923352006707581422

    
    """



    """
    # model selection
    model = ep.SIRModel(g_fi)
    # model configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', 0.005)  # Infection probability
    config.add_model_parameter('lambda', 0.01)  # Recovery probability
    config.add_model_parameter('gamma', 0.0)  # Removal probability
    config.add_model_parameter("fraction_infected", initial_fraction_infected)  # initial fraction of infected
    # Set the most central nodes as the initial status
    # initial_status = {node: 1 if node in top_central_nodes else 0 for node in g_fi.nodes()}
    # model.set_initial_status(initial_status)
    model.set_initial_status(config)

    # Simulation
    # iterations = model.iteration_bunch(24*3)
    iterations = model.iteration_bunch(len(interpolated_data))
    trends = model.build_trends(iterations)
    # plot_infection(trends, 'SIS Model Simulation')
    similarity = compare_infection_to_data(trends, interpolated_data, plot=True,
                                           title='Simulation Compared to Real Data')
    print("score: ", (int)(similarity))
    # print('end:', datetime.datetime.now())
    # print(similarity)
    """

    """
    [{'trends': {
    'node_count': {
    0: [950, 947, 943, 937, 929, 923, 915, 908, 902, 893],  # healthy
    1: [50, 53, 57, 63, 71, 77, 85, 92, 98, 107]},  # infected
     'status_delta': {
    0: [0, -3, -4, -6, -8, -6, -8, -7, -6, -9],
    1: [0, 3, 4, 6, 8, 6, 8, 7, 6, 9]}}}]
    """

if __name__ == "__main__":
    main()



"""
max impressions, shares
suomi:
# 195018, 324  tweet_id = '1923352006707581422'  # Purra: Monilapsisissa maahanmuuttajaperheissä summa voi nousta lähemmäs viittäkin tonnia
# 39681, 50 tweet_id = '1924088518910808080'  # Minja Koskela: Olin tänään monen muun tavoin marssimassa Epäluottamuslause-mielenosoituksessa
# 39393, 50 tweet_id = '1924437434521010605'  # Orpo: Orpo: Gazan siviilien kärsimyksen on loputtava. Suomi vaatii Israelia...
# 30717, 75 tweet_id = '1923052071332217205'  # Purra: Edustaja Bergbom totesi osuvasti, että kyselytunti oli taas...
# 19542, 25 tweet_id = '1922689448447300082'  # P.TOVERI Länsi ei ole sodassa Venäjän kanssa. Unohdettiin ilmeisesti kertoa...
# 2880, 0 tweet_id = '1924428162257047752'  # Haavisto: Maanantaiaamuna oli mahdollisuus käydä korkealaatuista keskustelua nuorten tilanteesta...
usa:
# 1615566, 23840 tweet_id = '1923477098141802909'  # Mike Collins: So it’s legal for a president to ship millions of illegal aliens into
# 190631, 21954 tweet_id = '1923754485874119007'  # Bernie Sanders: 68,000 Americans already die every year because they don’t have access to the health care...
# 224253, 376 tweet_id = '1923342619909701903'  # Chip Roy: Why there’s a problem.
# 43672, 42 tweet_id = '1923080523703779830'  # Sen, Ted Cruz (R-Tex.) has been a major proponent of the idea
# 1387, 18 tweet_id = '1923743558680428995'  # Claudia Tenney: The numbers don't lie, President Trump's America First economic plan is working!
globaali:
# 33952979, 28224 tweet_id = '1923793436068487574'  # Trump: 🇦🇪🇺🇸
saksa:
# 210790 tweet_id = '1924127364780261454'  # Alice Weidel: Bielefeld: Ein Syrer verletzt 5 Menschen zum Teil schwer.
"""