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
    tweet_data['timestamp'] = pd.to_datetime(tweet_data['timestamp'])
    # sort by date
    tweet_data = tweet_data.sort_values(by='timestamp')
    # Reset the index after sorting
    tweet_data.reset_index(drop=True, inplace=True)

    # add a 'share' count
    tweet_data['share_count'] = tweet_data['retweet_count'] + tweet_data['quote_count']

    return tweet_data


# def interpolating_data(tweet_data, frequency='1h'):
#     # interpolating the data to make the observations evenly spaced in time
#     # set index
#     tweet_data.set_index('timestamp', inplace=True)
#     # resample data
#     resampled_data = tweet_data.resample(frequency, on='timestamp').mean()
#     # fill NaN's from the data with interpolated values
#     resampled_data = resampled_data.interpolate(method='linear')
#     # reset index
#     resampled_data.reset_index(inplace=True)
#     tweet_data.reset_index(inplace=True)
#     return resampled_data

def interpolate_data(tweet_data, frequency='1h'):
    # interpolating the data to make the observations evenly spaced in time
    # the above method doesn't work as intended so this is attempt #2
    # limit the data to 3 days

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


    # limit to 3 days
    interpolated_data = interpolated_data.iloc[:3*24]

    # round every column down except for the index column (timestamp)
    interpolated_data = interpolated_data.apply(lambda x: np.round(x).astype(int))

    # Reset the index
    interpolated_data.reset_index(inplace=True)

    return interpolated_data


def plot_metric_over_time(tweet_data, metric):
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

    # Create a second y-axis for impression_count (number of people that have been shown the tweet)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Impressions', color='tab:orange')
    ax2.plot(tweet_data['timestamp'],
             tweet_data['impression_count'],
             marker='o', linestyle='-', color='tab:orange', label='Impressions')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))

    # Final formatting
    # fig.tight_layout()
    plt.title(f'{metric} and Impressions Over Time')
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
    p = np.polyfit(x, np.log(y), 1)

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
             label=f'Fitted Exponential Curve\ny = {a:.4f} * exp({b:.4f} * x) (R² = {r2:.2f})')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))

    plt.xlabel('Timestamp')
    plt.ylabel(metric)
    plt.title('Exponential fit to tweet data')
    plt.legend()
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
    p_min = tweet_data[metric].min()
    p_max = tweet_data[metric].max()

    # dividing the value range into bins
    k = (p_max - p_min) / bin_count
    bin_ranges = []
    for i in range(bin_count):
        bin_ranges.append((p_min + i * k,
                           p_min + i * k + k))
    # print(bins

    # divide the measurements to the bins
    bins = np.zeros(bin_count)
    for value in tweet_data[metric]:
        for i, value_range in enumerate(bin_ranges):
            if value < value_range[1]:
                bins[i] += 1
                break

    # Fit a linear model to the log-transformed data
    x = np.arange(bin_count)
    y = bins
    y_log = np.log(y + 0.000001)
    p = np.polyfit(x, y_log, 1)

    # Convert the linear model back to exponential form
    a = np.exp(p[1])
    b = p[0]
    x_fitted = np.linspace(np.min(x), np.max(x), len(x))
    y_fitted = a * np.exp(b * x_fitted)

    r2 = r2_score(y_log, np.log(y_fitted))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, width=0.8, align='center', alpha=0.6, label='Data')
    plt.plot(x_fitted, y_fitted, 'r-',
             label=f'Exponential fit\ny = {a:.4f} * exp({b:.4f} * x) (R² = {r2:.2f})')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric}')
    plt.xticks(x, [f'{round(j[0])} - {round(j[1] - 1)}' for j in bin_ranges], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison(original_data, resampled_data, metric='impression_count'):
    """
    Plots the original and resampled DataFrames for comparison.

    Parameters:
    - original_df: The original DataFrame with time series data.
    - resampled_df: The resampled DataFrame with time series data.
    - title: Title of the plot.
    """

    # align the first data point
    offset = original_data['timestamp'][0] - resampled_data['timestamp'][0]

    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.plot(original_data['timestamp'], original_data[metric], label=f'Original {metric} Count', marker='o',
             linestyle='-', color='blue')

    # Plot resampled data
    plt.plot((resampled_data['timestamp'] + offset), resampled_data[metric], label=f'Resampled {metric} Count',
             marker='x', linestyle='--', color='red')

    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-d.%-m.\n%H:%M'))
    # plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    # Adding labels and title
    plt.title(f'{metric.replace("_", " ").title()}: Original vs Resampled Data')
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

    # score = mean_squared_error(tweet_data[data_type], infected)
    # score = mean_absolute_error(tweet_data[data_type], infected)
    score = 1 - r2_score(tweet_data[data_type], infected)

    # # Calculate the lower and upper bounds for the 5% range
    # lower_bound = tweet_data[data_type] * 0.95
    # upper_bound = tweet_data[data_type] * 1.05
    # # Count how many predicted values are within the bounds
    # score = ((infected >= lower_bound) & (infected <= upper_bound)).sum()
    # # lower is better
    # score = len(tweet_data) - score

    # mse = abs(sum(trends[0]['trends']['node_count'][1]) - tweet_data['impression_count'].iloc[-1])

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
    file_name = f'barabasi_albert_graph_{name}_{nodes}_{connections}_{seed}.pkl'
    g = load_graph(file_name)
    if not g:
        print('Generating a new Barabási-Albert graph')
        g = nx.barabasi_albert_graph(nodes, connections, seed)
        save_graph(g, file_name)
    return g

def get_erdos_renyi_graph(name, nodes, probability, seed):
    file_name = f'erdos_renyi_graph_{name}_{nodes}_{(str)(probability).replace(".", "")}_{seed}.pkl'
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


def run_simulation(graph, tweet_data, beta, lambda_, gamma, initial_fraction_infected, infection_model, data_type, plot=False, return_MSE=True):
    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)


    config.add_model_parameter('fraction_infected', initial_fraction_infected)  # initial fraction of infected

    if infection_model.upper() == 'SIR':
        config.add_model_parameter('gamma', gamma)
        model = ep.SIRModel(graph)

    elif infection_model.upper() == 'SIS':
        config.add_model_parameter('lambda', lambda_)
        model = ep.SISModel(graph)

    model.set_initial_status(config)
    """trying to set the most central nodes as infected"""
    # # Calculate node centrality
    # centrality = nx.degree_centrality(graph)
    # # centrality = nx.betweenness_centrality(graph)  # takes too long
    # # print(centrality)
    #
    #
    # # Sort nodes by centrality in descending order
    # sorted_nodes = sorted([(node, value) for node, value in centrality.items()], key=lambda x: x[1], reverse=True)
    # # top_node_ids = {node for node, _ in sorted_nodes}
    # # Select the 100 most central nodes
    # # for node in sorted_nodes:
    # #     node_id = node[0]
    # #     config.add_node_configuration("status", node_id, 0)
    #
    # initial_infected = {node: 1 for node, _ in sorted_nodes[:100]}
    # for node in centrality:
    #     if node not in initial_infected:
    #         initial_infected[node] = 0
    #
    # # config.add_node_set_configuration('nodes', initial_infected)
    # # model.set_initial_status(config, infected_nodes=initial_infected)
    # # model.add_model_initial_configuration("Infected", initial_infected)
    # config.add_node_set_configuration("status", initial_infected)
    # config.add_node_set_configuration("nodes", initial_infected)
    # model.set_initial_status(config)
    """"""

    # Run the simulation
    iterations = model.iteration_bunch(len(tweet_data))
    print(5)
    trends = model.build_trends(iterations)
    print(6)

    # Compare with real data and calculate similarity
    similarity = compare_infection_to_data(trends, tweet_data, plot=plot, data_type=data_type, title=f'{infection_model} simulation of {data_type} with parameters b={beta}, l={lambda_}, g={gamma}')

    if return_MSE:
        return similarity

    # return the sum of all nodes infected
    return sum(trends[0]['trends']['node_count'][1])




def model_infection(graph, tweet_data, tweet_id, beta_range, lambda_range, gamma_range, initial_fraction_infected, model, data_type, save_results=True):
    """
    :param graph:
    :param tweet_data:
    :param tweet_id:
    :param beta_range: np.arange(0.1, 0.05, -0.01)  # Infection
    :param lambda_range: np.arange(0.1, 0.05, -0.01)  # Recovery
    :param gamma_range: np.arange(0.1, 0.05, -0.05)  # Removal
    :param initial_fraction_infected:
    """

    ### simulate all parameter combinations ###
    param_combinations = []
    if model.upper() == 'SIR':
        print(f'{datetime.datetime.now().strftime("%H:%M")}:'
              f' Starting simulation with {len(beta_range) * len(gamma_range)} combinations.')
        for i, beta in enumerate(beta_range):
            for gamma in gamma_range:
                # simulate the parameters
                score = run_simulation(graph, tweet_data, beta, 0, gamma, initial_fraction_infected, plot=False, infection_model=model, data_type=data_type)
                param_combinations.append((beta, 0, gamma, (int)(score)))
            print(f'{datetime.datetime.now().strftime("%H:%M")}: Simulation {((i+1)/len(beta_range))*100:.0f} % complete.')
    elif model.upper() == 'SIS':
        print(f'{datetime.datetime.now().strftime("%H:%M")}:'
              f' Starting simulation with {len(beta_range) * len(lambda_range)} combinations.')
        for i, beta in enumerate(beta_range):
            for lambda_ in lambda_range:
                # simulate the parameters
                score = run_simulation(graph, tweet_data, beta, lambda_, 0, initial_fraction_infected, plot=False, infection_model=model, data_type=data_type)
                param_combinations.append((beta, lambda_, 0, (int)(score)))
            print(f"{datetime.datetime.now().strftime('%H:%M')}: Simulation {((i+1)/len(beta_range))*100:.0f} % complete.")

    df = pd.DataFrame(param_combinations, columns=['beta', 'lambda', 'gamma', 'score'])
    if save_results:
        ### save the results to a CSV file ###
        now = datetime.datetime.now().strftime('%d_%H_%M')
        id = tweet_id[-4:]
        file_name = f'simulation_results_{id}_{now}.csv'
        df.to_csv(file_name, index=False)
        print(f'Results saved to {file_name}')


    ### display the result of the best simulation ###
    # Find the row with the minimum score
    best_row = df.loc[df['score'].idxmin()]
    # Extract the parameters
    beta, lambda_, gamma = best_row['beta'], best_row['lambda'], best_row['gamma']
    score = run_simulation(graph, tweet_data, beta, lambda_, gamma, initial_fraction_infected, plot=True, infection_model=model, data_type=data_type)
    print(f'MSE of the best simulation with parameters b,g,l=({beta}, {lambda_}, {gamma}): ', score)







# tweet_id = '1922356232074993957'  # Mari Rantanen: Suunta on oikea 👇🏻 Politiikalla on väliä.
def main():
    # tweet_id = '1922689448447300082'  # P.TOVERI Länsi ei ole sodassa Venäjän kanssa. Unohdettiin ilmeisesti kertoa...
    # tweet_id = '1923052071332217205'  # Purra: Edustaja Bergbom totesi osuvasti, että kyselytunti oli taas...
    # tweet_id = '1923080523703779830'  # Sen, Ted Cruz (R-Tex.) has been a major proponent of the idea
    tweet_id = '1923352006707581422'  # Purra: Monilapsisissa maahanmuuttajaperheissä summa voi nousta lähemmäs viittäkin tonnia
    # tweet_id = '1923342619909701903'  # Chip Roy: Why there’s a problem.
    # tweet_id = '1923477098141802909'  # Mike Collins: So it’s legal for a president to ship millions of illegal aliens into
    # tweet_id = '1923743558680428995'  # Claudia Tenney: The numbers don't lie, President Trump's America First economic plan is working!
    # tweet_id = '1923754485874119007'  # Bernie Sanders: 68,000 Americans already die every year because they don’t have access to the health care...
    # tweet_id = '1923793436068487574'  # Trump: 🇦🇪🇺🇸
    # tweet_id = '1924088518910808080'  # Minja Koskela: Olin tänään monen muun tavoin marssimassa Epäluottamuslause-mielenosoituksessa
    # tweet_id = '1924127364780261454'  # Alice Weidel: Bielefeld: Ein Syrer verletzt 5 Menschen zum Teil schwer.
    # tweet_id = '1924428162257047752'  # Haavisto: Maanantaiaamuna oli mahdollisuus käydä korkealaatuista keskustelua nuorten tilanteesta...
    # tweet_id = '1924437434521010605'  # Orpo: Orpo: Gazan siviilien kärsimyksen on loputtava. Suomi vaatii Israelia...

    # load and resample data
    tweet_data = load_csv(tweet_id)
    interpolated_data = interpolate_data(tweet_data, '1h')

    ### note: this part is dumb
    # ### add a change in shares column, change is between the running average between the two previous columns
    # # Calculate the running average of the two previous values
    # rolling_avg = interpolated_data['impression_count'].rolling(window=2).mean().shift(1)
    # # Compute the change as the difference between current value and the previous average
    # interpolated_data['impression_change'] = interpolated_data['impression_count'] - rolling_avg
    # # # set the first two rows to 1/3 and 2/3 of the third value
    # interpolated_data.loc[0, 'impression_change'] = 0
    # interpolated_data.loc[1, 'impression_change'] = (int)(interpolated_data.loc[2, 'impression_change'] / 2)
    # # interpolated_data.loc[0, 'impression_change'] = (int)(interpolated_data.loc[2, 'impression_change'] / 3)
    # # interpolated_data.loc[1, 'impression_change'] = (int)(interpolated_data.loc[2, 'impression_change'] * 2 / 3)

    ### add an 'impression_change' column
    # Compute the change as the difference between current and previous value
    interpolated_data['impression_change'] = interpolated_data['impression_count'] - interpolated_data['impression_count'].shift(1)
    interpolated_data.loc[0, 'impression_change'] = 0
    interpolated_data['impression_change'] = interpolated_data['impression_change'].astype(int)

    # print(tweet_data['impression_count'])
    # print(interpolated_data['impression_change'])
    # print(interpolated_data['timestamp'])
        # print("impression_change total: ", interpolated_data['impression_change'].sum())

    # shares = retweets + quotes
    shares = interpolated_data['share_count'].max()


    # plot_comparison(tweet_data, interpolated_data, 'impression_count')
    # plot_comparison(tweet_data, interpolated_data, 'like_count')
    # plot_comparison(tweet_data, interpolated_data, 'retweet_count')
    # plot_comparison(tweet_data, interpolated_data, 'reply_count')
    # plot_comparison(tweet_data, interpolated_data, 'quote_count')
    # plot_comparison(tweet_data, interpolated_data, 'bookmark_count')

    ### plotting metrics over time
    # plot_metric_over_time(interpolated_data, 'impression_change')
    # plot_metric_over_time(tweet_data, 'like_count')
    # plot_metric_over_time(tweet_data, 'like_count')
    # plot_metric_over_time(tweet_data, 'retweet_count')
    # plot_metric_over_time(tweet_data, 'reply_count')

    ### plotting the ratio of likes to impressions
    # plot_ratio_over_time(tweet_data)

    ### plotting the polynomial fits
    # fit_polynomial(interpolated_data, 'like_count')
    # fit_polynomial(interpolated_data, 'impression_change', degree=3)
    # fit_polynomial(interpolated_data, 'impression_count')
    # fit_polynomial(interpolated_data, 'reply_count')
    # fit_polynomial(interpolated_data, 'retweet_count')

    ### plotting the exponential fits
    # fit_exponential(interpolated_data, 'impression_count')
    # fit_logarithmic(interpolated_data, 'impression_count')

    ### plotting the power law fit
    # power_law_distribution(interpolated_data, 'impression_count')
    # power_law_distribution(interpolated_data, 'like_count')
    # power_law_distribution(interpolated_data, 'retweet_count')
    # power_law_distribution(interpolated_data, 'reply_count')

    # return

    """4."""
    # We will use the ammount of X/Twitter users as the population
    # https://worldpopulationreview.com/country-rankings/twitter-users-by-country as a source
    # for the population size

    # # X users in 2024 according to World Population Review
    # population_finland = 1600000  # 1.6M
    # population_usa = 111300000  # 111.3M
    # population_germany = 17000000  # 17M

    # Population defined as 100 * the largest number of shares observer in a country
    # Here we will count both retweets and quotes as shares
    population_finland = 324 * 100  # Riikka Purra
    population_usa = 2384 * 100  # Mike Collins, Trumps numbers are too global


    # Network creation, creating a random graph based on population size
    # g_fi = nx.erdos_renyi_graph(32200, 0.005, seed=2025)
    # Barabási–Albert model: connection or "follower" counts follow the power law, just like real Twitter follower counts
    # I'm not sure how to choose initial connections so 10 sounds good enough
    # print('fi:', datetime.datetime.now())


    print('BA:', datetime.datetime.now())
    # Barabási-Albert-graph
    g_fi_ba = get_barabasi_albert_graph('fi', population_finland, 3, seed=2025)
    print('DONE\nER:', datetime.datetime.now())
    # g_fi_er = get_erdos_renyi_graph("fi", population_finland, 0.01, seed=2025)
    # g_fi_er = get_erdos_renyi_graph("fi", population_finland, 0.0000926, seed=2025)  # Average degree 3: 3/32399=0.0000926
    g_fi_er = get_erdos_renyi_graph('fi', population_finland, 0.003, seed=2025)
    print('DONE:', datetime.datetime.now())

    # g_fi_er = g_fi_ba

    print(g_fi_ba)
    print(g_fi_er)


    # print('usa:', datetime.datetime.now())
    # g_usa = generate_graph("usa", population_usa, 10, seed=2025)
    # print('start:', datetime.datetime.now())

    # initial_fraction_infected = interpolated_data['impression_change'][0] / population_finland
    # if initial_fraction_infected < 0.05:
    #     initial_fraction_infected = 0.05
    # initial_fraction_infected = 0.005

    """  parameters for running many simulations at once  """
    # beta_range = np.arange(0.01, 0.15, 0.01)  # Infection
    beta_range = np.arange(0.005, 0.1, 0.005)  # Infection
    lambda_range = np.arange(0.01, 0.2, 0.01)  # Recovery, SIS
    gamma_range = np.arange(0.01, 0.2, 0.01)  # Removal, SIR

    # impressions gained in the first 15 minutes as a fraction of the total population
    initial_fraction_infected = interpolated_data['impression_change'][1] / population_finland / 4
    # or just 0.001
    # initial_fraction_infected = 0.01
    # initial infection rate is the maximum impression gained in an hour
    # initial_fraction_infected = interpolated_data['impression_change'].max() / population_finland

    """ run many simulations to find optimal parameters """
    # SIR
    # model_infection(g_fi_er, interpolated_data, tweet_id, beta_range, lambda_range, gamma_range, initial_fraction_infected,'SIR', 'impression_change')
    # SIS
    # model_infection(g_fi_ba, interpolated_data, tweet_id, beta_range, lambda_range, gamma_range, initial_fraction_infected,'SIS', 'impression_change')

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

    beta = 0.03 # best SIR values PURRA
    lambda_ = 0
    gamma = 0.1

    beta = 0.004 # best SIR values PURRA
    lambda_ = 0
    gamma = 0.13

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

    result = run_simulation(g_fi_er, interpolated_data, beta, lambda_, gamma, initial_fraction_infected,
                            infection_model='SIR', data_type='impression_change', plot=True, return_MSE=False)
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