import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import pandas as pd
import pytz
import schedule
import tweepy

import keys
import csv

# Initial tweet queue
# [tweet_id, priority], only check tweet if priority=0, if >0, push to the back of the queue
tweet_queue = deque()

# loading private access tokens saved in keys.py (obviously not on public GitHub)
bearer_token = keys.BEARER_TOKEN
consumer_key = keys.CONSUMER_KEY
consumer_secret = keys.CONSUMER_SECRET
access_token = keys.ACCESS_TOKEN
access_token_secret = keys.ACCESS_TOKEN_SECRET

# authenticate as a user
# in theory only bearer_token is needed, but for some reason the other keys are needed, or it throws a bunch of errors
client = tweepy.Client(bearer_token=bearer_token,
                       consumer_key=consumer_key,
                       consumer_secret=consumer_secret,
                       access_token=access_token,
                       access_token_secret=access_token_secret)

def save_tweet_metrics(tweet_id, metrics):
    file_name = f'tweet_data/{tweet_id}.csv'
    file_exists = os.path.isfile(file_name)

    with open(file_name, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    # print(f"Metrics saved to {file_name}")


def check_priority(tweet_id):
    file_name = f'tweet_data/{tweet_id}.csv'
    if not os.path.exists(file_name):
        return 0
    tweet_data = pd.read_csv(file_name)

    if len(tweet_data) < 2:
        return 0

    change = tweet_data['impression_count'].iloc[-1] / tweet_data['impression_count'].iloc[-2]
    # print(change)
    if change > 1.05:  # if change in impressions is bigger than 5 %
        return 0
    elif change > 1.005:  # these numbers are very arbitrary
        return 1
    elif change > 1.0005:
        return 2
    else:
        return 3

# def fetch_tweet(tweet_id):
#     try:
#         tweet = client.get_tweet(tweet_id, tweet_fields=["public_metrics"])
#
#         metrics = tweet.data['public_metrics']
#         # Add current time to the metrics
#         metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
#         save_tweet_metrics(tweet_id, metrics)
#         print(f"{datetime.now().strftime('%H.%M')}: Successfully saved metrics for tweet {tweet_id}.")
#
#     except tweepy.TweepyException as e:
#         print(f"Error fetching tweet: {e}")

def fetch_tweet(tweet_id, max_retries=15):
    retries = 0
    while retries < max_retries:
        try:
            tweet = client.get_tweet(tweet_id, tweet_fields=['public_metrics'])
            metrics = tweet.data['public_metrics']
            metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_tweet_metrics(tweet_id, metrics)
            print(f"{datetime.now().strftime('%H.%M')}: Successfully saved metrics for tweet {tweet_id}")
            return
        except tweepy.TweepyException as e:
            print(f"Error fetching tweet: {e}")
            retries += 1
            print(f"Retrying in 60 seconds... ({retries}/{max_retries})")
            time.sleep(60)
    print(f"Failed to fetch tweet {tweet_id} after {max_retries} attempts.")


def add_tweet_to_queue(tweet_id):
    priority = check_priority(tweet_id)  # lower number is higher priority
    tweet_queue.appendleft([tweet_id, priority])
    print(f"Added tweet {tweet_id} to the front of the queue with priority {priority}.")

def remove_tweet_from_queue(tweet_id):
    for item in tweet_queue:
        if item[0] == tweet_id:
            tweet_queue.remove(item)
            print(f"Removed tweet {tweet_id} from the queue.")
            return
    print(f"Tweet ID {tweet_id} not found in the queue.")

    # try:
    #     tweet_queue.remove(tweet_id)
    #     print(f"Removed tweet {tweet_id} from the queue.")
    # except ValueError:
    #     print(f"Tweet ID {tweet_id} not found in the queue.")


def process_next_tweet():
    if tweet_queue:
        current_id = tweet_queue.popleft()
        tweet_id = current_id[0]
        priority = current_id[1]

        if priority <= 0:
            # only check tweet if priority is 0
            fetch_tweet(tweet_id)
            # decide what the priority is
            priority = check_priority(tweet_id)
            tweet_queue.append([tweet_id, priority])
            # print(tweet_queue)
        else:
            priority -= 1
            tweet_queue.append([tweet_id, priority])
            process_next_tweet()  # process tweets until an API call is made

    else:
        print("Tweet queue is empty. Checking again in 15 minutes.")


def monitor_tweets():
    # Schedule the function to run every 15 minutes
    # Twitter API's free tier is limited to one request every 15 minutes
    print("Starting in 15 seconds.")

    # The user should be able to add or remove tweets from the queue from the terminal
    # Start a separate thread to handle user input
    def input_thread():
        while True:
            user_input = input("Enter +<tweet_id> to add or -<tweet_id> to remove: ").strip()
            if user_input.startswith('+'):
                add_tweet_to_queue(user_input[1:])
            elif user_input.startswith('-'):
                remove_tweet_from_queue(user_input[1:])
            elif user_input:
                print("Invalid input. Use +<id> to add or -<id> to remove.")

    threading.Thread(target=input_thread, daemon=True).start()

    # gives the user 15 minútes to add tweets if non are in queue
    time.sleep(15)

    schedule.every(15).minutes.do(process_next_tweet)
    print("Starting scheduler. Press Ctrl+C to stop.")

    # fetch a tweet immediately after starting the program
    process_next_tweet()

    while True:
        schedule.run_pending()
        time.sleep(5)


def get_original_posting_time(tweet_ids: list, timezone='Europe/Helsinki'):
    """
    Adds an empty row for the original posting time of the tweet.
    Useful for adding original posting times to all the tweets retrospectively
    """

    def fetch_time(tweet_id):
        try:
            tweet = client.get_tweet(tweet_id, tweet_fields=['public_metrics', 'created_at'])
            # Initialize metrics as a dictionary with zeros
            metrics = {key: 0 for key in tweet.data['public_metrics'].keys()}

            # created_at gives the time in UTC
            utc_time = pd.to_datetime(tweet.data['created_at'])
            local_tz = pytz.timezone(timezone)
            local_time = utc_time.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')
            metrics['timestamp'] = local_time

            # save metrics
            save_tweet_metrics(tweet_id, metrics)
            print(f"{datetime.now().strftime('%H.%M')}: Successfully saved metrics for tweet {tweet_id}")
        except Exception as e:
            print(f"Error fetching tweet {tweet_id}: {e}")

    print(f"{datetime.now().strftime('%H.%M')}: Adding the original posting time to {len(tweet_ids)} tweets.\nStarting scheduler. Press Ctrl+C to stop.")

    for tweet_id in tweet_ids:
        fetch_time(tweet_id)
        time.sleep(15*60 + 5)  # sleep 15 minutes



def main():
    # add_tweet_to_queue('1922689448447300082')  # P.TOVERI Länsi ei ole sodassa Venäjän kanssa. Unohdettiin ilmeisesti kertoa...
    # add_tweet_to_queue('1923052071332217205')  # Purra: Edustaja Bergbom totesi osuvasti, että kyselytunti oli taas...
    # add_tweet_to_queue('1923080523703779830')  # Sen, Ted Cruz (R-Tex.) has been a major proponent of the idea
    # add_tweet_to_queue('1923352006707581422')  # Purra: ”Monilapsisissa maahanmuuttajaperheissä summa voi nousta lähemmäs viittäkin tonnia
    # add_tweet_to_queue('1923342619909701903')  # Chip Roy: Why there’s a problem.
    # add_tweet_to_queue('1923477098141802909')  # Mike Collins: So it’s legal for a president to ship millions of illegal aliens into
    # add_tweet_to_queue('1923743558680428995')  # Claudia Tenney: The numbers don't lie, President Trump's America First economic plan is working!
    # add_tweet_to_queue('1923754485874119007')  # Bernie Sanders: 68,000 Americans already die every year because they don’t have access to the health care...
    # add_tweet_to_queue('1923793436068487574')  # Trump: 🇦🇪🇺🇸
    # add_tweet_to_queue('1924088518910808080')  # Minja Koskela: Olin tänään monen muun tavoin marssimassa Epäluottamuslause-mielenosoituksessa
    # add_tweet_to_queue('1924127364780261454')  # Alice Weidel: Bielefeld: Ein Syrer verletzt 5 Menschen zum Teil schwer.
    # add_tweet_to_queue('1924428162257047752')  # Haavisto: Maanantaiaamuna oli mahdollisuus käydä korkealaatuista keskustelua nuorten tilanteesta...
    # add_tweet_to_queue('1924437434521010605')  # Orpo: Gazan siviilien kärsimyksen on loputtava. Suomi vaatii Israelia...
    # add_tweet_to_queue('1926997808369840418')
    # add_tweet_to_queue('1927003736154554560')
    # add_tweet_to_queue('1927013798415585709')

    monitor_tweets()

    # # adding original posting times to all the tweets retrospectively
    # tweets=[1923080523703779830, 1923352006707581422, 1924088518910808080,
    #         1924127364780261454, 1924428162257047752, 1924437434521010605, 1923342619909701903,
    #         1923477098141802909, 1923743558680428995, 1923754485874119007, 1923793436068487574]
    # get_original_posting_time(tweets)


if __name__ == "__main__":
    main()
