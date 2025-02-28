import pandas as pd
import numpy as np
import random

#import matplotlib.pyplot as plt
#import plotly.express as px

#from IPython.display import clear_output, display, Markdown, HTML

def cluster_attributes() -> dict:
    """
    Create dictionary that contains attributes for each hidden user cluster.
    Used for generating user actions for dataset.
    For each cluster contains session timing and lenght parameters, chance of
    converting to another hidden cluster (incl. churned), and chance for 
    making a transaction with distribution on transaction size.

    Churned cluster (f) contains chance of reactivation + reactivation target
    hidden cluster, based on the hidden cluster at the time of churn.
    """
    transactions = [1, 2, 4, 5, 8, 12, 20]

    a_stats = {
        'description': 'Not implemented'
    }

    b_stats = {
        'description': 'Regular buyer',
        'session_start_mu': 12.98,
        'session_start_sigma': 0.3,
        'session_length_mu': 3600,  # 1 hour in seconds
        'session_length_sigma': 1200,  # 20 minutes in seconds
        'conversion_chance_target': {
            'a': 0,
            'b': 0,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 6,
            'g': 0
        },
        'transaction_probability': 70,
        'transaction_size': transactions,
        'transaction_size_weight': [5, 15, 15, 40, 10, 10, 5]
    }

    c_stats = {
        'description': 'Starts similar to d, turns quickly into b',
        'session_start_mu':11.2,
        'session_start_sigma': 0.5,
        'session_length_mu': 3600,  # 1 hour in seconds
        'session_length_sigma': 1200,  # 20 minutes in seconds
        'conversion_chance_target': {
            'a': 0,
            'b': 0,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 25,
            'g': 0
        },
        'transaction_probability': 90,
        'transaction_size': transactions,
        'transaction_size_weight': [1, 1, 5, 5, 6, 45, 10]
    }

    d_stats = {
        'description': 'Spends initially, churns fast',
        'session_start_mu':11.2,
        'session_start_sigma': 0.5,
        'session_length_mu': 1200,  # 20 minutes in seconds
        'session_length_sigma': 500,
        'conversion_chance_target': {
            'a': 0,
            'b': 1,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 25,
            'g': 0
        },
        'transaction_probability': 90,
        'transaction_size': transactions,
        'transaction_size_weight': [1, 1, 5, 5, 6, 45, 10]
    }

    e_stats = {
        'description': 'Not implemented'
    }

    f_stats = {
        'description': 'Churned user',
        # Previous cluster, reactivation chance
        'a': {},
        'b': {
            'a': 0,
            'b': 0.5,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 0,
            'g': 0
            },
        'c': {
            'a': 0,
            'b': 4,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 0,
            'g': 0
            },
        'd': {
            'a': 0,
            'b': 0,
            'c': 0,
            'd': 0.1,
            'e': 0,
            'f': 0,
            'g': 0
            },
        'e': {},
        'f': {},
        'g': {
            'a': 0,
            'b': 0,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 0,
            'g': 2
        }
    }

    g_stats = {  # Free to play
        'description': 'Free to play, never buys, sometimes converts',
        'session_start_mu': 11.3,
        'session_start_sigma': 0.2,
        'session_length_mu': 1200,  # 20 minutes in seconds
        'session_length_sigma': 500,
        'conversion_chance_target': {
            'a': 0,
            'b': 0.1,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 5,
            'g': 0
        }
    }


    cluster_stats = {
        'a': a_stats,
        'b': b_stats,
        'c': c_stats,
        'd': d_stats,
        'e': e_stats,
        'f': f_stats,
        'g': g_stats,
    }

    return cluster_stats


def session_start(
        initial_timestamp: pd.Timestamp,
        mu: float,
        sigma: float,
        max_days=180
        ) -> pd.Timestamp:
    """
    Takes a timestamp of previous known activity and generates the starting
    time of next session, based on user hidden cluster session timing
    distribution parameters.

    Returns none if the next session start would fall outside dataset
    generation limits (i.e. no new session occured which would impact
    the analysis demonstration)
    """
    # Convert initial timestamp to pandas Timestamp (if not already)
    initial_time = pd.Timestamp(initial_timestamp)

    # Generate lognormal distribution
    seconds_after = np.random.lognormal(mean=mu, sigma=sigma)

    # Limit timescale to max_days
    max_seconds = max_days * 24 * 60 * 60  # Convert days to seconds
    if seconds_after > max_seconds:
        return None

    session_start_time = initial_time + pd.Timedelta(seconds=seconds_after)
    return session_start_time


def session_end(
        mean: float,
        std_dev: float,
        start_timestamp: pd.Timestamp=None
        ):
    """
    Generate a timestamp offset from the start using a bounded normal distribution.

    Parameters:
        start_timestamp (str or pd.Timestamp): Starting timestamp in 'YYYY-MM-DD HH:MM:SS' format.
        mean (float): Mean of the normal distribution (in seconds).
        std_dev (float): Standard deviation of the normal distribution (in seconds).

    Returns:
        float: Session length in seconds
        pd.Timestamp: Generated timestamp.
    """

    # Define bounds (2 standard deviations)
    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 5 * std_dev

    # Generate a valid value within bounds
    while True:
        value = np.random.normal(loc=mean, scale=std_dev)
        if lower_bound <= value <= upper_bound:
            break

    # Rescale the distribution to fit within the bounds
    scaled_value = value

    # Add the offset to the starting timestamp (if given)
    if start_timestamp is not None:
        # Convert start_timestamp to pandas Timestamp
        start_time = pd.Timestamp(start_timestamp)
        generated_timestamp = start_time + pd.Timedelta(seconds=scaled_value)
    else:
        generated_timestamp = None

    return scaled_value, generated_timestamp


def determine_session_end_cluster(
        session_start_cluster: str,
        cluster_stats: dict
        ) -> str:
    """
    Determine the session_end_cluster based on the conversion chances in cluster_stats.

    Parameters:
        session_start_cluster (str): The starting cluster for the session.
        cluster_stats (dict): Dictionary containing conversion_chance_target for clusters.

    Returns:
        str: The session_end_cluster
    """
    # Get the conversion chance target for the session_start_cluster
    conversion_chances = cluster_stats.get(session_start_cluster, {}).get('conversion_chance_target', {})

    # Prepare the options and their probabilities
    options = list(conversion_chances.keys())
    probabilities = [conversion_chances[option] for option in options]

    # Add the default case: staying in the same cluster
    options.append(session_start_cluster)
    probabilities.append(100 - sum(probabilities))  # Remaining chance to stay

    # Normalize probabilities (in case rounding issues arise)
    probabilities = [max(0, p) for p in probabilities]  # Ensure no negative values
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    # Roll for the session_end_cluster
    session_end_cluster = random.choices(options, weights=probabilities, k=1)[0]

    return session_end_cluster


def calculate_reactivation(
        starting_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        reactivation_stats: dict,
        churn_cluster: str):
    """
    Determine if reactivation occurs between two timestamps,
    and if so, generate a random timestamp for the reactivation.

    Parameters:
        starting_timestamp (pd.Timestamp): The starting timestamp.
        end_timestamp (pd.Timestamp): The ending timestamp.
        reactivation_chance (float): Daily reactivation chance as a percentage (e.g., 3 for 3%).

    Returns:
        pd.Timestamp or None: Reactivation timestamp if reactivation occurs, otherwise None.
        str or None: Target cluster (hidden user cluster after reactivation)
    """
    # Failsafe in case no time of churning is given (may occur if outside
    # acceptable boudaries of this analysis)
    if starting_timestamp is None:
        return None, None
    # Ensure timestamps are in pandas format
    starting_timestamp = pd.Timestamp(starting_timestamp)
    end_timestamp = pd.Timestamp(end_timestamp)

    reactivation_chances = reactivation_stats.get(churn_cluster, {})

    # Calculate the total reactivation probability
    total_reactivation_chance = sum(reactivation_chances.values())
    days_between = (end_timestamp - starting_timestamp).days
    total_chance_over_days = 1 - (1 - total_reactivation_chance / 100) ** days_between

    # Roll to check if reactivation occurs
    if random.random() > total_chance_over_days:
        return None, None  # No reactivation

    # If reactivation occurs, pick a target cluster
    target_clusters = list(reactivation_chances.keys())
    probabilities = list(reactivation_chances.values())
    target_cluster = random.choices(target_clusters, weights=probabilities, k=1)[0]

    # Generate a random date within the range (excluding starting date)
    possible_dates = pd.date_range(start=starting_timestamp + pd.Timedelta(days=1), end=end_timestamp)
    reactivation_date = random.choice(possible_dates)

    # Generate a random time within the chosen date
    random_seconds = random.randint(0, 86400 - 1)  # Total seconds in a day
    reactivation_time = pd.Timestamp(reactivation_date) + pd.Timedelta(seconds=random_seconds)

    return reactivation_time, target_cluster


def session_length_and_actions(
        session_start_timestamp: pd.Timestamp,
        user_id: str,
        session_start_cluster: str,
        running_session_number: int,
        original_cluster: str,
        session_start_action_id: str='session_start'
        ):
    """
    Generator function which yields user actions as rows to be appended to
    fct_user_action (a table which logs user actions)

    This function generates basic actions like session start and session end.
    Inside this function there is another generator, which yields other
    actions that occure within the session.
    """
    cluster_stats = cluster_attributes()

    # Generate session_end
    mu = cluster_stats[session_start_cluster]['session_length_mu']
    sigma = cluster_stats[session_start_cluster]['session_length_sigma']
    session_length, session_end_timestamp = session_end(mu, sigma, session_start_timestamp)

    session_end_cluster = determine_session_end_cluster(session_start_cluster, cluster_stats)

    # Add session row to fct_user_action
    session_id = f"{user_id}-{running_session_number}"
    row = {
        'user_id': user_id,
        'session_id': session_id,
        'running_session_number': int(running_session_number),
        'session_start': session_start_timestamp,
        'session_length': session_length,
        'session_end': session_end_timestamp,
        'original_cluster': original_cluster,
        'session_start_cluster': session_start_cluster,
        'session_end_cluster': session_end_cluster,
        'action_id': session_start_action_id,
        'action_timestamp': session_start_timestamp,
        'transaction_size': None
    }

    yield row

    # This is now basic implementation for basic functionality
    # TODO:
    # [] Generate time distribution for transaction to occur, check against end of session time,
    # if occurs then yield row and rerun with the probability now starting at the previous transaction

    # Transaction:
    activity = 'transaction'
    for activity_row in activity_row_generator(session_start_cluster, activity, session_start_timestamp, session_length):
        row = {
            'user_id': user_id,
            'session_id': session_id,
            'running_session_number': int(running_session_number),
            'session_start': session_start_timestamp,
            'session_length': session_length,
            'session_end': session_end_timestamp,
            'original_cluster': original_cluster,
            'session_start_cluster': session_start_cluster,
            'session_end_cluster': session_end_cluster,
            'action_id': activity_row.get('action_id'),
            'action_timestamp': activity_row.get('action_timestamp'),
            'transaction_size': activity_row.get('transaction_size')
        }
        yield row
        

def activity_row_generator(
        session_cluster: str,
        activity_type: str,
        session_start_timestamp: pd.Timestamp,
        session_length: int):
    """
    This generator generates activity rows for fct_user_action.
    For now it only generates transactions. In future this might generate any
    type of action.

    Transaction is generated as occuring based on hidden user cluster probability
    and distribution, timing will be at the halfway of user session (for now)
    """
    cluster_stats = cluster_attributes()
    probability = cluster_stats.get(session_cluster).get(f'{activity_type}_probability')
    if not probability:  # Check if activity probability found in cluster attributes, otherwise break
        return

    # Transaction basic, in future do this with time distribution probability against session end in while loop 
    if activity_type == 'transaction':
        sizes = cluster_stats.get(session_cluster).get('transaction_size')
        weights = cluster_stats.get(session_cluster).get('transaction_size_weights')

        if random.random() < probability / 100:
            transaction = random.choices(sizes, weights=weights, k=1)[0]
            transaction_timestamp = session_start_timestamp + pd.Timedelta(seconds= session_length / 2)
            output_row = {
                'session_start': session_start_timestamp,
                'action_id': activity_type,
                'action_timestamp': transaction_timestamp,
                'transaction_size': transaction
            }
            yield output_row


def recursive_reactivation_function(churned: list):
    """
    This recursive generator function takes list of churned users, containing
    user_id, time of churn, user hidden cluster at the time of churn, and time
    left to reactivate before the end of this simulation.

    It will calculate whether a reactivation will happen before the end of
    simulation based on user hidden cluster at the time of churn.

    If reactivation occurs, user actions are generated based on reactivation 
    target hidden cluster until either end of simulation timeframe is reached
    or the user churns again. All the users who reactivate and churn are then
    appended to new list, which if not empty will pass recursivevely to this
    function, causing this function to yield activity rows of reactivating
    users and calculate churn and reactivation until no reactivation occurs
    to any churned user within accepted timeframe, or no reactivated user
    churns again within accepted timeframe.
    """
    churned_new = []

    for row in churned:
        reactivation_time, target_cluster = calculate_reactivation(
            starting_timestamp=row['session_end'],
            end_timestamp=row['last_accepted_timestamp'],
            reactivation_stats=cluster_attributes().get('f'),
            churn_cluster=row['session_start_cluster']
            )
        
        if reactivation_time:
            session_end_timestamp = None
            last_accepted_timestamp = row['last_accepted_timestamp']
            cluster_stats = cluster_attributes()
            session_start_cluster = target_cluster
            user_id = row['user_id']
            running_session_number = row['running_session_number']
            original_cluster = row['original_cluster']

            while session_end_timestamp is None or (session_end_timestamp < last_accepted_timestamp):
                running_session_number += 1  # Running session number taken from the last known running session number and added one every time a new session is generated
                reactivation_action_id = None
                
                if session_end_timestamp is None:  # First time after reactivation
                    first_accepted_timestamp = row['session_end']
                    session_start_timestamp = reactivation_time
                    reactivation_action_id = 'reactivation'
                else:  # Subsequent sessions after initial reactivation
                    mu = cluster_stats[session_start_cluster]['session_start_mu']
                    sigma = cluster_stats[session_start_cluster]['session_start_sigma']
                    first_accepted_timestamp = session_end_timestamp  # Previous session end timestamp
                    session_start_timestamp = session_start(session_end_timestamp, mu, sigma)

                # Adjust session_start_timestamp
                session_start_timestamp = adjust_timestamps(session_start_timestamp, start_bound=first_accepted_timestamp, end_bound=last_accepted_timestamp)

                # Break if session_start_timestamp is None
                if session_start_timestamp is None:
                    break
                
                ## Get user action rows
                for row in session_length_and_actions(
                            session_start_timestamp,
                            user_id, session_start_cluster,
                            running_session_number,
                            original_cluster,
                            session_start_action_id=reactivation_action_id if reactivation_action_id else 'session_start'
                        ):
                    yield row
                    last_row = row
                
                # Check if session ended in churn, if yes then pass churn parameters and break
                previous_start_cluster = last_row.get('session_start_cluster')
                previous_session_number = last_row.get('running_session_number')
                session_end_cluster = last_row.get('session_end_cluster')
                session_id = last_row.get('session_id')
                session_end_timestamp = last_row.get('session_end')

                # Debug test:
                if not session_end_cluster:
                    raise ValueError(f'''session_end_cluster missing, here's last row: {last_row}''')
                
                if session_end_cluster == 'f':  # Churn
                    break  # This user will be handled separately as churned from now on

                # Update session_start_cluster for next session
                session_start_cluster = session_end_cluster  # If user cluster changed, next session will follow new cluster rules
            
            if session_end_cluster == 'f':  # Churn
                churned_new.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'running_session_number': previous_session_number,
                    'session_end': session_end_timestamp,
                    'original_cluster': original_cluster,
                    'session_start_cluster': previous_start_cluster,
                    'session_end_cluster': session_end_cluster,
                    'last_accepted_timestamp': last_accepted_timestamp
                })
        
    if churned_new:
        yield from recursive_reactivation_function(churned_new)


def initial_clusters(cohort_size: int, shares: dict) -> dict:
    """
    Allocate cohort users to clusters randomly based on given shares.
    
    Parameters:
        cohort_size (int): Total number of users in the cohort.
        shares (dict): Dictionary of cluster shares as percentages.
        
    Returns:
        dict: Dictionary of user IDs mapped to their clusters.
    """
    # Normalize the shares: Ensure missing 'g' is added and shares sum to 100
    shares_calc = {key: (value if value is not None else 0) for key, value in shares.items()}
    total_provided_share = sum(shares_calc.values())
    
    if total_provided_share > 100:
        raise ValueError("Error: Shares exceed 100%")
    
    if shares.get('g') is None:
        g_share = {'g': 100 - total_provided_share}
        shares.update(g_share)

    # Prepare clusters and their probabilities
    clusters = list(shares.keys())
    probabilities = [shares[cluster] / 100 for cluster in clusters]
    
    # Randomly assign users to clusters based on probabilities
    users = {}
    for i in range(cohort_size):
        users[i] = random.choices(clusters, weights=probabilities, k=1)[0]
    
    return users


def installation_timestamps(
        users: dict,
        start_date: str,
        days: int
    ) -> pd.DataFrame:
    """
    Generate a pandas DataFrame with timestamps for each customer.

    Parameters:
        customer_dict (dict): Dictionary with customer IDs as keys.
        start_date (str): The starting date (e.g., '2025-01-01') as a string in 'YYYY-MM-DD' format.
        days (int): The length of the timeframe in days.

    Returns:
        pd.DataFrame: DataFrame with customer IDs and their generated timestamps.
    """
    # Convert start_date to pandas Timestamp and calculate the end date
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = start_timestamp + pd.Timedelta(days=days)
    
    # Number of customers
    num_customers = len(users)
    
    # Generate random timestamps within the timeframe
    random_seconds = np.random.randint(0, days * 24 * 60 * 60, size=num_customers)
    timestamps = start_timestamp + pd.to_timedelta(random_seconds, unit='s')

    # Adjust timestamps
    adjusted_timestamps = [adjust_timestamps(timestamp, start_bound=start_timestamp, end_bound=end_timestamp, initial=True) for timestamp in timestamps]

    # Create a DataFrame
    df = pd.DataFrame({
        'customer_id': list(users.keys()),
        'cluster': list(users.values()),
        'timestamp': adjusted_timestamps
    })
    
    return df


def adjust_timestamps(
        timestamp: pd.Timestamp,
        start_bound: pd.Timestamp=None,
        end_bound: pd.Timestamp=None,
        initial: bool=False
    ) -> pd.Timestamp:
    '''
    Takes any randomly generated timestamp (usually session start) and adjusts
    it to somewhat reflect real world realities, including:
        - Less activity during nighttime
        - Less activity during workday/schoolday, excl. lunch hours
        - Increased activity during weekends
    
    Parameters:
        timestamp: pd.Timestamp is the given timestamp to be adjusted
        start_bound: pd.Timestamp is either the beginning of the timeframe
            (if initial installation generation) or previous bounding timestamp
            (installation or end of last session)
        end_bound: pd.Timestamp is the end of the timeframe (when generating initial
            installations), otherwise not applicable
        initial: bool governs if timestamps should be bounded within timeframe,
            or if they can freely land after end date,
            and whether the previous timestamp should be taken into account 
    '''
    if timestamp is None:
        return None
    
    # Helper function to generate a random date within a timeframe
    def random_date_exclude(start, end, exclude):
        dates = pd.date_range(start=start, end=end).difference([exclude])
        return random.choice(dates)

    # Extract hour, minute, and weekday
    hour = timestamp.hour
    weekday = timestamp.weekday()  # 0=Monday, 6=Sunday

    ## Rule 1: Early night
    
    # Rule 1: Hour 00-03, and day is the start day
    if 0 <= hour < 3 and timestamp.date() == start_bound.date():
        if initial:
            if random.random() < 0.8:  # 80% chance
                new_date = random_date_exclude(start_bound, end_bound, start_bound)
                timestamp = timestamp.replace(year=new_date.year, month=new_date.month, day=new_date.day)
            # 20% chance nothing changes
        else:  # Not initial, i.e. 00-03 and day is the same as previous timestamp
            pass  # Nothing changes in this case
    
    ## Rule 2: Early night (continues)

    # Rule 2: Hour 00-03, and day is not the start day
    if initial and (0 <= hour < 3 and timestamp.date() != start_bound.date()):
        if random.random() < 0.8:  # 80% chance
            timestamp -= pd.Timedelta(hours=3)  # Shift 3 hours earlier
        #    timestamp -= pd.Timedelta(days=1)  # Move to the previous day
        # 20% chance nothing changes
    # Modified rule 2
    elif not initial and (0 <= hour < 3 and timestamp.date() != start_bound.date()):
        if timestamp - start_bound > pd.Timedelta(hours=5) and random.random() < 0.8:  # Over 5 hours since last, 80% chance
            timestamp -= pd.Timedelta(hours=3)  # Shift 3 hours earlier
    # Note: if moving timestamps forward, no need to check the previous timestamp as no risk in conflicting timestamps
    
    # Rule 3: Hour 03-05
    elif 3 <= hour < 5:
        if random.random() < 0.8:  # 80% chance
            timestamp += pd.Timedelta(hours=4)  # Move 4 hours forward
        # 20% chance nothing changes

    # Rule 4: Hour 05-07
    elif 5 <= hour < 7:
        if random.random() < 0.8:  # 80% chance
            timestamp += pd.Timedelta(hours=2)  # Move 2 hours forward
        # 20% chance nothing changes


    # This block handled way different in adjusting timestamps after initial install

    if initial:
        # Rule 5: Time 09-00 (9 AM to midnight), Monday or Tuesday
        if 9 <= hour <= 23 and weekday in [0, 1]:  # Monday (0) or Tuesday (1)
            if random.random() < 0.1:  # 10% chance
                random_sunday = random.choice(pd.date_range(start=start_bound, end=end_bound, freq='W-SUN'))
                timestamp = timestamp.replace(year=random_sunday.year, month=random_sunday.month, day=random_sunday.day)
            # 90% chance nothing changes

        # Rule 6: Time 09-00, Thursday or Friday
        elif 9 <= hour <= 23 and weekday in [3, 4]:  # Thursday (3) or Friday (4)
            if random.random() < 0.1:  # 10% chance
                random_saturday = random.choice(pd.date_range(start=start_bound, end=end_bound, freq='W-SAT'))
                timestamp = timestamp.replace(year=random_saturday.year, month=random_saturday.month, day=random_saturday.day)
            # 90% chance nothing changes

        # Rule 7: Time 09-00, Wednesday
        elif 9 <= hour <= 23 and weekday == 2:  # Wednesday
            if random.random() < 0.1:  # 10% chance
                random_weekend = random.choice(pd.date_range(start=start_bound, end=end_bound, freq='W-SAT').union(
                    pd.date_range(start=start_bound, end=end_bound, freq='W-SUN')))
                timestamp = timestamp.replace(year=random_weekend.year, month=random_weekend.month, day=random_weekend.day)
            # 90% chance nothing changes
    else:  #not initial
        # Check if the current day is a weekday (Monday to Friday)
        if weekday in [0, 1, 2, 3, 4]:  # 0=Monday, ..., 4=Friday
            if random.random() < 0.04:  # 4% chance
                # Generate the next Saturday or Sunday within the range
                next_weekend = random.choice(
                    pd.date_range(start=timestamp + pd.Timedelta(days=1), end=timestamp + pd.Timedelta(days=7), freq='W-SAT')
                    .union(pd.date_range(start=timestamp + pd.Timedelta(days=1), end=timestamp + pd.Timedelta(days=7), freq='W-SUN'))
                )
                # Move timestamp to the selected weekend day, keeping time unchanged
                timestamp = timestamp.replace(year=next_weekend.year, month=next_weekend.month, day=next_weekend.day)

    # Note: Moving timestamps forward, no need to check previous timestamps

    # Rule 8: Weekday (Mon-Fri), Hour 09-11
    if weekday < 5 and 9 <= hour < 11:  # Weekdays (Mon-Fri)
        if random.random() < 0.5:  # 50% chance
            timestamp += pd.Timedelta(hours=2)  # Add 2 hours
        # 50% chance nothing changes

    # Rule 9: Weekday (Mon-Fri), Hour 13-17
    if weekday < 5 and 13 <= hour < 17:  # Weekdays (Mon-Fri)
        if random.random() < 0.5:  # 50% chance
            new_hour = random.randint(17, 23)  # Randomize hour between 17 and 23
            timestamp = timestamp.replace(hour=new_hour)
        # 50% chance nothing changes

    if initial:
        # Rule 10: Catch all that still fell outside bounds
        if timestamp < start_bound or timestamp > end_bound:
            # Generate a random time within the timeframe
            random_seconds = random.randint(0, int((end_bound - start_bound).total_seconds()))
            timestamp = start_bound + pd.Timedelta(seconds=random_seconds)
    else:  # Not initial
        if timestamp > end_bound:  # Timestamp falls after last accepted time
            return None
        
    return timestamp





# Define generate_dataset

def generate_dataset(cluster_shares: dict, size: int, start_date: str):
    users_dict = initial_clusters(
        cohort_size=size,
        shares=cluster_shares
        )
    

    # Generate installation timestamps for all users
    df = installation_timestamps(users_dict, start_date, 31)

    columns = [
        'user_id', 'session_id', 'session_start', 'session_length', 'session_end',
        'original_cluster', 'session_start_cluster', 'session_end_cluster',
        'action_id', 'action_timestamp', 'transaction_size'
    ]
    #fct_user_action = pd.DataFrame(columns=columns)

    rows = []
    churned = []
    
    cluster_stats = cluster_attributes()

    for _, user_row in df.iterrows():
        user_id = user_row['customer_id']
        original_cluster = user_row['cluster']
        install_timestamp = user_row['timestamp']

        # Add initial "install" row to fct_user_action
        rows.append({
            'user_id': user_id,
            'session_id': None,
            'session_start': None,
            'session_length': None,
            'session_end': None,
            'original_cluster': original_cluster,
            'session_start_cluster': None,
            'session_end_cluster': None,
            'action_id': 'install',
            'action_timestamp': install_timestamp,
            'transaction_size': None
        })

        # Prepare for session generation
        session_start_cluster = original_cluster
        session_end_timestamp = None
        running_session_number = 0

        last_accepted_timestamp = install_timestamp + pd.Timedelta(days=120)

        while session_end_timestamp is None or (session_end_timestamp < last_accepted_timestamp):
            # Determine session_start_timestamp
            if session_end_timestamp is None:  # First session
                if random.random() < 0.8:  # 80% chance
                    session_start_timestamp = install_timestamp + pd.Timedelta(seconds=30)
                else:  # 20% chance
                    mu = cluster_stats[session_start_cluster]['session_start_mu']
                    sigma = cluster_stats[session_start_cluster]['session_start_sigma']
                    session_start_timestamp = session_start(install_timestamp, mu, sigma)
                start_boundary = install_timestamp
            else:  # Subsequent sessions
                mu = cluster_stats[session_start_cluster]['session_start_mu']
                sigma = cluster_stats[session_start_cluster]['session_start_sigma']
                session_start_timestamp = session_start(session_end_timestamp, mu, sigma)
                start_boundary = session_end_timestamp  # Previous session end timestamp

            # Adjust session_start_timestamp
            session_start_timestamp = adjust_timestamps(session_start_timestamp, start_bound=start_boundary, end_bound=last_accepted_timestamp)

            # Break if session_start_timestamp is None
            #print(f'DEBUG: {session_start_timestamp}')
            if session_start_timestamp is None:
                break
            
            ## Get user action rows
            for row in session_length_and_actions(session_start_timestamp, user_id, session_start_cluster, running_session_number, original_cluster):
                rows.append(row)
            
            # Check if session ended in churn, if yes then pass churn parameters and break
            last_row = rows[-1]
            previous_start_cluster = last_row.get('session_start_cluster')
            previous_session_number = last_row.get('running_session_number')
            session_end_cluster = last_row.get('session_end_cluster')
            session_id = last_row.get('session_id')
            session_end_timestamp = last_row.get('session_end')
            if session_end_cluster == 'f':  # Churn
                break  # This user will be handled separately as churned from now on

            # Update session_start_cluster for next session
            session_start_cluster = session_end_cluster  # If user cluster changed, next session will follow new cluster rules
            running_session_number += 1
        
        if session_end_cluster == 'f':  # Churn
            churned.append({
                'user_id': user_id,
                'session_id': session_id,
                'running_session_number': previous_session_number,
                'session_end': session_end_timestamp,
                'original_cluster': original_cluster,
                'session_start_cluster': previous_start_cluster,
                'session_end_cluster': session_end_cluster,
                'last_accepted_timestamp': last_accepted_timestamp
            })
            
    if churned:
        for row in recursive_reactivation_function(churned):
            rows.append(row)
        
    fct_user_action = pd.DataFrame(rows)
    fct_user_action['running_session_number'] = pd.to_numeric(fct_user_action['running_session_number'], errors='coerce').astype('Int64')
#    df_churned = pd.DataFrame(churned)
#    display(fct_user_action)
#    display(df_churned)
    return fct_user_action


def analysis_query(name):
    query = f'''
        with

        base as (
            SELECT * 
            FROM {name}
        ),

        user_install as (
        select
            user_id,
            min(action_timestamp) as install_timestamp
        from base
        where action_id = 'install'
        group by 1
    )
    --select * from user_install
    ,

    transactions as (
        SELECT
            base.*,
            user_install.install_timestamp,
            strftime('%s', base.action_timestamp) - strftime('%s', user_install.install_timestamp) AS time_since_install_seconds,
            (julianday(base.action_timestamp) - julianday(user_install.install_timestamp)) AS time_since_install_days
        from base
        join user_install
            on base.user_id = user_install.user_id
        where action_id = 'transaction'
    )
    --select * from transactions;
    ,

    users as (
        select
            count(distinct user_id) as users_total
        from base
    ),

    transactions_cum as (
        select
            time_since_install_seconds,
            sum(transaction_size) over (
            order by time_since_install_seconds
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as cum_transactions
        from transactions
    )
    select
        transactions_cum.*,
        cum_transactions / users_total as total_ltv_per_user
    from transactions_cum
    join users on 1=1
    order by time_since_install_seconds
    '''

    return query
