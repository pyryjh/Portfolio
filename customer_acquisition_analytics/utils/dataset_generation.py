import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import plotly.express as px

from IPython.display import clear_output, display, Markdown, HTML

def hello():
    print('Hello world!')
    


## Production functions


def initial_clusters(cohort_size: int, shares: dict):
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


def installation_timestamps(users: dict, start_date: str, days: int) -> pd.DataFrame:
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
    adjusted_timestamps = [adjust_timestamps(timestamp, start=start_timestamp, end=end_timestamp, initial=True) for timestamp in timestamps]

    # Create a DataFrame
    df = pd.DataFrame({
        'customer_id': list(users.keys()),
        'cluster': list(users.values()),
        'timestamp': adjusted_timestamps
    })
    
    return df


def adjust_timestamps(timestamp: pd.Timestamp, start: pd.Timestamp=None, end: pd.Timestamp=None, initial: bool=False) -> pd.Timestamp:
    '''
    timestamp is the given timestamp to be adjusted
    start is either the beginning of the timeframe (if initial installation generation) or previous bounding timestamp (installation or end of last session)
    end is the end of the timeframe (when generating initial installations), otherwise not applicable
    initial boolean governs if timestamps should be bounded within timeframe, or if they can freely land after end date, and if the previous timestamp
        should be taken into account 
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
    if 0 <= hour < 3 and timestamp.date() == start.date():
        if initial:
            if random.random() < 0.8:  # 80% chance
                new_date = random_date_exclude(start, end, start)
                timestamp = timestamp.replace(year=new_date.year, month=new_date.month, day=new_date.day)
            # 20% chance nothing changes
        else:  # Not initial, i.e. 00-03 and day is the same as previous timestamp
            pass  # Nothing changes in this case
    
    ## Rule 2: Early night (continues)

    # Rule 2: Hour 00-03, and day is not the start day
    if initial and (0 <= hour < 3 and timestamp.date() != start.date()):
        if random.random() < 0.8:  # 80% chance
            timestamp -= pd.Timedelta(hours=3)  # Shift 3 hours earlier
        #    timestamp -= pd.Timedelta(days=1)  # Move to the previous day
        # 20% chance nothing changes
    # Modified rule 2
    elif not initial and (0 <= hour < 3 and timestamp.date() != start.date()):
        if timestamp - start > pd.Timedelta(hours=5) and random.random() < 0.8:  # Over 5 hours since last, 80% chance
            timestamp -= pd.Timedelta(hours=3)  # Shift 3 hours earlier
    # Note: f moving timestamps forward, no need to check the previous timestamp as no risk in conflicting timestamps
    
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
                random_sunday = random.choice(pd.date_range(start=start, end=end, freq='W-SUN'))
                timestamp = timestamp.replace(year=random_sunday.year, month=random_sunday.month, day=random_sunday.day)
            # 90% chance nothing changes

        # Rule 6: Time 09-00, Thursday or Friday
        elif 9 <= hour <= 23 and weekday in [3, 4]:  # Thursday (3) or Friday (4)
            if random.random() < 0.1:  # 10% chance
                random_saturday = random.choice(pd.date_range(start=start, end=end, freq='W-SAT'))
                timestamp = timestamp.replace(year=random_saturday.year, month=random_saturday.month, day=random_saturday.day)
            # 90% chance nothing changes

        # Rule 7: Time 09-00, Wednesday
        elif 9 <= hour <= 23 and weekday == 2:  # Wednesday
            if random.random() < 0.1:  # 10% chance
                random_weekend = random.choice(pd.date_range(start=start, end=end, freq='W-SAT').union(
                    pd.date_range(start=start, end=end, freq='W-SUN')))
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
        if timestamp < start or timestamp > end:
            # Generate a random time within the timeframe
            random_seconds = random.randint(0, int((end - start).total_seconds()))
            timestamp = start + pd.Timedelta(seconds=random_seconds)
    else:  # Not initial
        if timestamp > end:  # Timestamp falls after last accepted time
            return None
        
    return timestamp




## Auxiliary functions


def initial_clusters_deterministic(
        cohort_size: int,
        shares: dict
        ):
    
    shares_calc = {key: (value if value is not None else 0) for key, value in shares.items()}
    total_provided_share = sum(shares_calc.values())

    if total_provided_share > 100:
        raise ValueError('Error: Shares exceed 100%')

    if shares.get('g') is None:
        g_share = {'g': 100 - total_provided_share}
        shares.update(g_share)
    
    
    # Calculate the number of customers per cluster
    cluster_sizes = {key: int(cohort_size * (share / 100)) for key, share in shares.items()}
    
    # Handle rounding errors to ensure the total matches cohort_size
    total_assigned = sum(cluster_sizes.values())
    if total_assigned < cohort_size:
        # Add the remainder to the cluster g ('free to play')
        # largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        cluster_sizes['g'] += cohort_size - total_assigned
    
    # Create the customers dictionary
    users = {}
    start_index = 0
    for cluster, size in cluster_sizes.items():
        for i in range(start_index, start_index + size):
            users[i] = cluster
        start_index += size
    
    return users


def plot_hourly_distribution(df, timestamp_column='timestamp'):
    """
    Plot a line graph showing hourly aggregated counts of timestamps.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing timestamps.
        timestamp_column (str): Column name containing the timestamps.
    
    Returns:
        None: Displays the plot.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Aggregate counts by hour
    hourly_counts = df[timestamp_column].dt.floor('H').value_counts().sort_index()
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-', linewidth=2)
    plt.title('Hourly Distribution of Timestamps', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Count of Timestamps', fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plotly_hourly_distribution(df, timestamp_column='timestamp'):
    """
    Create an interactive Plotly line graph showing hourly aggregated counts of timestamps.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing timestamps.
        timestamp_column (str): Column name containing the timestamps.
    
    Returns:
        None: Displays the interactive plot.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Aggregate counts by hour
    hourly_aggregated = (
        df[timestamp_column]
        .dt.floor('h')  # Use 'h' (lowercase) for rounding to the nearest hour
        .value_counts()
        .sort_index()  # Ensure chronological order
        .reset_index()  # Convert index to a column
        .rename(columns={'timestamp': 'hour', 0: 'count'})  # Rename columns
    )
    hourly_aggregated = hourly_aggregated.reset_index()
#    hourly_aggregated = hourly_aggregated.rename(columns={'index': 'hour'})
    display(hourly_aggregated)
    
    # Add formatted columns for hover information
    hourly_aggregated['Date'] = hourly_aggregated['hour'].dt.date
    hourly_aggregated['Weekday'] = hourly_aggregated['hour'].dt.strftime('%A')
    hourly_aggregated['Time'] = hourly_aggregated['hour'].dt.time
    
    # Create a Plotly line plot
    fig = px.line(
        hourly_aggregated,
        x='hour',
        y='count',
        title='Hourly Distribution of Timestamps',
        labels={'hour': 'Hour', 'count': 'Installations'},
        hover_data={
            'hour': False,  # Exclude raw hour from hover
            'Date': True,
            'Weekday': True,
            'Time': True,
            'count': True,
        },
    )
    
    # Customize layout
    fig.update_traces(marker=dict(size=8))  # Add markers for hover points
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Count of Timestamps',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.show()
