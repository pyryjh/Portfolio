def hello():
    print('Hello world!')
    
def initial_clusters(
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
    customers = {}
    start_index = 0
    for cluster, size in cluster_sizes.items():
        for i in range(start_index, start_index + size):
            customers[i] = cluster
        start_index += size
    
    return customers