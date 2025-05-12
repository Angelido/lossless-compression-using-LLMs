import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
from collections import Counter


# ====== read_rank_list_to_dataframe ====== #
def read_rank_list_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads a rank list from a file and returns it as a DataFrame with a single column 'Rank List'.
    Ignores the first line (header) and assumes each following line is in the format: [1 2 3 4].
    
    Input:
        - file_path: Path to the file containing the rank list.
    
    Return:
        - df: A Pandas DataFrame containing the rank list.
    """
    rank_list: List[List[int]] = []
    with open(file_path, 'r') as f:
        next(f)  # salto l’header
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = list(map(int, line.strip('[]').split()))
            rank_list.append(nums)
    
    # Creo prima una Series, poi la trasformo in DataFrame
    series = pd.Series(rank_list, name='Rank List')
    df = series.to_frame()
    return df


# ====== get_file_size_mb ====== #
def get_file_size_mb(filepath: str) -> float:
    """Returns the file size in megabytes (MB).
    
    Input:
    - filepath: Path to the file.
    
    Return:
    - size_bytes: Size of the file in MB.
    """
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)  # convert to MB



# ====== analyze_rank_list_df ====== #
def analyze_rank_list_df(
    df: pd.DataFrame,
    column: str = "Rank List"
) -> Dict[str, Any]:
    """
    Analyze a DataFrame containing a column of integer lists.
    Computes frequency, max, min, total count, sum, and number of lists,
    as well as central tendency and dispersion metrics and selected percentiles.

    Input:
    - df: pandas DataFrame with a column of integer lists
    - column: name of the target column (default 'Rank List')

    Return:
    - dict with keys:
        - frequency: dict mapping number → count
        - max: maximum value in all lists
        - min: minimum value in all lists
        - total_numbers: total number of integers
        - total_sum: sum of all integers
        - num_lists: number of lists processed
        - mean: average of all values
        - median: median value
        - mode: most frequent value
        - variance: population variance
        - std: population standard deviation
        - percentiles: dict of selected percentiles (10, 25, 50, 75, 90)
    """
    # Extract and flatten the list of lists
    rank_list = df[column].dropna().tolist()
    flat_list = [num for sublist in rank_list for num in sublist]

    # Compute frequency counts
    freq = dict(Counter(flat_list))
    total_numbers = len(flat_list)

    # Basic statistics: max, min, sum, and number of lists
    max_value = max(flat_list) if flat_list else None
    min_value = min(flat_list) if flat_list else None
    total_sum = sum(flat_list)
    num_lists = len(rank_list)

    # Central tendency and dispersion
    mean_val = np.mean(flat_list) if flat_list else None
    median_val = np.median(flat_list) if flat_list else None
    # Mode: pick smallest value in case of multiple modes
    modes = [k for k, v in freq.items() if v == max(freq.values())] if flat_list else []
    mode_val = min(modes) if modes else None
    variance_val = np.var(flat_list, ddof=0) if flat_list else None
    std_val = np.std(flat_list, ddof=0) if flat_list else None

    # Selected percentiles
    percentile_points = [10, 25, 50, 75, 90]
    percentiles = {}
    if flat_list:
        arr = np.array(flat_list)
        for p in percentile_points:
            percentiles[p] = np.percentile(arr, p)

    return {
        "frequency": freq,
        "max": max_value,
        "min": min_value,
        "total_numbers": total_numbers,
        "total_sum": total_sum,
        "num_lists": num_lists,
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "variance": variance_val,
        "std": std_val,
        "percentiles": percentiles
    }
    


# ====== compute_range_percentages ====== #
def compute_range_percentages(
    frequency: Dict[int, int],
    total_numbers: int
) -> Dict[str, float]:
    """
    Compute the percentage of occurrences for specific ranges.

    Input:
    - frequency: dict mapping number → count
    - total_numbers: total count of all numbers

    Return:
    - Dict[str, float]: percentages for the keys:
        'eq_0', '0_5', '0_10', '0_20', '0_50'
    """
    if total_numbers <= 0:
        return {key: 0.0 for key in ['eq_0', '0_5', '0_10', '0_20', '0_50']}

    # count for exactly zero
    count_eq_0 = frequency.get(0, 0)

    # helper to sum counts in [0, upper]
    def count_upto(upper: int) -> int:
        return sum(cnt for num, cnt in frequency.items() if 0 <= num <= upper)

    # compute counts
    count_0_5 = count_upto(5)
    count_0_10 = count_upto(10)
    count_0_20 = count_upto(20)
    count_0_50 = count_upto(50)

    # compute percentages
    pct_eq_0 = (count_eq_0 / total_numbers) * 100
    pct_0_5 = (count_0_5 / total_numbers) * 100
    pct_0_10 = (count_0_10 / total_numbers) * 100
    pct_0_20 = (count_0_20 / total_numbers) * 100
    pct_0_50 = (count_0_50 / total_numbers) * 100

    return {
        'eq_0': pct_eq_0,
        '0_5': pct_0_5,
        '0_10': pct_0_10,
        '0_20': pct_0_20,
        '0_50': pct_0_50
    }
    


# ====== compute_list_statistics ====== #
def compute_list_statistics(
    df: pd.DataFrame,
    column: str = "Rank List"
) -> Dict[str, float]:
    """
    Compute numerical summary statistics for each list and across lists.

    Input:
    - df: pandas DataFrame with a column of integer lists
    - column: name of the target column (default 'Rank List')

    Return:
    - dict with keys:
        - mean_length: average length of each list
        - min_length: minimum list length
        - max_length: maximum list length
        - std_length: standard deviation of list lengths
        - mean_of_means: average of per-list means
        - min_mean: minimum per-list mean
        - max_mean: maximum per-list mean
        - std_of_means: standard deviation of per-list means
    """
    rank_list = df[column].dropna().tolist()
    lengths = [len(lst) for lst in rank_list]
    means = [np.mean(lst) if lst else 0 for lst in rank_list]

    mean_length = np.mean(lengths) if lengths else None
    min_length = min(lengths) if lengths else None
    max_length = max(lengths) if lengths else None
    std_length = np.std(lengths, ddof=0) if lengths else None

    mean_of_means = np.mean(means) if means else None
    min_mean = min(means) if means else None
    max_mean = max(means) if means else None
    std_of_means = np.std(means, ddof=0) if means else None

    return {
        "mean_length": mean_length,
        "min_length": min_length,
        "max_length": max_length,
        "std_length": std_length,
        "mean_of_means": mean_of_means,
        "min_mean": min_mean,
        "max_mean": max_mean,
        "std_of_means": std_of_means
    }

    

# ====== plot_frequency_curve ====== #
def plot_frequency_curve(
    frequency: Dict[int, int],
    save_fig: bool = False,
    plot_name: str = 'frequency_curve.png',
    show_fig: bool = True,
    log_x: bool = False,
    log_y: bool = False
) -> None:
    """
    Creates a frequency curve plot from a frequency dictionary.
    The x-axis represents the numbers and the y-axis represents their frequency.

    Input:
    - frequency: dict[int, int] mapping number → frequency
    - save_fig: if True, saves the plot in the 'Figures' folder
    - plot_name: filename for saving the plot (inside 'Figures/')
    - show_fig: if True, displays the plot; otherwise lo chiude
    - log_x: if True, usa scala logaritmica sull'asse x
    - log_y: if True, usa scala logaritmica sull'asse y
    """
    
    # Sort data
    sorted_numbers = sorted(frequency.keys())
    sorted_frequencies = [frequency[num] for num in sorted_numbers]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_numbers, sorted_frequencies, marker='o', linestyle='-',
             linewidth=2, markersize=6)

    # Apply log scales if requested
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    title = 'Frequency Curve'
    if log_x and log_y:
        title += ' (log–log scale)'
    elif log_x:
        title += ' (log x)'
    elif log_y:
        title += ' (log y)'
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_fig:
        os.makedirs('Figures', exist_ok=True)
        plot_path = os.path.join('Figures', plot_name)
        plt.savefig(plot_path)
        print(f"The plot has been saved to '{plot_path}'")

    if show_fig:
        plt.show()
    else:
        plt.close()
        


# ====== plot_smoothed_frequency_curve ====== #
def plot_smoothed_frequency_curve(
    frequency: Dict[int, int],
    smooth_window: int = 5,
    save_fig: bool = False,
    plot_name: str = 'frequency_curve_smoothed.png',
    show_fig: bool = True,
    log_x: bool = False,
    log_y: bool = False
) -> None:
    """
    Plot a smoothed frequency curve using a moving-average filter.

    Input:
    - frequency: dict[int, int] mapping number → frequency
    - smooth_window: size of the moving-average window (must be odd and >= 3)
    - save_fig: if True, save the figure under 'Figures/'
    - plot_name: filename for saving the plot
    - show_fig: if True, display the plot; otherwise close it
    - log_x: if True, use logarithmic scale on x-axis
    - log_y: if True, use logarithmic scale on y-axis
    """
    # Sort data
    x = np.array(sorted(frequency.keys()))
    y = np.array([frequency[k] for k in x])

    # Ensure window is odd and not larger than data
    if smooth_window < 3:
        raise ValueError("smooth_window must be >= 3")
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window > len(y):
        smooth_window = len(y) if len(y) % 2 == 1 else len(y) - 1

    # Compute moving average
    window = np.ones(smooth_window) / smooth_window
    y_smooth = np.convolve(y, window, mode='same')

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', alpha=0.3, label='Original', linewidth=1)
    plt.plot(x, y_smooth, marker='o', linestyle='-', label=f'Smoothed (w={smooth_window})', linewidth=2)

    # Apply log scales if requested
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    title = 'Smoothed Frequency Curve'
    if log_x and log_y:
        title += ' (log–log scale)'
    elif log_x:
        title += ' (log x)'
    elif log_y:
        title += ' (log y)'
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Save or show
    if save_fig:
        os.makedirs('Figures', exist_ok=True)
        path = os.path.join('Figures', plot_name)
        plt.savefig(path)
        print(f"Saved smoothed frequency plot to '{path}'")

    if show_fig:
        plt.show()
    else:
        plt.close()
 
 

# ====== plot_frequency_histogram ====== #
def plot_frequency_histogram(
    frequency: Dict[int, int],
    save_fig: bool = False,
    plot_name: str = 'frequency_histogram.png',
    show_fig: bool = True,
    log_x: bool = False,
    log_y: bool = False
) -> None:
    """
    Plots a histogram of the frequency distribution.

    Input:
    - frequency: dict[int, int] mapping number → frequency
    - save_fig: if True, saves the plot in the 'Figures' folder
    - plot_name: filename for saving the plot (inside 'Figures/')
    - show_fig: if True, displays the plot; otherwise it closes it
    - log_x: if True, applies log scale to x-axis
    - log_y: if True, applies log scale to y-axis
    """
    sorted_items = sorted(frequency.items())  # Sort by number
    numbers = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.bar(numbers, counts, width=0.8, color='skyblue', edgecolor='black')

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    title = 'Frequency Histogram'
    if log_x and log_y:
        title += ' (log–log scale)'
    elif log_x:
        title += ' (log x)'
    elif log_y:
        title += ' (log y)'
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_fig:
        os.makedirs('Figures', exist_ok=True)
        plot_path = os.path.join('Figures', plot_name)
        plt.savefig(plot_path)
        print(f"The histogram has been saved to '{plot_path}'")

    if show_fig:
        plt.show()
    else:
        plt.close()



# ====== plot_cdf ====== #
def plot_cdf(
    frequency: Dict[int, int],
    save_fig: bool = False,
    plot_name: str = 'cdf_curve.png',
    show_fig: bool = True,
    log_x: bool = False,
    log_y: bool = False
) -> None:
    """
    Plot the cumulative distribution function (CDF) for a frequency dictionary.

    Input:
    - frequency: dict mapping number → count
    - save_fig: if True, save the plot under 'Figures/'
    - plot_name: filename for saving the plot
    - show_fig: if True, display the plot; otherwise close it
    - log_x: if True, use logarithmic scale on x-axis
    - log_y: if True, use logarithmic scale on y-axis
    """
    # Sort values and compute cumulative frequencies
    numbers = np.array(sorted(frequency.keys()))
    counts = np.array([frequency[n] for n in numbers])
    cum_counts = np.cumsum(counts)
    cdf_values = cum_counts / cum_counts[-1]  # Normalize to [0,1]

    plt.figure(figsize=(10, 6))
    plt.plot(numbers, cdf_values, marker='o', linestyle='-')

    # Apply log scales if requested
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    # Labels and title
    plt.xlabel('Number')
    plt.ylabel('CDF (P(X ≤ x))')
    title = 'Cumulative Distribution Function'
    if log_x and log_y:
        title += ' (log–log scale)'
    elif log_x:
        title += ' (log x)'
    elif log_y:
        title += ' (log y)'
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save or show figure
    if save_fig:
        import os
        os.makedirs('Figures', exist_ok=True)
        path = os.path.join('Figures', plot_name)
        plt.savefig(path)
        print(f"Saved CDF plot to '{path}'")

    if show_fig:
        plt.show()
    else:
        plt.close()