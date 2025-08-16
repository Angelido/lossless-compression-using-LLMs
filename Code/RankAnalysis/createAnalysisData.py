import pandas as pd
import matplotlib.pyplot as plt
from rankAnalysisUtility import (
    read_rank_list_to_dataframe,
    get_file_size_mb,
    analyze_rank_list_df,
    compute_range_percentages,
    compute_list_statistics,
    plot_frequency_curve,
    plot_frequency_histogram,
    plot_cdf
) 

def collect_model_stats(
    folder_path: str,
    model_names: list[str],
    extension: str = "_rank_list.txt",
    column: str = "Rank List"
) -> pd.DataFrame:
    """
    Loop over model_names, read each rank list, compute all metrics,
    and return a pandas DataFrame with one row per model.
    
    Input:
    - folder_path: Path to the folder containing rank list files.
    - model_names: List of model names (without file extension).
    - extension: File extension of the rank list files.
    - column: Column name in the DataFrame to analyze.
    
    Return:
    - df_stats: DataFrame containing aggregated statistics for each model.
    """
    records = []
    
    for model in model_names:
        file_name = f"{folder_path}{model}{extension}"
        file_size = get_file_size_mb(file_name)
        df = read_rank_list_to_dataframe(file_name)
        
        # Global metrics
        rank_results = analyze_rank_list_df(df, column)
        freq = rank_results["frequency"]
        total = rank_results["total_numbers"]
        
        # Range percentages
        pct = compute_range_percentages(freq, total)
        
        # Per-list metrics
        list_stats = compute_list_statistics(df, column)
        
        # Aggregate into a single record
        rec = {
            "model": model,
            "file_size_MB": file_size,
            
            # Range percentages
            "pct_eq_0": pct["eq_0"],
            "pct_0_1": pct["0_1"],
            "pct_0_3": pct["0_3"],
            "pct_0_7": pct["0_7"],
            "pct_0_15": pct["0_15"],
            "pct_0_31": pct["0_31"],
            "pct_0_63": pct["0_63"],
            
            # Overall distribution
            "min_value": rank_results["min"],
            "max_value": rank_results["max"],
            "mean": rank_results["mean"],
            "median": rank_results["median"],
            "mode": rank_results["mode"],
            "variance": rank_results["variance"],
            "std_dev": rank_results["std"],
            "total_sum": rank_results["total_sum"],
            "num_lists": rank_results["num_lists"],
            "total_numbers": rank_results["total_numbers"],
            
            # Per-list statistics
            "mean_list_length": list_stats["mean_length"],
            "min_list_length": list_stats["min_length"],
            "max_list_length": list_stats["max_length"],
            "std_list_length": list_stats["std_length"],
            "mean_of_means": list_stats["mean_of_means"],
            "min_mean": list_stats["min_mean"],
            "max_mean": list_stats["max_mean"],
            "std_of_means": list_stats["std_of_means"],
        }
        
        records.append(rec)
    
    # Build DataFrame
    df_stats = pd.DataFrame.from_records(records)
    return df_stats


if __name__ == "__main__":
    
    folder_path = "TextInformation/"
    model_names = [
        "DeepSeek", 
        "DeepSeekQuantized", 
        "CodeGemma",
        "CodeGemmaQuantized",
        "CodeT5",
        "Gemma",
        "GemmaQuantized",
        "Granite",
        "GraniteCode",
        "GraniteCodeQuantized",
        "Llama1B",
        "Llama1BQuantized",
        "Llama3B",
        "Llama3BQuantized",
        "Phi2",
        "Qwen3",
        "SmolLM3",
        "StarCoder2_4",
        "StarCoder2_8",
        "StarCoder2_16",
        "StarCoder2_32",
        "UnixCoder"
    ]
    extension = "_rank_list.txt"
    
    df_stats = collect_model_stats(folder_path, model_names, extension)
    print(df_stats.head())
    
    # Save the DataFrame to a CSV file
    output_csv = "All_Models_Statistics.csv"
    df_stats.to_csv(output_csv, index=False)
    print(f"Saved aggregated stats for {len(model_names)} models to '{output_csv}'")