import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ============= compute_rank_bins ================
def compute_rank_bins(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes rank bins from the percentage columns in the DataFrame.
    Renames the columns for clarity and returns the modified DataFrame.
    
    Input:
    - data: DataFrame containing percentage columns like 'pct_eq_0', 'pct_0_1', etc.
    
    Return:
    - df: DataFrame with renamed columns for rank bins.
    """
    df = data.copy()

    # Rinominazione coerente per leggibilità
    df["bin_eq_0"] = df["pct_eq_0"]
    df["bin_0_1"]  = df["pct_0_1"] - df["pct_eq_0"]
    df["bin_1_3"]  = df["pct_0_3"] - df["pct_0_1"]
    df["bin_3_7"]  = df["pct_0_7"] - df["pct_0_3"]
    df["bin_7_15"] = df["pct_0_15"] - df["pct_0_7"]
    df["bin_15_31"] = df["pct_0_31"] - df["pct_0_15"]
    df["bin_31_63"] = df["pct_0_63"] - df["pct_0_31"]

    return df


# ============= plot_mean_vs_meanofmeans ================
def plot_mean_vs_meanofmeans(
    data: pd.DataFrame,
    models: list[str] = None,
    figsize: tuple[int,int] = (12, 6),
    save: bool = False,
    save_path: str = "mean_vs_meanofmeans.png"
) -> None:
    """
    Per ogni modello produce un grafico a barre affiancate:
    - media generale (`mean`)
    - media dei mean delle liste (`mean_of_means`)
    
    Args:
        data (pd.DataFrame): DataFrame con almeno le colonne 'model', 'mean', 'mean_of_means'
        models (list[str], optional): quali modelli includere (default = tutti)
        figsize (tuple, optional): dimensione figura (default = (12,6))
        save (bool, optional): salva il grafico come file (default = False)
        save_path (str, optional): nome file (default = ...)
    """
    df = data.copy()
    if models is not None:
        df = df[df['model'].isin(models)]
    df = df[['model', 'mean', 'mean_of_means']].set_index('model')

    labels = df.index.tolist()
    mean_vals = df['mean'].values
    mm_vals = df['mean_of_means'].values
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, mean_vals, width, label='mean')
    ax.bar(x + width/2, mm_vals, width, label='mean_of_means')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Mean vs Mean of Means per Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# ============= plot_rank_heatmap ================
def plot_rank_heatmap(data: pd.DataFrame, bin_cols: list[str], bin_labels: list[str]) -> None:
    """
    Plotta una heatmap delle distribuzioni nei bin di rank.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    df_bins = data[['model'] + bin_cols].copy()
    df_bins.set_index('model', inplace=True)

    plt.figure(figsize=(10, len(df_bins) * 0.5 + 2))
    sns.heatmap(df_bins, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': '% of items'})
    plt.title("Heatmap - Rank Bin Distribution")
    plt.xlabel("Rank Range")
    plt.ylabel("Model")
    plt.xticks(ticks=np.arange(len(bin_cols)) + 0.5, labels=bin_labels, rotation=45)
    plt.tight_layout()
    plt.show()



# ============= plot_rank_radar ================
def plot_rank_radar(data: pd.DataFrame, bin_cols: list[str], bin_labels: list[str]) -> None:
    """
    Plotta un radar plot delle distribuzioni nei bin di rank.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df_bins = data[['model'] + bin_cols].copy()
    df_bins.set_index('model', inplace=True)

    labels = bin_labels
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in df_bins.index:
        values = df_bins.loc[model].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Radar Plot - Rank Bin Distribution")
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()



# ============= plot_rank_cumulative ================
def plot_rank_cumulative(data: pd.DataFrame, bin_cols: list[str], bin_labels: list[str]) -> None:
    """
    Plotta la distribuzione cumulativa dei bin di rank per ciascun modello.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df_bins = data[['model'] + bin_cols].copy()
    df_bins.set_index('model', inplace=True)

    plt.figure(figsize=(10, 6))
    for model in df_bins.index:
        cumulative = np.cumsum(df_bins.loc[model].values)
        plt.plot(bin_labels, cumulative, marker='o', label=model)

    plt.title("Cumulative Rank Distribution")
    plt.xlabel("Rank Bin")
    plt.ylabel("Cumulative % of items")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



# ============= plot_llm_throughput_vs_params ================
def plot_llm_throughput_vs_params(
    df: pd.DataFrame,
    palette: str = "tab10",
    annotate_points: bool = False,
    save: bool = False,
    save_path: str = "llm_throughput_vs_memory.png",
    legend_title: str = "Model"
) -> None:
    """
    Plots scatter of Throughput vs Model Memory Usage, colored by model.
    Adds a legend with model names outside the plot area.
    
    Input:
    - df: DataFrame containing 'model', 'llm_throughput', and 'memory_usage_GB' columns.
    - palette: Color palette for the models.
    - annotate_points: If True, annotates points with model names.
    - save: If True, saves the plot to a file.
    - save_path: Path to save the plot if `save` is True.
    - legend_title: Title for the legend.
    """
    required = {"model", "llm_throughput", "memory_usage_GB"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required}")

    models = df['model'].unique()
    colors = dict(zip(models, sns.color_palette(palette, len(models))))

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(right=0.77)  # Leave space for the legend

    for model in models:
        sub = df[df['model'] == model]
        ax.scatter(
            sub['memory_usage_GB'],
            sub['llm_throughput'],
            label=model,
            color=colors[model],
            s=100,
            alpha=0.8
        )
        if annotate_points:
            for _, row in sub.iterrows():
                ax.text(
                    row['memory_usage_GB'],
                    row['llm_throughput'],
                    model,
                    fontsize=9,
                    ha='left',
                    va='bottom'
                )

    ax.set_xlabel("Model Memory Usage (GB)", fontsize=12)
    ax.set_ylabel("LLM Throughput", fontsize=12)
    ax.set_title("LLM Throughput vs Model Memory Usage", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # External legend
    legend = ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1),  # Position outside the plot
        loc="upper left",
        borderaxespad=0.
    )
    legend._legend_box.align = "left"

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


#============== main ================
if __name__ == "__main__":
    
    data=pd.read_csv('All_Models_Statistics.csv')
    
    plot_mean_vs_meanofmeans(data)
    
    # Collect model statistics and compute rank bins
    data_ability = compute_rank_bins(data)
    print(data_ability.head())
    models = [
        "Llama3B",
        "Llama3BQuantized",
        "Phi2",
        "Qwen3",
        "SmolLM3",
        "StarCoder2_4",
        "StarCoder2_8",
        "StarCoder2_16",
        "StarCoder2_32"
    ]
    bin_cols = ['bin_eq_0', 'bin_0_1', 'bin_1_3', 'bin_3_7', 'bin_7_15', 'bin_15_31', 'bin_31_63']
    bin_labels = ['=0', '0–1', '1–3', '3–7', '7–15', '15–31', '31–63']


    plot_rank_heatmap(data_ability, bin_cols, bin_labels)
    plot_rank_radar(data_ability, bin_cols, bin_labels)
    plot_rank_cumulative(data_ability, bin_cols, bin_labels)
    
    # Collect model statistics and plot throughput vs memory usage
    data_plot = data[["model", "llm_throughput", "memory_usage_GB"]]
    
    plot_llm_throughput_vs_params(data_plot, save=True, save_path="llm_throughput_vs_memory.png")