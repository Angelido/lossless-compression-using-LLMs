import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================
# This file is used to extract information from the LLM+compression
# results and to generate plots. In particular, it plots on the same
# graph the results for a specific language and a specific compression
# algorithm across all the models used.
# =======================================================


#============== model_name_preprocesing ================
def model_name_preprocesing(
    model: pd.DataFrame
) -> pd.DataFrame:
    """
    Preprocess the model names in the DataFrame.
    
    Input:
    - model (pd.DataFrame): DataFrame containing model names.
    
    Return:
    - pd.DataFrame: DataFrame with preprocessed model names.
    """

    #======== Check if DataFrame contains required columns =======
    
    if "output_file" not in model.columns:
        raise ValueError("DataFrame must contain 'output_file' column.")
    
    if "model" not in model.columns:
        raise ValueError("DataFrame must contain 'model' column.")
    
    if not pd.api.types.is_string_dtype(model["output_file"]):
        raise TypeError("'output_file' column must be of string type.")
    
    if not pd.api.types.is_string_dtype(model["model"]):
        raise TypeError("'model' column must be of string type.")
    
    #======= filtering StarCoder model names based on output_file =======
    
    # Define conditions for filtering and suffixes
    conditions = [
        data["output_file"].str.startswith("Results/CompressedFiles/StarCoder16"),
        data["output_file"].str.startswith("Results/CompressedFiles/StarCoder4"),
    ]
    suffixes = ["16", "4"]
    
    # Update the 'model' column based on conditions and suffixes
    data["model"] = np.select(conditions, [data["model"] + s for s in suffixes], default=data["model"])
    
    #======= Change model names for specific models =======
    
    models_name={
        "TheBloke/deepseek-coder-1.3b-base-AWQ": "DeepSeekCoder 1.3B",
        "bigcode/starcoder2-3b16": "StarCoder2 3B-16",
        "bigcode/starcoder2-3b4": "StarCoder2 3B-4",
        "unsloth/Llama-3.2-1B-bnb-4bit": "Llama 3.2 1B",
        "microsoft/unixcoder-base": "UnixCoder Base",
        "microsoft/phi-2": "Phi-2",
        "PrunaAI/ibm-granite-granite-3b-code-base-bnb-4bit-smashed": "Granite 3B",
        "PrunaAI/google-codegemma-2b-AWQ-4bit-smashed": "CodeGemma 2B",
    }
    
    data["model"]= data["model"].replace(models_name)
    
    return model



#============== create_data_subset ================
def create_data_subset(
    data: pd.DataFrame, 
    language: str, 
    compression: str, 
    binary: bool
) -> pd.DataFrame:
    """
    Create a subset of the DataFrame based on language, compression, and binary status.
    
    Input:
    - data (pd.DataFrame): DataFrame containing the data.
    - language (str): Language to filter by.
    - compression (str): Compression method to filter by.
    - binary (bool): Binary status to filter by.
    
    Return:
    - pd.DataFrame: Filtered DataFrame.
    """
    
    return data[(data["language"] == language) & 
                (data["compression"] == compression) & 
                (data["binary"] == binary)]



#============== plot_model_performance ================
def plot_model_performance(
    df: pd.DataFrame
) -> None:
    """
    Plots a scatter plot of Throughput vs Compressed Size for a given DataFrame,
    with model names annotated on the points.
    
    Input:
    - df (pd.DataFrame): DataFrame containing 'compressed_size_MB', 'Throughput_MBps', 'model', and 'language' columns.
    If the DataFrame is empty, it will display 'Unknown' as the language.
    """
    
    # Check if DataFrame is empty
    Language = df['language'].iloc[0] if not df.empty else 'Unknown'
    
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot
    sns.scatterplot(
        data=df,
        x='compressed_size_MB',
        y='Throughput_MBps',
        style='model',
        s=100
    )
    # Annotate the points with model names
    for _, row in df.iterrows():
        plt.text(
            row['compressed_size_MB'],
            row['Throughput_MBps'],
            row['model'],
            fontsize=9,
            ha='right',
            va='bottom'
        )
    plt.xlabel("Compressed Size (MB)")
    plt.ylabel("Throughput (MB/s)")
    plt.title(f"{Language} Compressed Size vs Throughput")
    plt.grid(True)
    
    plt.show()


#============== plot_model_performance_multi ================
def plot_model_performance_multi(
    *dfs: pd.DataFrame, 
    palette: str = "tab10", 
    annotate: bool = True,
    save: bool = False,
    save_path: str = "model_performance_plot.png"
) -> None:
    """
    Plots a scatter plot of Throughput vs Compressed Size for each 'language' and 'model',
    in a 3-row × 2-column layout, without a legend but with model names directly on the points.
    
    Input:
    - *dfs: DataFrames for each language to plot.
    - palette (str): Seaborn color palette to use for different models.
    - annotate (bool): Whether to annotate points with model names.
    """
    
    # Check if at least one DataFrame is provided
    df = pd.concat(dfs, ignore_index=True)
    
    # Take the first 6 languages
    langs = sorted(df['language'].unique())[:6]
    n = len(langs)
    
    # Prepare colors for models using the specified palette
    models = df['model'].unique()
    colors = dict(zip(models, sns.color_palette(palette, len(models))))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.flatten()
    
    # Draw scatter plots for each language
    for i, lang in enumerate(langs):
        ax = axes[i]
        sub = df[df['language'] == lang]
        sns.scatterplot(
            data=sub,
            x="compressed_size_MB",
            y="Throughput_MBps",
            hue="model",
            palette=colors,
            s=80,
            ax=ax,
            legend=False
        )
        # Write the model names on the points
        if annotate:
            for _, row in sub.iterrows():
                ax.text(
                    row['compressed_size_MB'],
                    row['Throughput_MBps'],
                    row['model'],
                    fontsize=9,
                    alpha=0.8,
                    ha='left',
                    va='bottom'
                )
        
        ax.set_title(lang, fontsize=14, fontweight='bold')
        ax.set_xlabel("Compressed Size (MB)", fontsize=11)
        ax.set_ylabel("Throughput (MB/s)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Remove empty subplots if there are fewer than 6 languages
    for j in range(n, 6):
        fig.delaxes(axes[j])
        
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    #plt.suptitle("Model Performance by Language", fontsize=16, y=0.98)
    plt.show()
    
    
    
#============== plot_model_performance_multi_percentage ================
def plot_model_performance_multi_percentage(
    *dfs: pd.DataFrame, 
    palette: str = "tab10", 
    annotate: bool = True,
    save: bool = False,
    save_path: str = "model_performance_plot.png"
) -> None:
    """
    Plots a scatter plot of Throughput vs Compression Percentage for each 'language' and 'model',
    in a 3-row × 2-column layout.
    """
    
    # Unione dei DataFrame e calcolo della percentuale di compressione
    df = pd.concat(dfs, ignore_index=True)
    df['compression_percentage'] = (df['compressed_size_bytes'] / df['original_size_bytes']) * 100

    
    # Prende i primi 6 linguaggi
    langs = sorted(df['language'].unique())[:6]
    n = len(langs)
    
    # Prepara i colori per i modelli
    models = df['model'].unique()
    colors = dict(zip(models, sns.color_palette(palette, len(models))))
    
    # Crea la figura
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.flatten()
    
    # Plot per ogni linguaggio
    for i, lang in enumerate(langs):
        ax = axes[i]
        sub = df[df['language'] == lang]
        
        sns.scatterplot(
            data=sub,
            x="compression_percentage",
            y="Throughput_MBps",
            hue="model",
            palette=colors,
            s=80,
            ax=ax,
            legend=False
        )
        
        if annotate:
            for _, row in sub.iterrows():
                ax.text(
                    row['compression_percentage'],
                    row['Throughput_MBps'],
                    row['model'],
                    fontsize=9,
                    alpha=0.8,
                    ha='left',
                    va='bottom'
                )
        
        ax.set_title(lang, fontsize=14, fontweight='bold')
        ax.set_xlabel("Compression Percentage (%)", fontsize=11)  # Etichetta modificata
        ax.set_ylabel("Throughput (MB/s)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Tick equidistanti e ben formattati
        min_x = sub['compression_percentage'].min()
        max_x = sub['compression_percentage'].max()
        xticks = np.arange(np.floor(min_x), np.ceil(max_x) + 1)

        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x)}%" for x in xticks])

    
    # Rimuovi subplot vuoti
    for j in range(n, 6):
        fig.delaxes(axes[j])
        
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



#============== main ================
if __name__ == "__main__":
    
    compression_to_study = "bzip2-3"
    # compression_to_study = "zstd12"
    binary = True
    
    # data = pd.read_csv('Results/Compression_info.csv')
    data = pd.read_csv('Compression_info.csv')
    
    # Change the model names in the DataFrame
    data= model_name_preprocesing(data)
    
    # Create new columns for throughput and compressed size
    data["compressed_size_MB"] = data["compressed_size_bytes"] / (1024 * 1024)
    data["Throughput_MBps"] = 100 / data["total_time_s"]
    
    # Define the languages to analyze
    languages = ["Python", "C", "CSharp", "Java", "JavaScript", "Cpp"]
    
    data_lang={}
    
    # Create a subset of the data for each language
    for lang in languages:
        data_lang[lang] = create_data_subset(data, lang, compression_to_study, binary)
    
    # plot_model_performance(data_lang["Python"])
    
    plot_model_performance_multi_percentage(
        data_lang["Python"], 
        data_lang["C"], 
        data_lang["CSharp"], 
        data_lang["Java"], 
        data_lang["JavaScript"],
        data_lang["Cpp"],
        save=True,
        save_path=f"{compression_to_study}_performance_plot.png"
    )
    