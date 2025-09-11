import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import warnings


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


# ============= plot_rank_heatmap (cumulativa, ordinata) ================
def plot_rank_heatmap(
    data: pd.DataFrame, 
    bin_cols: list[str], 
    bin_labels: list[str],
    save: bool = False,
    save_path: str = "rank_bin_heatmap_cumulative.png"
) -> None:
    """
    Heatmap cumulativa per riga, ordinata per la prima colonna (asc).
    Annotazioni: 'cumulata (singola)'. Gestisce percentuali o frazioni.
    """

    # Select & sort
    df_bins = data[['model'] + bin_cols].copy()
    df_bins = df_bins.sort_values(by=bin_cols[0], ascending=False)
    df_bins.set_index('model', inplace=True)

    # Fractions vs percent
    vals = df_bins[bin_cols].astype(float)
    max_val = np.nanmax(vals.to_numpy()) if vals.size else 0.0
    use_percent = max_val <= 1.0
    scale = 100.0 if use_percent else 1.0
    cbar_label = "% of items" if use_percent else "Value"

    # Scaled values and cumulative per row
    vals_scaled = vals * scale
    cum_scaled = vals_scaled.cumsum(axis=1)
    cum_scaled.columns = bin_labels  # x labels

    annot = pd.DataFrame(index=cum_scaled.index, columns=cum_scaled.columns, dtype=object)
    for j, col in enumerate(bin_cols):
        cum_col = cum_scaled.iloc[:, j]
        uniq_col = vals_scaled.iloc[:, j]
        if use_percent:
            annot.iloc[:, j] = [f"{c:.1f}% ({u:.1f}%)" for c, u in zip(cum_col, uniq_col)]
        else:
            annot.iloc[:, j] = [f"{c:.3g} ({u:.3g})" for c, u in zip(cum_col, uniq_col)]

    # Plot
    plt.figure(figsize=(10, max(2, len(df_bins) * 0.5 + 2)))
    ax = sns.heatmap(
        cum_scaled,                  # numeric values for color
        annot=annot.to_numpy(),      # string annotations
        fmt="",                      # strings already formatted
        cmap="YlOrRd",
        cbar_kws={'label': cbar_label},
        linewidths=0.3,
        linecolor="white"
    )
    ax.set_title("Heatmap - Cumulative Rank Bin Distribution")
    ax.set_xlabel("Rank Range")
    ax.set_ylabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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



# ============= plot_llm_throughput_vs_params_curve ================
def plot_llm_throughput_vs_params_curve(
    df: pd.DataFrame,
    palette: str = "tab10",
    annotate_points: bool = False,
    save: bool = False,
    save_path: str = "llm_throughput_vs_memory.png",
    legend_title: str = "Model",
    logy_scale: bool = False,
    logx_scale: bool = False,
    add_trend: str | None = None,   # None, "loess". Don't use linear: doesn't work
    loess_frac: float = 0.3         # smoothing fraction for LOESS/fallback
) -> None:
    """
    Scatter Throughput vs Memory, colorato per modello, con linea tratteggiata di trend opzionale.
    - Se logx_scale/logy_scale sono veri, il fit avviene nello stesso spazio (log o lineare).
    - add_trend: None or "loess"= smoothing locale (richiede statsmodels; fallback a media mobile)
    """
    required = {"model", "throughput_MB_s", "memory_usage_GB"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required}")

    models = df['model'].unique()
    colors = dict(zip(models, sns.color_palette(palette, len(models))))

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(right=0.77)  # spazio per la legenda esterna

    # Scatter
    for model in models:
        sub = df[df['model'] == model]
        ax.scatter(
            sub['memory_usage_GB'],
            sub['throughput_MB_s'],
            label=model,
            color=colors[model],
            s=100,
            alpha=0.8
        )
        if annotate_points:
            for _, row in sub.iterrows():
                ax.text(
                    row['memory_usage_GB'],
                    row['throughput_MB_s'],
                    model,
                    fontsize=9,
                    ha='left',
                    va='bottom'
                )

    ax.set_xlabel("Model Memory Usage (GB)", fontsize=12)
    ax.set_ylabel("LLM Throughput (MB/s)", fontsize=12)
    ax.set_title("LLM Throughput vs Model Memory Usage", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legenda esterna
    legend = ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.
    )
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass

    # Scale log se richiesto
    if logx_scale:
        ax.set_xscale('log', base=10)
    if logy_scale:
        ax.set_yscale('log', base=10)

    # ---------- Trend line (opzionale) ----------
    if add_trend in {"linear", "loess"}:
        # Prepare data for fitting (respecting log flags)
        x = df["memory_usage_GB"].to_numpy(dtype=float)
        y = df["throughput_MB_s"].to_numpy(dtype=float)

        # Mask invalid for logs
        mask = np.isfinite(x) & np.isfinite(y)
        if logx_scale:
            mask &= x > 0
        if logy_scale:
            mask &= y > 0

        x, y = x[mask], y[mask]
        if x.size >= 2:
            # Transform according to log flags
            X = np.log10(x) if logx_scale else x
            Y = np.log10(y) if logy_scale else y

            # Sort by X for a nice monotone line
            order = np.argsort(X)
            Xs, Ys = X[order], Y[order]
            xs = (10**Xs) if logx_scale else Xs   # x in display space
            # Fit/predict
            if add_trend == "linear":
                # Linear regression in transformed space
                a, b = np.polyfit(X, Y, 1)
                Yhat = a + b * Xs
                # R^2 in transformed space
                r = np.corrcoef(Y, (a + b * X))[0, 1]
                r2 = float(r*r) if np.isfinite(r) else np.nan
                # Back-transform Y for plotting
                ys = 10**Yhat if logy_scale else Yhat
                label_tr = f"Trend (linear)  R²={r2:.3f}"
            else:  # "loess"
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    # LOWESS in transformed space
                    frac = np.clip(loess_frac, 0.05, 0.9)
                    sm = lowess(Y, X, frac=frac, return_sorted=False)
                    # Interpolate on sorted grid for smooth line
                    # Qui usiamo un semplice re-ordinamento
                    Yhat = sm[np.argsort(X)]
                    ys = 10**Yhat if logy_scale else Yhat
                    # Pseudo R^2 (correlazione tra Y e sm) nel transform space
                    r = np.corrcoef(Y, sm)[0, 1]
                    r2 = float(r*r) if np.isfinite(r) else np.nan
                    label_tr = f"Trend (LOESS)  R²={r2:.3f}"
                except Exception:
                    # Fallback: moving average smoothing in transformed space
                    import pandas as pd
                    win = max(3, int(len(Ys) * loess_frac))
                    Yhat = pd.Series(Ys).rolling(window=win, min_periods=1, center=True).mean().to_numpy()
                    ys = 10**Yhat if logy_scale else Yhat
                    r = np.corrcoef(Y, np.interp(X, Xs, Yhat))[0, 1]
                    r2 = float(r*r) if np.isfinite(r) else np.nan
                    label_tr = f"Trend (moving avg)  R²={r2:.3f}"

            # Draw dashed trend line
            ax.plot(
                xs, ys,
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                label=label_tr
            )

            # Refresh legend to include trend line
            leg = ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1),
                            loc="upper left", borderaxespad=0.)
            try:
                leg._legend_box.align = "left"
            except Exception:
                pass
        else:
            warnings.warn("Not enough points to compute trend line.")

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============= plot_llm_throughput_vs_params ================
def plot_llm_throughput_vs_params(
    df: pd.DataFrame,
    palette: str = "tab10",
    annotate_points: bool = False,
    save: bool = False,
    save_path: str = "llm_throughput_vs_memory.png",
    legend_title: str = "Model",
    logy_scale: bool = False,
    logx_scale: bool = False,
    add_hline: bool = False,        
    hline_y: float = 0.095,
    highlight_above: bool = True    
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
    required = {"model", "throughput_MB_s", "memory_usage_GB"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required}")

    models = df['model'].unique()
    colors = dict(zip(models, sns.color_palette(palette, len(models))))

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(right=0.77)  # Leave space for the legend
    
    models_above = set()

    for model in models:
        sub = df[df['model'] == model]
        ax.scatter(
            sub['memory_usage_GB'],
            sub['throughput_MB_s'],
            label=model,
            color=colors[model],
            s=100,
            alpha=0.8
        )
        # cerchi rossi per punti sopra soglia
        if add_hline and highlight_above:
            mask = sub['throughput_MB_s'] > hline_y
            if mask.any():
                models_above.add(model)
                ax.scatter(
                    sub.loc[mask, 'memory_usage_GB'],
                    sub.loc[mask, 'throughput_MB_s'],
                    s=220,
                    facecolors='none',
                    edgecolors='red',
                    linewidths=1.8
                )

        if annotate_points:
            for _, row in sub.iterrows():
                ax.text(
                    row['memory_usage_GB'],
                    row['throughput_MB_s'],
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
    
    # Scale log se richiesto
    if logx_scale:
        ax.set_xscale('log', base=10)  # in versioni vecchie: ax.set_xscale('log') o basex
    if logy_scale:
        ax.set_yscale('log', base=10)
    
    if add_hline:
        ax.axhline(
            y=hline_y,
            color='red',
            linestyle='--',
            linewidth=1.75,
            alpha=0.9,
            label=f"y = {hline_y:g}"
        )

    # legenda con proxy per "Above threshold"
    handles, labels = ax.get_legend_handles_labels()
    if add_hline and highlight_above:
        proxy = Line2D([], [], linestyle='None', marker='o', markersize=8,
                       markerfacecolor='none', markeredgecolor='red',
                       markeredgewidth=1.8, label=f"Above {hline_y:g}")
        handles.append(proxy)
        labels.append(f"Above {hline_y:g}")

    # evidenzia in legenda i modelli sopra soglia
    for txt in legend.get_texts():
        name = txt.get_text()
        if name in models_above:
            txt.set_color('red')
            txt.set_fontweight('bold')
            try:
                txt.set_underline(True)  # se supportato dalla tua versione di matplotlib
            except Exception:
                pass
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass

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


    plot_rank_heatmap(data_ability, bin_cols, bin_labels, True)
    plot_rank_radar(data_ability, bin_cols, bin_labels)
    plot_rank_cumulative(data_ability, bin_cols, bin_labels)
    
    # Collect model statistics and plot throughput vs memory usage
    data_plot = data[["model", "llm_throughput", "memory_usage_GB"]]
    
    plot_llm_throughput_vs_params(
        data_plot, 
        save=True, 
        save_path="llm_throughput_vs_memory.png"
    )
    plot_llm_throughput_vs_params_curve(
        data_plot, 
        save=True, 
        save_path="llm_throughput_vs_memory_with_curve.png", 
        add_trend="loess", 
        loess_frac=0.5
    )
    plot_llm_throughput_vs_params(
        data_plot, 
        save=True, 
        save_path="llm_throughput_vs_memory_final_models.png",
        add_hline=True,
        hline_y=0.095,
        highlight_above=True
    )