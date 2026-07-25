import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from src.helpers import line_utils, data_utils, matrix_utils, plot_utils, stats_utils
from src.algorithms import classifier, estimators, reconstructor


def single_run_orthogonal(settings: tuple, data_dir, outdir) -> dict:

    filename = settings[0]
    linespacing = settings[1]
    multiple = settings[2]
    fxn = settings[3][0]
    # fxn_method = settings[3][1]
    if_fft_method = settings[3][1]
    sampling_method = settings[4]
    normalize_residual = settings[5]
    bootstrap = settings[6]

    # load the bathymetry data from the file
    bathy_dict = data_utils.load_file(filename=filename,
                        folder=str(data_dir),
                        verbose=False)

    depth = data_utils.remove_edge_Nans(depth=bathy_dict['depth'], ndv=bathy_dict['ndv'])
    resolution = bathy_dict['resolution']

    # Line Selection and Statistical Reconstruction
    column_indices = matrix_utils.get_column_indices(array_len=depth.shape[1],
                                        resolution=resolution,
                                        linespacing_meters=linespacing)

    # Compute uncertainties
    line_spacing = np.array([linespacing])
    train, test, incs, idxs, _ = line_utils.select_lines(data=depth, Mline_spacings=line_spacing, x_res=resolution,
                                                            ndv=bathy_dict['ndv'], MlineOnly=True)
    filled = reconstructor.fill_across_track(data=train[0], across_idx=idxs[0], method=sampling_method, overlap_bias='mean')
    rows_keep = line_utils.clip_rows_with_ndvs_indices(filled, ndv=bathy_dict['ndv'])
    filled_clipped = filled[rows_keep, :]
    depth = depth[rows_keep,:]

    # Create mask
    interp_mask = np.zeros(depth.shape, dtype=bool)
    interp_mask[:, column_indices] = True

    #  Testbed Classifier 
    spectral_slope = classifier.seabed_classifier(filled_clipped, seabed= filename.replace(".tif", ""), outdir=None, title=None, resolution=resolution, plot=False)
    if spectral_slope <= 2.7:
        if if_fft_method:
            fxn_method = 'PSD95'
        else:
            fxn_method = 'SAR'
    elif spectral_slope > 2.7:
        if if_fft_method:
            fxn_method = 'PSD99'
        else:
            fxn_method = 'SER'

    # Compute uncertainty
    uncertainty_segment_data = matrix_utils.matrix2strip(filled_clipped,
                                            column_indices=column_indices,
                                            multiple=multiple)
    uncertainty_data, _ = estimators.compute_residual(uncertainty_segment_data, normalize_residual=normalize_residual)

    # # Compute Residuals
    residual_segment_data = matrix_utils.matrix2strip(depth,
                                                      column_indices=column_indices,
                                                      multiple=multiple)

    residual_data, _ = estimators.compute_residual(residual_segment_data, normalize_residual=normalize_residual)

    # Different inputs for Spectral and Spatial functions

    if if_fft_method:
        strip = fxn(data=uncertainty_data,
                    resolution=resolution,
                    multiple=multiple, method=fxn_method)
    else:
        strip = fxn(data=uncertainty_data,
                    multiple=multiple, method=fxn_method, line_spacing=linespacing)

    if strip.shape != residual_data.shape:
        raise ValueError(
            f"Uncertainty strip shape {strip.shape} does not match residual data shape {residual_data.shape}",
            f"filename: {filename}, linespacing: {linespacing}, multiple: {multiple}, method: {fxn_method}, sampling: {sampling_method}")

    # Reconstruct Matrix from Strip
    residuals = matrix_utils.strip2matrix(data_strip=residual_data,
                             original_shape=depth.shape,
                             column_indices=column_indices)

    output_uncertainty = matrix_utils.strip2matrix(data_strip=strip,
                                      original_shape=depth.shape,
                                      column_indices=column_indices)

    stats, uncertainty_cleaned, residuals_cleaned = stats_utils.uncertainty_comparison(residuals=residuals,
                                              uncertainties=output_uncertainty, interp_mask=interp_mask)

    if bootstrap:
        ci_stats = stats_utils.bootstrap_by_gaps(residuals, output_uncertainty, column_indices, n_boot=500, ci=95, interp_mask=interp_mask, min_block_width=2, fraction=0.6) # 500

    results = {"filename": filename,
               "linespacing": linespacing,
               "multiple": multiple,
               "sampling_method": sampling_method,
               "method": fxn_method,
               "residuals": residuals_cleaned,
               "uncertainty": uncertainty_cleaned,
               "stats": stats,
               "spectral_slope": spectral_slope,
               }

    if bootstrap:
        results["ci_stats"] = ci_stats

    return results

def create_spacing_by_depth(mean, multiplier, res):
    div = np.floor(mean/res)
    mean = div*res
    result = np.linspace(mean, mean*multiplier, multiplier)
    return result[1:]

def create_combined_spacings (spacing_by_depth, spacing_wanted, minimum_spacing):
    spacing = np.unique(np.sort(np.concatenate((spacing_by_depth, spacing_wanted))))
    spacing = spacing[spacing>=minimum_spacing]
    return spacing 


### Pass Coverage Plot with 95% Bootstrap CI
def plot_pass_by_method_grouped_spacing(
    df,
    outdir=None,
    pass_percent=90,
    sampling=None,
):

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12
    })

    plt.rcParams.update({
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",})

    required_cols = [
        'Seabed',
        'Method',
        'Spacing',
        'CI_Pass Mean',
        'CI_Lower',
        'CI_Upper'
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    colors = {
        'SAR': '#4CD92B',   # green
        'SER': '#4CD92B',   #green # blue '#4472C4'
        'PSD95': '#E97132',   # orange
        'PSD99': '#E97132',   # orange
        'SMR': '#A5A5A5'    # gray
    }

    out_imgs = []

    for seabed, dsg in df.groupby('Seabed'):

        dsg = dsg.copy()

        # Apply pass threshold
        dsg = dsg[np.round(dsg['CI_Pass Mean']) >= pass_percent]

        if dsg.empty:
            continue

        spacings = sorted(dsg['Spacing'].unique())

        # Only methods that actually exist for this seabed
        methods = [
            m for m in ['PSD95', 'PSD99', 'SAR', 'SER', 'SMR']
            if m in dsg['Method'].unique()
        ]

        fig, ax = plt.subplots(figsize=(8, 4))

        x = np.arange(len(spacings))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):

            sub = dsg[dsg['Method'] == method]

            means = []
            lower_err = []
            upper_err = []

            for sp in spacings:

                row = sub[sub['Spacing'] == sp]

                if row.empty:
                    means.append(np.nan)
                    lower_err.append(np.nan)
                    upper_err.append(np.nan)
                    continue

                row = row.iloc[0]

                mean = row['CI_Pass Mean']

                means.append(mean)
                lower_err.append(mean - row['CI_Lower'])
                upper_err.append(row['CI_Upper'] - mean)

            xpos = x + (i - (len(methods) - 1) / 2) * width

            ax.bar(
                xpos,
                means,
                width,
                label=f'{method}-99' if method == 'PSD' else method,
                color=colors.get(method, None),
                edgecolor='gray',
            )

            ax.errorbar(
                xpos,
                means,
                yerr=[lower_err, upper_err],
                fmt='none',
                elinewidth=1.5,
                ecolor='black',
                capsize=3,
                zorder=4
            )

        # Axes formatting
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(s)) for s in spacings])

        ax.set_xlabel('Line Spacing (m)', fontweight='bold')
        ax.set_ylabel('Pass Coverage (%)', fontweight='bold')

        ymin = max(0, np.floor(dsg['CI_Lower'].min()) - 1)
        ymax = min(100, np.ceil(dsg['CI_Upper'].max()) + 1)

        ax.set_ylim(ymin, ymax)


        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)

        title = seabed.replace('_', ' ')
        if sampling is not None:
            title += f' ({sampling.capitalize()})'

        # ax.set_title(title)

        legend = ax.legend(loc='best')
        for text in legend.get_texts():
            text.set_fontweight('bold')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()

        if outdir:
            out_path = (
                Path(outdir)
                / f"{seabed}_multi_pass coverages.png"
            )

            fig.savefig(
                out_path,
                dpi=300,
                bbox_inches='tight'
            )

            out_imgs.append(out_path)

        plt.show()

    return out_imgs


# Weighted Composite Score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def composite_score(
    df,
    weights,
    group_cols=['Seabed', 'Method', 'Sampling'],
    path='weight.csv',
    expected_cols=None,
    pass_threshold=95   # <<< ADDED
):
    """
    Compute a composite score for performance metrics and summarize by both method and spacing.
    """

    # --- 1) Validate and normalize metrics ---
    if expected_cols is None:
        expected_cols = [
            'Seabed', 'Method', 'Sampling', 'Spacing',
            'Pass', 'RMSE', 'MAE', 'Bias', 'Correlation',
            'Severity_score', 'Reliability_error', 'Over_penalty',
            'Uncertainty', 'Residuals'
        ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_stats = df.copy()

    # --- 1b) APPLY PASS PERCENT THRESHOLD (HARD FILTER) ---
    df_stats = df_stats[round(df_stats['Pass']) >= pass_threshold].copy()

    if df_stats.empty:
        raise ValueError(
            f"No methods meet the Pass threshold of {pass_threshold:.2f}"
        )

    # --- 2) Automatically detect numeric columns ---
    base_metrics = [
        c for c in expected_cols
        if c not in group_cols + ['Uncertainty', 'Residuals']
    ]

    numeric_cols = [
        c for c in base_metrics
        if pd.api.types.is_numeric_dtype(df_stats[c])
        or np.issubdtype(df_stats[c].dtype, np.number)
    ]

    for c in numeric_cols:
        df_stats[c] = pd.to_numeric(df_stats[c], errors='coerce')

    # --- 3) Normalize ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_stats[numeric_cols])

    df_norm = pd.DataFrame(index=df_stats.index)

    # --- 4) Create standardized columns dynamically ---
    for c in numeric_cols:
        col_s = c.lower() + '_s'
        if c.lower() in [
            'rmse', 'bias', 'std', 'mae', 'over_penalty',
            'sharpness', 'severity_score', 'reliability_error'
        ]:
            df_norm[col_s] = 1 - np.abs(X_scaled[:, numeric_cols.index(c)])
        else:
            df_norm[col_s] = X_scaled[:, numeric_cols.index(c)]

    # --- 5) Keep metadata columns ---
    df_norm[group_cols + ['Spacing']] = df_stats[group_cols + ['Spacing']]

    # Include raw metrics for reference
    for c in numeric_cols:
        df_norm[c] = df_stats[c]

    # Clip standardized values
    for c in df_norm.columns:
        if c.endswith('_s'):
            df_norm[c] = df_norm[c].clip(0, 1)

    # --- 6) Compute composite score ---
    weights_dict = weights
    available_cols = [c for c in weights_dict.keys() if c in df_norm.columns]

    if not available_cols:
        raise ValueError(
            "No matching standardized metric columns found for provided weights."
        )

    df_norm['Composite_Score'] = sum(
        df_norm[c] * weights_dict[c] for c in available_cols
    )

    # --- 7a) Summary by Seabed, Method, Sampling ---
    summary = (
        df_norm.groupby(group_cols, as_index=False)[['Composite_Score']]
        .mean()
        .sort_values('Composite_Score', ascending=False)
    )

    # --- 7b) Summary by Spacing and Method ---
    summary_by_spacing = (
        df_norm.groupby(['Spacing', 'Method'], as_index=False)[['Composite_Score']]
        .mean()
        .sort_values(['Spacing', 'Composite_Score'], ascending=[True, False])
    )

    # --- 7c) Best method per spacing ---
    best_per_spacing = (
        summary_by_spacing.loc[
            summary_by_spacing.groupby('Spacing')['Composite_Score'].idxmax()
        ]
        .reset_index(drop=True)
        .sort_values('Spacing')
    )

    # --- 7d) Best method per Sampling and Spacing ---
    best_per_sampling_spacing = (
        df_norm.groupby(['Sampling', 'Spacing', 'Method'], as_index=False)[['Composite_Score']]
        .mean()
        .loc[
            lambda x: x.groupby(['Sampling', 'Spacing'])['Composite_Score'].idxmax()
        ]
        .reset_index(drop=True)
        .sort_values(['Sampling', 'Spacing'])
    )

    # --- 8) Export ---
    weights_df = pd.DataFrame({
        'Feature': list(weights_dict.keys()),
        'Weight': list(weights_dict.values()),
    })

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df_norm.to_excel(writer, sheet_name='Merged Data', index=False)
        weights_df.to_excel(writer, sheet_name='Weights', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
        summary_by_spacing.to_excel(writer, sheet_name='Summary by Spacing (All)', index=False)
        best_per_spacing.to_excel(writer, sheet_name='Best per Spacing', index=False)
        best_per_sampling_spacing.to_excel(
            writer,
            sheet_name='Best per Sampling & Spacing',
            index=False
        )

    return (
        df_norm,
        weights_dict,
        summary,
        summary_by_spacing,
        best_per_spacing,
        best_per_sampling_spacing
    )

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

def systematic_weight_sweep(
        df: pd.DataFrame,
        feature_cols,
        base_weights,
        sweep_ranges,
        groupby_cols=['Spacing', 'Method', 'Sampling'],
        top_k: int = 3,
        out_prefix: str = "weight_sweep",
        save_csv: bool = True,
        seed: int = 0,
        compare_ml = True,
        pass_threshold = 0.95
):
    np.random.seed(seed)

    # Sanity checks
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Normalize base weights
    w_base = {f: float(base_weights.get(f, 0.0)) for f in feature_cols}
    s = sum(w_base.values())
    for k in w_base:
        w_base[k] /= s

    # Sweep grid
    sweep_keys = list(sweep_ranges.keys())
    sweep_values = [np.linspace(lo, hi, n) for lo, hi, n in sweep_ranges.values()]
    grid = list(itertools.product(*sweep_values))
    print(f"Total sweep points: {len(grid)}")

    def compute_composite(weights_dict, df_subset):
        comps = df_subset.copy()
        vec = np.zeros(len(comps))
        for f, w in weights_dict.items():
            vec += w * comps[f].to_numpy()
        comps["Composite_Score"] = vec
        return comps.groupby(groupby_cols)["Composite_Score"].mean().reset_index()

    all_results = []

    # Perform sweeps
    for idx, combo in enumerate(grid):
        w_candidate = w_base.copy()
        for k, val in zip(sweep_keys, combo):
            w_candidate[k] = float(val)

        summed = sum(w_candidate.values())
        if summed > 0:
            w_candidate = {k: v / summed for k, v in w_candidate.items()}

        method_scores = compute_composite(w_candidate, df)
        method_scores["Sweep_ID"] = idx
        for k, v in w_candidate.items():
            method_scores[f"w_{k}"] = v
        all_results.append(method_scores)

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(grid)} sweep points...")

    results_df = pd.concat(all_results, ignore_index=True)


    # Mean Composite score
    mean_composite_stats = (results_df.groupby(["Sampling", "Spacing", "Method"])["Composite_Score"] .mean().reset_index(name="Mean_Composite"))
    best_by_mean_composite = (mean_composite_stats.sort_values(["Sampling", "Spacing", "Mean_Composite"], ascending=[True, True, False]).groupby(["Sampling", "Spacing"], as_index=False).first())

    overall_composite_stats = (mean_composite_stats.groupby(["Sampling", "Method"])["Mean_Composite"].mean().reset_index(name="Mean_of_Mean_Composite").sort_values(["Sampling", "Mean_of_Mean_Composite"], ascending=[True, False]))

    best_global_composite = (overall_composite_stats.groupby("Sampling", as_index=False).first())

    # Coverage aware composite score
    coverage = (mean_composite_stats.groupby(["Sampling", "Method"])["Spacing"].nunique().reset_index(name="Coverage_Count"))
    total_spacings = mean_composite_stats["Spacing"].nunique()
    coverage["Coverage_Ratio"] = coverage["Coverage_Count"] / total_spacings
    overall = (mean_composite_stats.groupby(["Sampling", "Method"])["Mean_Composite"].mean().reset_index())
    overall = overall.merge(coverage, on=["Sampling", "Method"])
    overall["Coverage_Adjusted_Score"] = (overall["Mean_Composite"] * overall["Coverage_Ratio"])
    overall.sort_values("Coverage_Adjusted_Score", ascending=False)
    best_global_adj_composite = (overall.groupby("Sampling", as_index=False).first())

    # export df
    df_stats = df[["Sampling", "Spacing", "Method", "Pass", "Composite_Score"]]
    # =========================
    # REGRET-BASED ROBUSTNESS
    # =========================

    # compute regret within each sweep
    regret_dfs = []

    for (sampling, spacing, sweep_id), sweep_df in results_df.groupby(
        ["Sampling", "Spacing", "Sweep_ID"]):
        sweep_df = sweep_df.copy()
        best_score = sweep_df["Composite_Score"].max()
        sweep_df["Regret"] = best_score - sweep_df["Composite_Score"]
        regret_dfs.append(sweep_df)
    regret_df = pd.concat(regret_dfs, ignore_index=True)

    regret_stats = (
        regret_df
        .groupby(["Sampling", "Spacing", "Method"])
        .agg(
            Mean_Regret=("Regret", "mean"),
            Std_Regret=("Regret", "std"),
            Max_Regret=("Regret", "max"),
            P90_Regret=("Regret", lambda x: np.percentile(x, 90))
        )
        .reset_index()
    )

    best_by_mean = (
    regret_stats
    .sort_values(["Sampling", "Spacing", "Mean_Regret"])
    .groupby(["Sampling", "Spacing"], as_index=False)
    .first())

    best_by_max = (
    regret_stats
    .sort_values(["Sampling", "Spacing", "Max_Regret"])
    .groupby(["Sampling", "Spacing"], as_index=False)
    .first())

    overall_stats = (
        regret_stats
        .groupby(["Sampling", "Method"])
        .agg(
            Mean_of_Mean=("Mean_Regret", "mean"),
            Mean_of_Std=("Std_Regret", "mean"),
            Max_of_Max=("Max_Regret", "max"),
            P90_of_P90=("P90_Regret", "mean")
        ).sort_values("Mean_of_Mean", ascending=True)
        .reset_index()
    )

    best_global_mean = (
    overall_stats
    .sort_values(["Sampling", "Mean_of_Mean"])
    .groupby("Sampling", as_index=False)
    .first())

    best_global_safe = (
    overall_stats
    .sort_values(["Sampling", "Max_of_Max"])
    .groupby("Sampling", as_index=False)
    .first())

    # coverage-aware regret analysis
    # === COVERAGE PER METHOD ===
    coverage = (
        regret_stats
        .groupby(["Sampling", "Method"])["Spacing"]
        .nunique()
        .reset_index(name="Coverage_Count"))

    total_spacings = regret_stats["Spacing"].nunique()

    coverage["Coverage_Ratio"] = coverage["Coverage_Count"] / total_spacings

    regret_stats = regret_stats.merge(
    coverage[["Sampling", "Method", "Coverage_Ratio"]],
    on=["Sampling", "Method"],
    how="left")

    regret_stats["Adj_Mean_Regret"] = (
    regret_stats["Mean_Regret"] / regret_stats["Coverage_Ratio"])

    regret_stats["Adj_Max_Regret"] = (
    regret_stats["Max_Regret"] / regret_stats["Coverage_Ratio"])

    best_by_mean_adj = (
    regret_stats
    .sort_values(["Sampling", "Spacing", "Adj_Mean_Regret"])
    .groupby(["Sampling", "Spacing"], as_index=False)
    .first())

    best_by_max_adj = (
        regret_stats
        .sort_values(["Sampling", "Spacing", "Adj_Max_Regret"])
        .groupby(["Sampling", "Spacing"], as_index=False)
        .first()
    )

    overall_adj_stats = (regret_stats.groupby(["Sampling", "Method"]).agg(
        Mean_of_AdjMean=("Adj_Mean_Regret", "mean"),
        Max_of_AdjMax=("Adj_Max_Regret", "max")).reset_index())

    best_global_adj_mean = (overall_adj_stats.sort_values(["Sampling", "Mean_of_AdjMean"]).groupby("Sampling", as_index=False).first())

    best_global_adj_safe = (overall_adj_stats.sort_values(["Sampling", "Max_of_AdjMax"]).groupby("Sampling", as_index=False).first())

    # === Rank and count ===
    if groupby_cols == ['Spacing', 'Method', 'Sampling']:

        rankings = []
        for (sampling, spacing), subdf in results_df.groupby(["Sampling", "Spacing"]):
            for sweep_id, sweep_df in subdf.groupby("Sweep_ID"):
                ranked = sweep_df.sort_values("Composite_Score", ascending=False)
                ranked["Rank"] = range(1, len(ranked) + 1)
                ranked["Sampling"] = sampling
                ranked["Spacing"] = spacing
                rankings.append(ranked)

        ranked_df = pd.concat(rankings, ignore_index=True)

        counts = (
            ranked_df[ranked_df["Rank"] <= top_k]
            .groupby(["Sampling", "Spacing", "Method", "Rank"])
            .size()
            .unstack("Rank", fill_value=0)
        )

        for r in range(1, top_k + 1):
            if r not in counts.columns:
                counts[r] = 0

        counts = (
            counts
            .rename(columns={1: "Top1_Count", 2: "Top2_Count", 3: "Top3_Count"})
            .reset_index()
        )

    else:
        raise ValueError("Only supports ['Spacing','Method','Sampling'].")

    # === Overall best per Sampling × Spacing ===
    overall_best_per_sampling_spacing = (
        counts
        .sort_values(["Sampling", "Spacing", "Top1_Count"], ascending=[True, True, False])
        .groupby(["Sampling", "Spacing"], as_index=False)
        .first())

    # === Overall best method & sampling per spacing (based on sweep dominance) ===
    overall_best_per_spacing_dominance = (
        counts
        .sort_values(["Spacing", "Top1_Count"], ascending=[True, False])
        .groupby("Spacing", as_index=False)
        .first()
    )

    # === Top 3 Overall Methods Across Spacings (Top1 Dominance Only) ===
    n_spacings = counts["Spacing"].nunique()

    overall_top3_across_spacings = (
        counts
        .groupby(["Sampling", "Method"], as_index=False)["Top1_Count"]
        .sum()
    )

    # Rank methods by dominance within each Sampling
    overall_top3_across_spacings = (
        overall_top3_across_spacings
        .sort_values(["Sampling", "Top1_Count"], ascending=[True, False])
    )

    # Extract top 3 per Sampling
    overall_top3_across_spacings = (
        overall_top3_across_spacings
        .groupby("Sampling")
        .head(3)
        .reset_index(drop=True)
    )

    # Percent share of total Top1 dominance within each Sampling
    overall_top3_across_spacings["Top1_Percent"] = (
        overall_top3_across_spacings
        .groupby("Sampling")["Top1_Count"]
        .transform(lambda x: x / x.sum() * 100)
    )


    if compare_ml:
        # === Identify overall best method across spacings (from Top1 dominance) ===
        overall_best_method = (
            overall_top3_across_spacings
            .groupby("Sampling")
            .head(1)
            .reset_index(drop=True)
        )

        # Ensure it's a classical method
        overall_best_classical = []
        for _, row in overall_best_method.iterrows():
            if "ML" not in row["Method"]:
                overall_best_classical.append(row)
            else:
                next_best = overall_top3_across_spacings[
                    (overall_top3_across_spacings["Sampling"] == row["Sampling"]) &
                    (~overall_top3_across_spacings["Method"].str.contains("ML"))
                ].head(1)
                if not next_best.empty:
                    overall_best_classical.append(next_best.iloc[0])
        best_classical_df = pd.DataFrame(overall_best_classical)

        mri_list = []

        for (sampling, spacing, sweep_id), subdf in results_df.groupby(["Sampling", "Spacing", "Sweep_ID"]):
            ml_methods = [m for m in subdf["Method"].unique() if "ML" in m]
            if not ml_methods:
                continue

            # Get the best classical method for this Sampling
            best_row = best_classical_df[best_classical_df["Sampling"] == sampling]
            if best_row.empty:
                # No classical method available for this sampling, skip
                continue

            best_classical_method = best_row["Method"].values[0]
            classical_score_series = subdf[subdf["Method"] == best_classical_method]["Composite_Score"]
            if classical_score_series.empty:
                # Method not present in this sweep, skip
                continue
            best_classical_score = classical_score_series.values[0]

            for ml_m in ml_methods:
                ml_score_series = subdf[subdf["Method"] == ml_m]["Composite_Score"]
                if ml_score_series.empty:
                    continue
                ml_score = ml_score_series.values[0]
                mri = (ml_score - best_classical_score) / best_classical_score * 100
                mri_list.append({
                    "Sampling": sampling,
                    "Spacing": spacing,
                    "Sweep_ID": sweep_id,
                    "ML_Method": ml_m,
                    "Best_Classical_Method": best_classical_method,
                    "Best_Classical_Score": best_classical_score,
                    "ML_Score": ml_score,
                    "MRI_percent": mri
                })

        mri_df = pd.DataFrame(mri_list)
        mean_mri_percent = mri_df["MRI_percent"].mean() if not mri_df.empty else np.nan
        print(f"Mean Relative Improvement of ML over best classical method: {mean_mri_percent:.2f}%")


    # === Save ===
    if save_csv:
        excel_path = f"{out_prefix}"
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_stats.to_excel(writer, sheet_name="Statistics", index=False)
            results_df.to_excel(writer, sheet_name="Results", index=False)
            counts.to_excel(writer, sheet_name="Top_Methods_Counts", index=False)
            overall_best_per_sampling_spacing.to_excel(writer, sheet_name="Best per Sampling & Spacing", index=False)
            overall_top3_across_spacings.to_excel(writer, sheet_name="Top3 Across Spacings",index=False )

            # BEST METHODS PER SPACING
            best_by_mean_composite.to_excel(writer, sheet_name="Best_by_Mean_Composite", index=False)
            best_by_mean.to_excel(writer, sheet_name="Best_by_Mean_Regret", index=False)
            best_by_max.to_excel(writer, sheet_name="Best_by_Max_Regret", index=False)
            best_by_mean_adj.to_excel(writer, sheet_name="Best_by_AdjMean_Regret", index=False)
            best_by_max_adj.to_excel(writer, sheet_name="Best_by_AdjMax_Regret", index=False)

            # GLOBAL (ACROSS SPACINGS)
            best_global_composite.to_excel(writer, sheet_name="Best_Global_Composite", index=False)
            best_global_adj_composite.to_excel(writer, sheet_name="Best_Global_AdjMean_Composite", index=False)
            best_global_mean.to_excel(writer, sheet_name="Best_Global_Mean_Regret", index=False)
            best_global_safe.to_excel(writer, sheet_name="Best_Global_Safe_Regret", index=False)
            best_global_adj_mean.to_excel(writer, sheet_name="Best_Global_AdjMean_Regret", index=False)
            best_global_adj_safe.to_excel(writer, sheet_name="Best_Global_AdjSafe_Regret", index=False)

            # Stats
            mean_composite_stats.to_excel(writer, sheet_name="Mean_Composite_Stats", index=False)
            overall_composite_stats.to_excel(writer, sheet_name="Global_Composite_Stats", index=False)
            overall.to_excel(writer, sheet_name="Coverage_Global_Composite_Stats", index=False)
            regret_df.to_excel(writer, sheet_name="Regret_All_Sweeps", index=False)
            overall_stats.to_excel(writer, sheet_name="Global_Regret_Stats", index=False)
            coverage.to_excel(writer, sheet_name="Coverage", index=False)
            regret_stats.to_excel(writer, sheet_name="Regret_with_Coverage_Stats", index=False)
            overall_adj_stats.to_excel(writer, sheet_name="Global_Adj_Regret_Stats", index=False)

            if compare_ml:
                mri_df.to_excel(writer, sheet_name="MRI_Percent", index=False)
                # Optional: also save the mean MRI as a separate sheet
                pd.DataFrame({"Mean_MRI_percent": [mean_mri_percent]}).to_excel( writer, sheet_name="MRI_Summary", index=False)

            # overall_best_per_spacing_agg.to_excel(writer, sheet_name="Best per Spacing Agg.", index=False )
            # diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)

        print(f"Saved Excel file: {excel_path}")

    # return results_df, counts, diagnostics_df
    return results_df, counts