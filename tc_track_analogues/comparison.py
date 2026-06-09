from scipy.stats import ttest_ind, cramervonmises_2samp
from pycircstat2.hypothesis import angular_randomisation_test
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .utils import flag_periods, get_month_doy_edges_labels


def get_counts_per_year(analogues, dates):
    cf_min, cf_max, f_min, f_max = dates
    year = analogues.time.dt.year
    counts = analogues.groupby(year).size()
    counts = counts.reindex(range(cf_min, f_max + 1), fill_value=0)
    counts = counts.rename("counts").reset_index()
    counts["period"] = flag_periods(counts.time, dates)
    return counts


# ================  Statistical tests  ================ #


def get_ttest_pval(dist, dist_period):
    dist_CF = dist[dist_period == "CF"]
    dist_F = dist[dist_period == "F"]
    return ttest_ind(dist_CF, dist_F).pvalue

def get_ttest_ci(dist, dist_period, level = 0.95):
    dist_CF = dist[dist_period == "CF"]
    dist_F = dist[dist_period == "F"]
    delta = dist_F.mean() - dist_CF.mean()
    CI = ttest_ind(dist_F, dist_CF).confidence_interval(level)
    return delta, CI

def get_cvm_pval(dist, dist_period):
    dist_CF = dist[dist_period == "CF"]
    dist_F = dist[dist_period == "F"]
    return cramervonmises_2samp(dist_CF, dist_F).pvalue


def get_circular_pval(dist, dist_period):
    dist_CF = dist[dist_period == "CF"]
    dist_F = dist[dist_period == "F"]
    return angular_randomisation_test([dist_CF, dist_F]).pval


def print_pval(ax, pval):
    if pval < 0.05:
        ax.text(
            0.03,
            0.97,
            f"p-value={pval:.2e}*",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                linewidth=0.8,
            ),
        )
    else:
        ax.text(
            0.03,
            0.97,
            f"p-value={pval:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                linewidth=0.8,
            ),
        )


# ================  Plots  ================ #
def plot_diff_in_freq(analogues, parameters, palette):
    # Compute the number of analogues per year
    counts = {
        c: get_counts_per_year(analogues[c], parameters.loc[c, "dates"])
        for c in analogues
    }

    # Plot the distributions and compute pvals
    fig, axs = plt.subplots(
        1,
        len(counts),
        figsize=[2.5 * len(counts), 2.5],
    )
    pvals = {}
    for i, c in enumerate(counts):
        # Bar plot
        sns.barplot(
            data=counts[c],
            y="counts",
            hue="period",
            x="period",
            palette=palette,
            ax=axs[i],
            errorbar="se",
            legend=False,
        )
        # Compute pvals and display
        pvals[c] = get_ttest_pval(counts[c].counts, counts[c].period)
        print_pval(axs[i], pvals[c])

        # Layout
        axs[i].set_title(c)
        axs[i].set_ylabel("")
        axs[i].set_xlabel("")
        axs[i].grid(axis="y")
        axs[i].set_axisbelow(True)
    axs[0].set_ylabel("#analogues / year")
    plt.tight_layout()

    return pvals


def plot_diff_in_intensity(analogues, palette, intensity_var="wind", type="ecdf"):
    # Remove catalogues that do not have the specific intensity variable
    analogues_with_intensity_var = {}
    for c in analogues:
        if intensity_var in analogues[c].columns:
            analogues_with_intensity_var[c] = analogues[c]
    analogues = analogues_with_intensity_var

    # Plot the distributions and compute pvals
    fig, axs = plt.subplots(
        1, len(analogues), figsize=[2.5 * len(analogues), 3], sharey=True
    )
    pvals = {}
    for i, c in enumerate(analogues):
        # Filter the analogues
        A = analogues[c]
        A = A[(A.period != "nan") & (~A[intensity_var].isna())]

        # Plot the distribution depending on requested type
        if type == "violin":
            sns.violinplot(
                data=A,
                y=intensity_var,
                hue="period",
                split=True,
                palette=palette,
                ax=axs[i],
                legend=False,
            )
        if type == "ecdf":
            sns.ecdfplot(
                data=A,
                x=intensity_var,
                hue="period",
                palette=palette,
                ax=axs[i],
                legend=False,
            )

        # Compute pval and print on the plot
        pvals[c] = get_cvm_pval(A[intensity_var], A.period)
        print_pval(axs[i], pvals[c])

        # Layout
        axs[i].set_ylabel("")
        axs[i].set_title(c)

    axs[0].set_ylabel("Wind intensity (m/s)")
    plt.tight_layout()

    return pvals

def plot_summary_diff_wind(analogues):
    plt.figure(figsize = (3,3))
    for i, c in enumerate(analogues):
        delta, CI = get_ttest_ci(analogues[c].wind, analogues[c].period)
        plt.scatter([i], [delta], color = 'k')
        plt.plot([i,i], [CI[0], CI[1]], color = 'k')
    plt.xticks(range(len(analogues)), analogues.keys())
    plt.axhline(y=0, color = 'k', linestyle = "--")
    plt.ylabel("$\Delta u$ in m/s")
    plt.grid(axis = 'y')
    sns.despine()

def plot_diff_in_seasonality(analogues, palette):
    fig, axs = plt.subplots(
        1,
        len(analogues),
        figsize=[2.5 * len(analogues), 2.5],
        subplot_kw=dict(projection="polar"),
    )
    pvals = {}
    for i, c in enumerate(analogues):
        # Compute doy and corresponding angle
        A = analogues[c].assign(doy=analogues[c].time.dt.dayofyear)
        A = A.assign(doy_angle=A.doy * 2 * np.pi / 365)
        A = A[A.period != "nan"]

        # Get edges and their labels for monthly doy
        bin_edges_doy, bin_edges_angle, tick_positions, month_labels = (
            get_month_doy_edges_labels()
        )

        # Plot the distribution
        sns.histplot(
            data=A,
            x="doy_angle",
            hue="period",
            ax=axs[i],
            bins=bin_edges_angle,
            legend=False,
            palette=palette,
        )

        # Compute p-values and print on the plot
        pvals[c] = get_circular_pval(A.doy_angle, A.period)
        print_pval(axs[i], pvals[c])

        # Layout
        axs[i].set_xticks(tick_positions)
        axs[i].set_xticklabels(month_labels, fontsize=7)
        axs[i].set_yticks([])
        axs[i].grid(axis="y")
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")

    plt.tight_layout()

    return pvals
