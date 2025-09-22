import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

output_path = "/sietch_colab/data_share/cxt/experiments"
windows = np.linspace(0, 1e6, 501)

results = {}
for what in ["constant", "zigzag", "rice"]:
    results[what] = {}
    for m in [0.5, 1, 2]:
        cache_file = f"{output_path}/coverage-{what}/cache.{str(m)}.pkl"
        stats_file = f"{output_path}/coverage-{what}/stats.{str(m)}.pkl"
        cache = pickle.load(open(cache_file, "rb"))
        stats = pickle.load(open(stats_file, "rb"))
        results[what][m] = (cache, stats)

# plot
cmap = plt.get_cmap("tab10")
midpoint = windows[1:] / 2 + windows[:-1] / 2
alpha_grid = [0.05, 0.25, 0.5, 0.75, 0.95]
colors = cmap(np.linspace(0, 1, len(alpha_grid)))

rows = 2
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3), constrained_layout=True, sharex="col", sharey="row")

# plot coverage vs position at basic mu
for ii, what in enumerate(["constant", "zigzag", "rice"]):
    preds, stats = results[what][1]

    yp = preds["ycorr_store"]
    yt = preds["ytrue_store"]
    od = stats["obs_div"]
    ed = stats["exp_div"]

    yt = np.log(ed) # actual TMRCA, not discretized
    
    for alpha, col in zip(alpha_grid, colors):
        lo = np.quantile(yp, alpha / 2, axis=1)
        hi = np.quantile(yp, 1 - alpha / 2, axis=1)
        pv = np.logical_and(yt >= lo, yt <= hi)
        cov = pv.mean(axis=0)
        # coverage vs position on sequence
        axs[0, ii].plot(midpoint / 1e6, cov, alpha=0.5, color=col)
        axs[0, ii].axhline(y=1 - alpha, linestyle="dashed", color=col)
        if alpha <= 0.5:
            ymin = cov[200:300].min()
            axs[0, ii].text(0.5, ymin, f"$\\alpha={1 - alpha:.2f}$", color=col, ha="center", va="top", size=10)
        else:
            ymax = cov[200:300].max()
            axs[0, ii].text(0.5, ymax, f"$\\alpha={1 - alpha:.2f}$", color=col, ha="center", va="bottom", size=10)

    axs[0, ii].set_ylim(0, 1)
    if ii == 0:
        axs[0, ii].set_ylabel("Proportion true TMRCA\nin posterior $\\alpha$-intervals")
    if what == "constant":
        axs[0, ii].set_title("HomSap\nPiecewiseConstantSize", size=10)
    elif what == "zigzag":
        axs[0, ii].set_title("HomSap\nZigzag_1S14", size=10)
    elif what == "rice":
        axs[0, ii].set_title("OrySat\nBottleneckMigration_3C07", size=10)


# plot posterior variance (e.g. interval width) vs position at basic mu
colors = ["red", "blue", "green"]
for ii, what in enumerate(["constant", "zigzag", "rice"]):
    for jj, m in enumerate([0.5, 1, 2]):
        preds, stats = results[what][m]
        yp = preds["ycorr_store"]
        ed = stats["exp_div"]
        od = stats["obs_div"]
        yt = np.log(ed) # actual TMRCA, not discretized
        var = yp.var(axis=1).mean(axis=0)
        ypos = var[230:270].max()
        axs[1, ii].plot(midpoint / 1e6, var, color=colors[jj], label=f"m={m}")
        if m == 0.5:
            lab = f"$\\mu / 2$"
        elif m == 1:
            lab = f"$\\mu$"
        else:
            lab = f"$2 \\mu$"
        axs[1, ii].text(0.5, ypos, lab, color=colors[jj], va="bottom", ha="center")
    if ii == 0:
        axs[1, ii].set_ylabel("Average posterior variance")
    #prior_var = np.sqrt(yt.var())
    #axs[1, ii].axhline(y=prior_var, color="black", linestyle="dashed")

fig.supxlabel("Position in prediction window (Mb)", size=10)
plt.savefig(f"{output_path}/supp-fig-posterior-calibration.png")
    
