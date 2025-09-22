import numpy as np
import os
import pickle
import stdpopsim
import msprime
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# parameter grid
grid_size = 35
fold_change = np.linspace(-1, 1, grid_size + 1)
base_x = 1
base_y = 1
x_grid = 2 ** fold_change * base_x
y_grid = 2 ** fold_change * base_y


# cached experiment
output_path = "/sietch_colab/data_share/cxt/experiments/bias-coalscale-mutrate/"
cache_file = f"{output_path}/cache.pkl"
cache = pickle.load(open(cache_file, "rb"))
statistics = cache["statistics"]
bias_corr = cache["bias_corr"]
rmse_corr = cache["rmse_corr"] # actually mse
bias = cache["bias"]
rmse = cache["rmse"] # actually mse

## DEBUG
#ycorr_store = cache["ycorr_store"]
#ytrue_store = cache["ytrue_store"]
#fig, axs = plt.subplots(1, 4, figsize=(8, 2), sharex=True, sharey=True, constrained_layout=True)
#for i, ax in enumerate(axs):
#    j = -(i + 1)
#    bad = np.argsort(rmse_corr)[j]
#    foo = [np.mean((x[bad] - y[bad]) ** 2) for x, y in zip(ycorr_store, ytrue_store)]
#    bad2 = np.argmax(foo)
#    ax.step(np.arange(500), ytrue_store[bad2][bad], where="post", color="black", label="true")
#    ax.step(np.arange(500), ycorr_store[bad2][bad], where="post", color="red", label="cxt")
#    ax.set_title(f"MSE: {foo[bad2]:.2f}")
#    ax.legend()
#fig.supxlabel("Window")
#fig.supylabel("log TMRCA")
#plt.savefig(f"/home/natep/public_html/cxt/debug.png")
#assert False


rows = 2
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3), constrained_layout=True, sharex=True, sharey=True)

# bias
vbound = max(np.abs(bias).max(), np.abs(bias_corr).max())
axs[0, 0].set_title("Uncalibrated")
img = axs[0, 0].pcolormesh(
    x_grid, y_grid,
    bias.reshape(grid_size, grid_size).T,
    cmap=plt.get_cmap("seismic"),
    norm=Normalize(vmin=-vbound, vmax=vbound),
)
axs[0, 0].plot(base_x, base_y, "o", markersize=4, c="green")
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log', base=2)

axs[0, 1].set_title("Calibrated")
img = axs[0, 1].pcolormesh(
    x_grid, y_grid,
    bias_corr.reshape(grid_size, grid_size).T,
    cmap=plt.get_cmap("seismic"),
    norm=Normalize(vmin=-vbound, vmax=vbound),
)
axs[0, 1].plot(base_x, base_y, "o", markersize=4, c="green")
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[0, 1], label="Bias (log TMRCA)")


# RMSE
vmax = max(rmse.max(), rmse_corr.max())
vmin = max(rmse.min(), rmse_corr.min())
img = axs[1, 0].pcolormesh(
    x_grid, y_grid,
    rmse.reshape(grid_size, grid_size).T,
    cmap=plt.get_cmap("magma"),
    norm=Normalize(vmin=vmin, vmax=vmax),
)
axs[1, 0].plot(base_x, base_y, "o", markersize=4, c="green")
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].set_yscale('log', base=2)

img = axs[1, 1].pcolormesh(
    x_grid, y_grid,
    rmse_corr.reshape(grid_size, grid_size).T,
    cmap=plt.get_cmap("magma"),
    norm=Normalize(vmin=vmin, vmax=vmax),
)
axs[1, 1].plot(base_x, base_y, "o", markersize=4, c="green")
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[1, 1], label="MSE (log TMRCA)")

fig.supxlabel("Coalescent unit scaling")
fig.supylabel("Mutation rate scaling")
#plt.savefig(f"/home/natep/public_html/cxt/supp-fig-bias-mutcoal.png")
plt.savefig(f"{output_path}/supp-fig-bias-mutcoal.png")
plt.clf()


# summary stats
rows = 1
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*3.5), constrained_layout=True, sharex=True, sharey=True, squeeze=False)
axs[0, 0].set_title("Mutations")
img = axs[0, 0].pcolormesh(
    x_grid, y_grid,
    statistics[:, 0].reshape(grid_size, grid_size).T / 1e6,
    cmap=plt.get_cmap("inferno"),
)
axs[0, 0].plot(base_x, base_y, "o", markersize=4, c="green")
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[0, 0], label="# events per bp")

axs[0, 1].set_title("Recombinations")
img = axs[0, 1].pcolormesh(
    x_grid, y_grid,
    statistics[:, 1].reshape(grid_size, grid_size).T / 1e6,
    cmap=plt.get_cmap("inferno"),
)
axs[0, 1].plot(base_x, base_y, "o", markersize=4, c="green")
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_yscale('log', base=2)
plt.colorbar(img, ax=axs[0, 1], label="# events per bp")

fig.supxlabel("Coalescent unit scaling")
fig.supylabel("Mutation rate scaling")
#plt.savefig(f"/home/natep/public_html/cxt/supp-fig-stats-mutcoal.png")
plt.savefig(f"{output_path}/supp-fig-stats-mutcoal.png")
plt.clf()

