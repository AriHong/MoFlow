import numpy as np
from scipy.ndimage import uniform_filter1d
from fastdtw import fastdtw
import matplotlib.pyplot as plt


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
def smooth(x, size=3):# smoothing
    return uniform_filter1d(x, size=size)
def pad_ends(x):# Add zero to both ends
    return np.concatenate([[0], x, [0]])

def bin_and_average(x, time, n_bins=20):
    bins = np.linspace(np.min(time), np.max(time), n_bins + 1)
    binned = np.digitize(time, bins) - 1
    out = np.array([np.mean(x[binned == i]) for i in range(n_bins)])
    return out

def get_dtw(adata, gene, genetime=False, 
            timekey = 'velo_s_pseudotime', n_bins=20):
    """DTW analysis
        
    Parameters
    ---------
    adata: `anndata.AnnData`
        anndata that contains the chromatin accessibility, unspliced abundance, spliced abundance, embedding space, and pseudotime information. 
    gene_time: optional, `bool` (default: False)
        `True` if use gene specific time. `False` if use global time.
    time_key: optional, `str` (default: velo_s_pseudotime)
        obs or layer key to annotate pseudotime.
    n_bins: optional, `int` (default: 20)
        the number of bin to discritize time
    Returns
    -------
    time_pad: `numpy.ndarray`
        A linearly spaced array representing the padded time points after dynamic time warping (DTW).
    c_pad: `numpy.ndarray`
        Padded and smoothed chromatin accessibility values for the gene over time.
    u_pad: `numpy.ndarray`
        Padded and smoothed unspliced abundance values for the gene over time.
    s_pad: `numpy.ndarray`
        Padded and smoothed spliced abundance values for the gene over time.
    path_c_s: `list of tuples`
        The optimal alignment path (DTW path) between chromatin accessibility (c_pad) and spliced abundance (s_pad).
    path_u_s: `list of tuples`
        The optimal alignment path (DTW path) between unspliced abundance (u_pad) and spliced abundance (s_pad).
    time_lag_c_s: `list of float`
        The time lags between the aligned chromatin accessibility and spliced abundance based on the DTW path.
    time_lag_u_s: `list of float`
        The time lags between the aligned unspliced abundance and spliced abundance based on the DTW path.
        """
    
    if genetime:
        time = adata[:, gene].layers[timekey].ravel()
    else:
        time = adata.obs[timekey].values
    time = np.clip(time, 0, np.quantile(time, 0.95))

    c = adata[:, gene].layers['Mc'].ravel()
    u = adata[:, gene].layers['Mu'].ravel()
    s = adata[:, gene].layers['Ms'].ravel()
    
    c = bin_and_average(c, time, n_bins)
    u = bin_and_average(u, time, n_bins)
    s = bin_and_average(s, time, n_bins)


    c_norm = normalize(c)
    u_norm = normalize(u)
    s_norm = normalize(s)

    c_smooth = smooth(c_norm)
    u_smooth = smooth(u_norm)
    s_smooth = smooth(s_norm)

    c_pad = pad_ends(c_smooth)
    u_pad = pad_ends(u_smooth)
    s_pad = pad_ends(s_smooth)
    
    time_pad = np.linspace(0, 1, len(c_pad))


    distance_c_s, path_c_s = fastdtw(c_pad, s_pad, )
    distance_u_s, path_u_s = fastdtw(u_pad, s_pad, )

    time_lag_c_s = [time_pad[j] - time_pad[i] for i, j in path_c_s]
    time_lag_u_s = [time_pad[j] - time_pad[i] for i, j in path_u_s]
    return(time_pad, c_pad, u_pad, s_pad, path_c_s, path_u_s, time_lag_c_s, time_lag_u_s)


def plot_dtw(adata, gene, genetime=False, timekey = 'velo_s_pseudotime', n_bins=20,
         figsave=None):
    ):
    """
    Plot the DTW alignment of chromatin accessibility, unspliced, and spliced abundance profiles over time 
    along with the corresponding time lags.

    Parameters
    ----------
    adata : `anndata.AnnData`
        Annotated data object containing the chromatin accessibility, unspliced and spliced abundance,
        embedding space, and pseudotime information.
    
    gene : `str`
        The gene name to plot the DTW alignment for.
    
    genetime : `bool`, optional (default: False)
        If `True`, use gene-specific time values. If `False`, use the global pseudotime.
    
    timekey : `str`, optional (default: 'velo_s_pseudotime')
        The key in `adata.obs` or `adata.layers` that stores the pseudotime information.
    
    n_bins : `int`, optional (default: 20)
        The number of bins to discretize the time values for averaging.
    
    figsave : `str` or `None`, optional (default: None)
        If provided, the figure will be saved to the specified file path. If `None`, the figure will not be saved.

    Returns
    -------
    None
    """
    time_pad, c_pad, u_pad, s_pad, path_c_s, path_u_s, time_lag_c_s, time_lag_u_s = get_dtw(adata, gene, genetime, timekey, n_bins)

    fig, axs = plt.subplots(2, 1, figsize=(4.8, 3.2), sharex=True)

    # Top: c vs s
    axs[0].plot(time_pad, c_pad, label='c', color='#102E50')
    axs[0].plot(time_pad, u_pad, label='u', color='#F5C45E')
    axs[0].plot(time_pad, s_pad, label='s', color='#BE3D2A')
    
    for i, j in path_c_s[::3]:  # skip some lines for clarity
        axs[0].plot([time_pad[i], time_pad[j]], [c_pad[i], s_pad[j]], 'k--', linewidth=0.5)
    for i, j in path_u_s[::3]:
        axs[0].plot([time_pad[i], time_pad[j]], [u_pad[i], s_pad[j]], 'k--', linewidth=0.5)
    
    axs[0].set_ylabel("Norm values")
    axs[0].legend()
    axs[0].set_title("DTW Alignment")


    axs[1].plot([time_pad[i] for i, _ in path_c_s], time_lag_c_s, label='c-s lag', color='#3E3F5B')
    axs[1].plot([time_pad[i] for i, _ in path_u_s], time_lag_u_s, label='u-s lag', color='#8AB2A6')
    axs[1].axhline(0, color='gray', linestyle='--')
    axs[1].set_ylabel("Time lag")
    axs[1].set_xlabel("Time")
    axs[1].legend()

    
    fig.suptitle(gene)

    plt.tight_layout()
    plt.show()
    
    if figsave is not None:
        fig.savefig(figsave)
