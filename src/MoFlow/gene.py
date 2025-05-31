import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import seaborn as sns




def sampling_neighbors(plotdata,step=(30,30),percentile=25):

    from scipy.stats import norm
    def gaussian_kernel(X, mu = 0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    embed_len = plotdata.shape[1]
    
    if len(step) != embed_len:
        step = np.repeat(step[0], embed_len)
    grs = []
    for dim_i in range(plotdata.shape[1]):
        m, M = np.min(plotdata[:, dim_i]), np.max(plotdata[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, step[dim_i])
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
   
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(10) # set random seed
    
    nn = NearestNeighbors()

    neighbors_1 = min((plotdata.shape[0]-1), 20)
    nn.fit(plotdata)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, neighbors_1)

    ix_choice = ixs[:,0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()

    neighbors_2 = min((plotdata.shape[0]-1), 20)
    nn.fit(plotdata)
    dist, ixs = nn.kneighbors(plotdata[ix_choice, :embed_len], neighbors_2)
    
    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, percentile)
    ix_choice = ix_choice[bool_density]
    return(ix_choice)

    
def scatter_gene(adata,
                 gene_list,
                 by='us',
                 color_by='celltype',
                 colors=None,
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 velocity_arrows=False,
                 downsample=1,
                 percentile=0,
                 figsize=None,
                 pointsize=2,
                 alpha=.5,
                 cmap='coolwarm',
                 view_3d_elev=None,
                 view_3d_azim=None,
                 full_name=False,
                 log_scale=False,
                 **kwargs

                 ):
    """Gene scatter plot.

    This function plots phase portraits of the specified plane.

    Parameters
    ----------
    data: :class:`~anndata.AnnData` 
        Data result from relay velocity model.
    gene_list: `str`,  list of `str`
        List of gene_list to plot.
    by: `str` (default: `us`)
        Plot unspliced-spliced plane if `us`. Plot chromatin-unspliced plane
        if `cu`.
        Plot 3D phase portraits if `cus`.
    color_by: `str` (default: `celltype`)
        Color by the four potential states if `celltype`. Other common values are
        leiden, louvain, cluster, etc.
        If not `celltype`, the color field must be present in `.uns`, which can be
        pre-computed with `scanpy.pl.scatter`.
        Color by can also be common values like `c` in `.layers`, which displays the log
        accessibility on U-S phase portraits.
    n_cols: `int` (default: 5)
        Number of columns to plot on each row.
    axis_on: `bool` (default: `True`)
        Whether to show axis labels.
    frame_on: `bool` (default: `True`)
        Whether to show plot frames.
    velocity_arrows: `bool` (default: `False`)
        Whether to show velocity arrows of cells on the phase portraits.
    downsample: `int` (default: 1)
        How much to downsample the cells. The remaining number will be
        `1/downsample` of original.
    figsize: `tuple` (default: `None`)
        Total figure size.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    linewidth: `float` (default: 2)
        Line width for connected anchors.
    cmap: `str` (default: `coolwarm`)
        Color map for log accessibilities or other continuous color keys when
        plotting on U-S plane.
    view_3d_elev: `float` (default: `None`)
        Matplotlib 3D plot `elev` argument. `elev=90` is the same as U-S plane,
        and `elev=0` is the same as C-U plane.
    view_3d_azim: `float` (default: `None`)
        Matplotlib 3D plot `azim` argument. `azim=270` is the same as U-S
        plane, and `azim=0` is the same as C-U plane.
    full_name: `bool` (default: `False`)
        Show full names for chromatin, unspliced, and spliced rather than
        using abbreviated terms c, u, and s.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype


    if by not in ['us', 'cu', 'cus']:
        raise ValueError("'by' argument must be one of ['us', 'cu', 'cus']")

    #set colorcode
    if color_by in adata.layers and is_numeric_dtype(adata.layers[color_by]):
        types = None
        colors = None
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]):
        if colors is not None:
            types = adata.obs[color_by].cat.categories
            if set(colors.keys()) & set(types) != set(types):
                raise ValueError("Currently, colors keys don't contain all categories of types.")

        elif color_by+'_colors' in adata.uns.keys():
            types = adata.obs[color_by].cat.categories
            colors = adata.uns[f'{color_by}_colors']
        else:
            raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')
    elif color_by == 'state' and 'fit_state' not in adata.layers.keys():
        raise ValueError('fit_state is not found. Please run '
                         'recover_dynamics_chrom function first or provide a '
                         'valid color key.')    
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')
     
    
    downsample = np.clip(int(downsample), 1, 10)
    gene_list = np.array(gene_list)
    missing_genes = gene_list[~np.isin(gene_list, adata.var_names)]
    
    #if len(missing_genes) > 0: 
    #    logg.update(f'{missing_genes} not found', v=0)
    gene_list = gene_list[np.isin(gene_list, adata.var_names)]
    gn = len(gene_list)

    #Set figure and axes
    if gn == 0:
        return
    if gn < n_cols:
        n_cols = gn
    if by == 'cus':
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(4*n_cols, 2.7*(-(-gn // n_cols)))
                                if figsize is None else figsize,
                                subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(3.2*n_cols, 2.7*(-(-gn // n_cols)))
                                if figsize is None else figsize)
    fig.patch.set_facecolor('white')
                     
    
    for g, gene in enumerate(gene_list):
        c = adata[:, gene].layers['Mc'].copy()
        u = adata[:, gene].layers['Mu'].copy()
        s = adata[:, gene].layers['Ms'].copy()

        c = c.A if sparse.issparse(c) else c
        u = u.A if sparse.issparse(u) else u
        s = s.A if sparse.issparse(s) else s
        c, u, s = np.ravel(c), np.ravel(u), np.ravel(s)

        if percentile>0:
            if by=='us':
                ix_choice = sampling_neighbors(np.array([u, s]).T, percentile=percentile)
            elif by=='cu':
                ix_choice = sampling_neighbors(np.array([c, u]).T, percentile=percentile)
            else:
                ix_choice = sampling_neighbors(np.array([c, u, s]).T, percentile=percentile)
        else:
            ix_choice = np.arange(c.shape[0])
                
        
        
        if velocity_arrows:
            if 'velo_u' in adata.layers.keys():
                vu = adata[:, gene].layers['velo_u'].copy()
            else:
                vu = np.zeros(adata.n_obs)
            max_u = np.max([np.max(u), 1e-6])
            u /= max_u
            vu = np.ravel(vu)
            vu /= np.max([np.max(np.abs(vu),), 1e-6])
            
            if 'velo_s' in adata.layers.keys():
                vs = adata[:, gene].layers['velo_s'].copy()
            else:
                vs = adata[:, gene].layers['velocity'].copy()
            max_s = np.max([np.max(s), 1e-6])
            s /= max_s
            vs = np.ravel(vs)
            vs /= np.max([np.max(np.abs(vs)), 1e-6])
            if 'velo_c' in adata.layers.keys():
                vc = adata[:, gene].layers['velo_c'].copy()
                max_c = np.max([np.max(c), 1e-6])
                c /= max_c
                vc = np.ravel(vc)
                vc /= np.max([np.max(np.abs(vc)), 1e-6])
            
            
        row = g // n_cols
        col = g % n_cols
        ax = axs[row, col]
        
        
        
        if types is not None:
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                filt = np.arange(adata.n_obs)[np.ravel(filt)]

                if type(colors) == dict:
                    color = colors[types[i]]
                else:
                    color = colors[i]
                
                if by == 'us':
                    
                    if velocity_arrows:
                        
                        filt = list(set(filt) & set(ix_choice))
                        ax.quiver(s[filt], u[filt], vs[filt], vu[filt], color=color, alpha=alpha,
			scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, **kwargs)
                    else:
                        ax.scatter(s[filt][::downsample], u[filt][::downsample], s=pointsize,
                                   c=color, alpha=alpha, **kwargs)
                        

                elif by == 'cu':
                    if velocity_arrows:
                        filt = list(set(filt) & set(ix_choice))
                        ax.quiver(u[filt], c[filt], vu[filt], vc[filt], color=color, alpha=alpha,
			scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, **kwargs)
                    else:
                        ax.scatter(u[filt][::downsample], c[filt][::downsample], s=pointsize,
                                   c=color, alpha=alpha, **kwargs)
      
                else:
                    
                    if velocity_arrows:
                        filt = list(set(filt) & set(ix_choice))
                        ax.quiver(s[filt], u[filt],c[filt], 
                                  vs[filt], vu[filt], vc[filt],
                                  color=color, alpha=0.4, length=0.1,
                                arrow_length_ratio=0.5, normalize=True, **kwargs)
                    else:
                        ax.scatter(s[filt][::downsample], u[filt][::downsample], c[filt][::downsample], s=pointsize,c=color, alpha=alpha, **kwargs)
                    
                        

        elif colors is None:
            color = adata[:, gene].layers[color_by].copy()
            color = color.A if sparse.issparse(color) else color
            color = color.ravel()
            
            outlier = 99.8
            non_zero = (u > 0) & (s > 0) & (c > 0)
            non_outlier = u < np.percentile(u, outlier)
            non_outlier &= s < np.percentile(s, outlier)
            non_outlier &= c < np.percentile(c, outlier)
            color -= np.min(color)
            color /= np.max(color)
            
            if by == 'us':
                if velocity_arrows:
                    ax.quiver(s[non_zero & non_outlier][::downsample], 
                            u[non_zero & non_outlier][::downsample],
                            vs[non_zero & non_outlier][::downsample],
                            vu[non_zero & non_outlier][::downsample],
                            
                            np.log1p(color[non_zero & non_outlier][::downsample]), cmap=cmap, alpha=0.7, 
                scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, **kwargs)
                else:
                    ax.scatter(s[non_zero & non_outlier][::downsample], u[non_zero & non_outlier][::downsample], s=pointsize,
                               c=color[non_zero & non_outlier][::downsample],
                        alpha=alpha, cmap=cmap, **kwargs)
            elif by =='cu':
                if velocity_arrows:
                    ax.quiver(u[non_zero & non_outlier][::downsample], 
                            c[non_zero & non_outlier][::downsample],
                            vu[non_zero & non_outlier][::downsample],
                            vc[non_zero & non_outlier][::downsample],
                            color[non_zero & non_outlier][::downsample],
                            cmap=cmap, alpha=0.7, 
                scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, **kwargs)
                else:
                    ax.scatter(u[non_zero & non_outlier][::downsample], c[non_zero & non_outlier][::downsample], s=pointsize, c=color[non_zero & non_outlier][::downsample],
                        alpha=alpha, cmap=cmap, **kwargs)
                    
            else:
                if velocity_arrows:
                    ax.quiver(s[non_zero & non_outlier][::downsample], u[non_zero & non_outlier][::downsample],c[non_zero & non_outlier][::downsample], 
                                vs[non_zero & non_outlier][::downsample], vu[non_zero & non_outlier][::downsample], vc[non_zero & non_outlier][::downsample],
                                c=np.log1p(color[non_zero & non_outlier][::downsample]),
                                                alpha=0.4, length=0.1,
                            arrow_length_ratio=0.5, normalize=True, **kwargs)
                else:
                    ax.scatter(s[non_zero & non_outlier][::downsample], u[non_zero & non_outlier][::downsample], c[non_zero & non_outlier][::downsample], s=pointsize,
                               c=np.log1p(color[non_zero & non_outlier][::downsample]), alpha=alpha, **kwargs)
                
                    
        else:
            if by == 'us':
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                                  vs[::downsample],
                                  vu[::downsample],
                                colors[::downsample], alpha=0.5,
                                scale_units='xy', scale=10, width=0.005,
                                headwidth=4, headaxislength=5.5, cmap=cmap, **kwargs)
                else:
                    ax.scatter(s[::downsample], u[::downsample], s=pointsize,
                                c=colors[::downsample], alpha=0.7, cmap=cmap, **kwargs)

            elif by == 'cu':
                if velocity_arrows:
                    ax.quiver(u[::downsample],
                                c[::downsample],
                                vu[::downsample],
                                vc[::downsample],
                                
                                colors[::downsample], alpha=0.5,
                                scale_units='xy', scale=10, width=0.005,
                                headwidth=4, headaxislength=5.5, cmap=cmap, **kwargs)
                else:
                    ax.scatter(u[::downsample], c[::downsample], s=pointsize,
                                c=colors[::downsample], alpha=0.7, cmap=cmap, **kwargs)

                    
            else:
                ax.scatter(s[::downsample], u[::downsample],
                                c[::downsample], s=pointsize,
                                c=colors[::downsample], alpha=0.7, cmap=cmap, **kwargs)
                if velocity_arrows:
                    ax.quiver(s[::downsample],
                                u[::downsample], c[::downsample],
                                vs[::downsample],
                                vu[::downsample],
                                vc[::downsample],
                                colors[::downsample], alpha=0.4, length=0.1,
                                arrow_length_ratio=0.5, normalize=True,
                                cmap=cmap, **kwargs)


        if by == 'cus' and \
                (view_3d_elev is not None or view_3d_azim is not None):
            ax.view_init(elev=view_3d_elev, azim=view_3d_azim)
        title = gene
        ax.set_title(f'{title}', fontsize=11)
        
        if by == 'us':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
        elif by == 'cu':
            ax.set_xlabel('unspliced' if full_name else 'u')
            ax.set_ylabel('chromatin' if full_name else 'c')
        elif by == 'cus':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
            ax.set_zlabel('chromatin' if full_name else 'c')
        if by in ['us', 'cu']:
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
        elif by == 'cus':
            if not axis_on:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
            if not frame_on:
                ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
                
       
    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()
    return fig, axs

def dynamic_plot(adata,
                 gene_list,
                 by='expression',
                 time_key= 'velo_s_pseudotime',
                 color_by='celltype',
                 gene_time=False,
                 axis_on=True,
                 frame_on=True,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 cmap='coolwarm',
                 **kwargs
                 ):
    """Gene dynamics plot.

    This function plots accessibility, expression, or velocity by time.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` or `~mudata.MuData`
        Data result from dynamics recovery.
    gene_list: `str`,  list of `str`
        List of gene_list to plot.
    by: `str` (default: `expression`)
        Plot accessibilities and expressions if `expression`. Plot velocities
        if `velocity`.
    color_by: `str` (default: `celltype`)
        Color by the four potential states if `state`. Other common values are
        leiden, louvain, cluster, etc.
        If not `celltype`, the color field must be present in `.uns`, which can
        be pre-computed with `scanpy.pl.scatter`.
    time_key: `str` (default: `velo_s_pseudotime`)
        Time key to plot.
        If gene_time is True, the time field must be present in `.layers`. 
        Unless, it must be present in `.obs`. 
    gene_time: `bool` (default: `False`)
        Whether to use individual gene fitted time, or shared global latent
        time.
    axis_on: `bool` (default: `True`)
        Whether to show axis labels.
    frame_on: `bool` (default: `True`)
        Whether to show plot frames.
    downsample: `int` (default: 1)
        How much to downsample the cells. The remaining number will be
        `1/downsample` of original.
    full_range: `bool` (default: `False`)
        Whether to show the full time range of velocities before smoothing or
        subset to only smoothed range.
    figsize: `tuple` (default: `None`)
        Total figure size.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    cmap: `str` (default: `coolwarm`)
        Color map for continuous color key.
    """
    
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    

    if by not in ['expression', 'velocity', 'expectation']:
        raise ValueError('"by" must be either "expression", "expectation" or "velocity".')
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata '
                         'obs, and the colors of categories can be found in '
                         'adata uns.')

    downsample = np.clip(int(downsample), 1, 10)
    gene_list = np.array(gene_list)
    missing_genes = gene_list[~np.isin(gene_list, adata.var_names)]

    gene_list = gene_list[np.isin(gene_list, adata.var_names)]
    gn = len(gene_list)
    if gn == 0:
        return
    
    if time_key is None:
        if gene_time:
            time_key = 'fit_t'
        else:
            time_key = 'velo_s_pseudotime'

    

    fig, axs = plt.subplots(gn, 3, squeeze=False, figsize=(10, 2.3*gn)
                            if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    for row, gene in enumerate(gene_list):
        if by == 'expression':
            c = adata[:, gene].layers['c']
            u = adata[:, gene].layers['u']
            s = adata[:, gene].layers['s'] 
        elif by == 'velocity':
            u = adata[:, gene].layers['velo_u']
            s = adata[:, gene].layers['velo_s']
            c = adata[:, gene].layers['velo_c']
        else:
            u = adata[:, gene].layers['uhat']
            s = adata[:, gene].layers['shat']
            c = adata[:, gene].layers['chat']
        
        c = c.A if sparse.issparse(c) else c
        u = u.A if sparse.issparse(u) else u
        s = s.A if sparse.issparse(s) else s
        c, u, s = np.ravel(c), np.ravel(u), np.ravel(s)


        if types is not None:
            time = np.array(adata[:, gene].layers[time_key] if gene_time 
                            else adata.obs[time_key])
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                filt = np.ravel(filt)
                if np.sum(filt) > 0:
                    axs[row, 0].scatter(time[filt][::downsample],
                                        c[filt][::downsample], s=pointsize,
                                        c=colors[i], alpha=0.6, **kwargs)
                    axs[row, 1].scatter(time[filt][::downsample],
                                        u[filt][::downsample],
                                        s=pointsize, c=colors[i], alpha=0.6, **kwargs)
                    axs[row, 2].scatter(time[filt][::downsample],
                                        s[filt][::downsample], s=pointsize,
                                        c=colors[i], alpha=0.6, **kwargs)
        else:
            time = np.array(adata[:, gene].layers[time_key] if gene_time 
                            else adata.obs[time_key])
            axs[row, 0].scatter(time[::downsample], c[::downsample],
                                s=pointsize,
                                c=colors[::downsample],
                                alpha=0.6, cmap=cmap, **kwargs)
            axs[row, 1].scatter(time[::downsample], u[::downsample],
                                s=pointsize,
                                c=colors[::downsample],
                                alpha=0.6, cmap=cmap, **kwargs)
            axs[row, 2].scatter(time[::downsample], s[::downsample],
                                s=pointsize,
                                c=colors[::downsample],
                                alpha=0.6, cmap=cmap, **kwargs)

        

        

        axs[row, 0].set_title(f'{gene} ATAC' + ('' if by == 'expression'
                              else ' velocity' if by=='velocity' else ' expectation'))
        axs[row, 0].set_xlabel('~t' if by == 'velocity' else 't')
        axs[row, 0].set_ylabel('dc/dt' if by == 'velocity' else 'c')
        axs[row, 1].set_title(f'{gene} unspliced' + ('' if by == 'expression'
                              else ' velocity' if by=='velocity' else ' expectation'))
        axs[row, 1].set_xlabel('~t' if by == 'velocity' else 't')
        axs[row, 1].set_ylabel('du/dt' if by == 'velocity' else 'u')
        axs[row, 2].set_title(f'{gene} spliced' + ('' if by == 'expression'
                              else ' velocity' if by=='velocity' else ' expectation'))
        axs[row, 2].set_xlabel('~t' if by == 'velocity' else 't')
        axs[row, 2].set_ylabel('ds/dt' if by == 'velocity' else 's')

        for j in range(3):
            ax = axs[row, j]
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
        
        
            
    fig.tight_layout()
    return(fig, axs)

def violinplot(adata,
                 gene_list,
                 by='alpha',
                 color_by='celltype',
                 colors=None,
                 axis_on=True,
                 frame_on=True,
                 n_cols=5,
                 figsize=None,
                 type='violin',
                 **kwargs):
    
    gene_list = np.array(gene_list)
    missing_genes = gene_list[~np.isin(gene_list, adata.var_names)]
    
    from pandas.api.types import is_categorical_dtype

    #set colorcode
    if color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]):
        if colors is not None:
            types = adata.obs[color_by].cat.categories
            if set(colors.keys()) & set(types) != set(types):
                raise ValueError("Currently, colors keys don't contain all categories of types.")

        elif color_by+'_colors' in adata.uns.keys():
            types = adata.obs[color_by].cat.categories
            colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'categorical available in adata obs. ')

    gene_list = gene_list[np.isin(gene_list, adata.var_names)]
    gn = len(gene_list)
    if gn == 0:
        return
    elif gn < n_cols:
        n_cols = gn

    fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                figsize=(3.2*n_cols, 2.7*(-(-gn // n_cols)))
                                if figsize is None else figsize)
    fig.patch.set_facecolor('white')
                     
    
    for g, gene in enumerate(gene_list):
        row = g // n_cols
        col = g % n_cols
        ax = axs[row, col]
        df = pd.DataFrame(adata[:, gene].layers[by], 
                          columns=[by],
                          index=adata.obs_names)
        df['cluster'] = adata.obs[color_by]
        if type == 'violin':
            if colors is None:
                sns.violinplot(df, x='cluster', y=by,
                           palette=colors, ax=ax, **kwargs)
            else:
                sns.violinplot(df, x='cluster', y=by,
                           palette=colors, ax=ax, **kwargs)
        elif type == 'box':
            if colors is None:
                sns.boxplot(df, x='cluster', y=by,
                           palette=colors, ax=ax)
            else:
                sns.boxplot(df, x='cluster', y=by,
                           palette=colors, ax=ax)
        ax.set_title(gene)
        if not axis_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
    fig.tight_layout()
    return fig, axs
        
        
