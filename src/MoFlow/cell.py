import matplotlib.pyplot as plt
import numpy as np


def scatter_cell(adata,
                 gene_list,
                 by='umap',
                 color_by='s',
                 cmap='coolwarm',
                 vmax=None,
                 vmin=None,
                 center=None,
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 figsize=None,
                 pointsize=2,
                 alpha=.5,
                 color_bar=False,
                 **kwargs
                 ):
    """Plot the kinetic parameters ('alpha_c', 'alpha', 'beta', 'gamma', 'splice', 'unsplice', or 'pseudotime') of one gene on the embedding space.
        
    Arguments
    ---------
    adata: :class:`~anndata.AnnData` or `~mudata.MuData`
        Data result from dynamics recovery.
    gene_list: `str`,  list of `str`
        List of gene_list to plot.
    by: `str` (default: `s`)
    
    """  
    
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    
    if color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    elif color_by in adata.layers and is_numeric_dtype(adata.layers[color_by]):
        types = None
        colors=None
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata '
                         'obs, and the colors of categories can be found in '
                         'adata uns.')
        
        
    gene_list = np.array(gene_list)
    missing_genes = gene_list[~np.isin(gene_list, adata.var_names)]

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
    
    embed = adata.obsm[f'X_{by}']
    x = embed[:, 0]
    y = embed[:, 1]
    
    for g, gene in enumerate(gene_list):
        row = g // n_cols
        col = g % n_cols
        ax = axs[row, col]
        
        if types is not None:
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                filt = np.arange(adata.n_obs)[np.ravel(filt)]
                if type(colors) == dict:
                    color = colors[types[i]]
                
                ax.scatter(x[filt], y[filt], s=pointsize, c=color, alpha=alpha,
                 **kwargs)
        else:
            if colors is not None:
                im = ax.scatter(x, y,  s=pointsize, c=colors, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            else:
                color = adata[:, gene].layers[color_by]
                if center is not None:
                    vmax = max(abs(color))
                    vmin = -vmax
                im = ax.scatter(x, y, s=pointsize, c=color, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            if color_bar:
                plt.colorbar(im, ax=ax)
                
        title = gene
        ax.set_title(f'{title}', fontsize=11)
        
        if not frame_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_frame_on(False)
    fig.tight_layout()
    return fig, axs

def scatter_cell_multi(adata,
                 gene,
                 by='umap',
                 color_by='expression',
                 vmax=None,
                 vmin=None,
                 center=None,
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 figsize=None,
                 pointsize=2,
                 alpha=.5,
                 cmap='coolwarm',
                 **kwargs
                 ):
    """Plot the kinetic parameters ('alpha', 'beta', 'gamma', 'splice', 'unsplice', or 'pseudotime') of one gene on the embedding space.
        
    Arguments
    ---------
    adata: :class:`~anndata.AnnData` or `~mudata.MuData`
        Data result from dynamics recovery.
    gene_list: `str`,  list of `str`
        List of gene_list to plot.
    by: `str` (default: `s`)
    
    """  
    

    if gene not in adata.var_names:
        raise ValueError()

    if color_by == 'expression':
        fig, axs = plt.subplots(1, 3, squeeze=False,
                                figsize=(3.2*3, 2.7)
                                if figsize is None else figsize)
    elif color_by == 'velocity':
        fig, axs = plt.subplots(1, 3, squeeze=False,
                                figsize=(3.2*3, 2.7)
                                if figsize is None else figsize)
    elif color_by =='parameter':
        fig, axs = plt.subplots(1, 4, squeeze=False,
                                figsize=(3.2*4, 2.7)
                                if figsize is None else figsize)
    elif color_by == 'all':
        fig, axs = plt.subplots(3, 4, squeeze=False,
                                figsize=(3.2*4, 2.7*3)
                                if figsize is None else figsize)
    else:
        raise ValueError('color key must be the one of expression,' 
                         'velocity, parameter, or all')

    fig.patch.set_facecolor('white')
    
    embed = adata.obsm[f'X_{by}']
    x = embed[:, 0]
    y = embed[:, 1]
    
    if color_by == 'expression':
        if vmax is None:
            vmax = 1
        if vmin is None:
            vmin = 0
        for i, c in zip(range(3), ['c', 'u', 's']):
            ax = axs[0, i]
            color = adata[:, gene].layers[c]
            
            im = ax.scatter(x, y,  s=pointsize, c=color, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            ax.set_title(c)
            plt.colorbar(im, ax=ax)
    elif color_by == 'velocity':
        if center is None:
            center = 0

        for i, c in zip(range(3), ['velo_c', 'velo_u', 'velo_s']):
            ax = axs[0, i]
            color = adata[:, gene].layers[c]
            vmax = max(abs(color))
            vmin = -vmax
            im = ax.scatter(x, y,  s=pointsize, c=color, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            ax.set_title(c)
            plt.colorbar(im, ax=ax)
    elif color_by == 'parameter':
        for i, c in enumerate(['alpha_c', 'alpha', 'beta', 'gamma']):
            ax = axs[0, i]
            color = adata[:, gene].layers[c]
            im = ax.scatter(x, y,  s=pointsize, c=color, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            ax.set_title(c)
            plt.colorbar(im, ax=ax)
    else:
        for i, c in zip(range(3), ['c', 'u', 's']):
            ax = axs[0, i]
            color = adata[:, gene].layers[c]
            vmax = 1
            vmin = 0
            cmap = 'coolwarm'
            im = ax.scatter(x, y,  s=pointsize, c=color, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            ax.set_title(c)
            plt.colorbar(im, ax=ax)
        for i, c in zip(range(3), ['velo_c', 'velo_u', 'velo_s']):
            ax = axs[1, i]
            color = adata[:, gene].layers[c]
            vmax = max(abs(color))
            vmin = -vmax
            im = ax.scatter(x, y,  s=pointsize, c=color, alpha=alpha,
                       vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)
            ax.set_title(c)
            plt.colorbar(im, ax=ax)
        for i, c in zip(range(4), ['alpha_c', 'alpha', 'beta', 'gamma']):
            ax = axs[2, i]
            color = adata[:, gene].layers[c]
            cmap = 'gnuplot'
            im = ax.scatter(x, y,  s=pointsize, c=color, alpha=alpha,
                       cmap=cmap, **kwargs)
            ax.set_title(c)
            plt.colorbar(im, ax=ax)
    for ax in axs:
        if not frame_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_frame_on(False)
    fig.tight_layout()
    return fig, axs
            