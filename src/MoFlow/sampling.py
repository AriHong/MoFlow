import pandas as pd
import numpy as np
from numpy.core.fromnumeric import size
import scipy
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt



def sampling_neighbors(gene_cus, step=(30,30), percentile=25):
    embed_len = gene_cus.shape[1]
    if embed_len != len(step):
        step = int(step[0]/(embed_len))
        step = np.repeat(step, embed_len)
    from scipy.stats import norm
    def gaussian_kernel(X, mu = 0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    grs = []
    
    for dim_i in range(embed_len):
        m, M = np.min(gene_cus[:, dim_i]), np.max(gene_cus[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, step[dim_i])
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(10) # set random seed
    
    nn = NearestNeighbors()

    neighbors_1 = min((gene_cus[:,:embed_len].shape[0]-1), 20)
    nn.fit(gene_cus[:,:embed_len])
    dist, ixs = nn.kneighbors(gridpoints_coordinates, neighbors_1)

    ix_choice = ixs[:,0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()

    neighbors_2 = min((gene_cus.shape[0]-1), 20)
    nn.fit(gene_cus[:,0:embed_len])
    dist, ixs = nn.kneighbors(gene_cus[ix_choice, 0:embed_len], neighbors_2)
    
    density_estimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_estimate > np.percentile(density_estimate, percentile)
    ix_choice = ix_choice[bool_density]
    return(ix_choice)

def sampling_inverse(gene_cus,target_amount=500):
    embed_len = gene_cus.shape[1]
    values = np.vstack([embed_len[:,i] for i in range(embed_len)])
    
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    # p2 = (1/p)/sum(1/p)
    p2 = (1/p)/sum(1/p)
    idx = np.arange(values.shape[1])
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=target_amount)
    return(idx_choice)

def sampling_circle(gene_cus,target_amount=500):
    embed_len = gene_cus.shape[1]
    values = np.vstack([embed_len[:,i] for i in range(embed_len)])
    
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    idx = np.arange(values.shape[1])
    tmp_p = np.square((1-(p/(max(p)))**2))+0.0001
    p2 = tmp_p/sum(tmp_p)
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=target_amount)
    return(idx_choice)

def sampling_random(gene_cus, target_amount=500):
    idx = np.random.choice(gene_cus.shape[0], size = target_amount, replace=False)
    return(idx)
    


def sampling_embedding(embedding,
                       para,
                    target_amount=500,
                    step=(30,30),
                      percentile=25):

    '''
    Guangyu
    '''
    if para == 'neighbors':
        idx = sampling_neighbors(embedding,step, percentile)
    elif para == 'inverse':
        #print('inverse')
        idx = sampling_inverse(embedding,target_amount)
    elif para == 'circle':
        idx = sampling_circle(embedding,target_amount)
    elif para == 'random':
        # print('random')
        idx = sampling_random(embedding,target_amount)
    else:
        print('downsampling_method is neighbors/inverse/circle/random')
    return(idx)


def adata_to_detail(data, para, gene):
    '''
    convert adata to detail format
    data: an anndata
    para: the varable name of unsplice, splice, and gene name
    para = ['Mc', 'Mu', 'Ms']
    '''
    data2 = data[:, data.var.index.isin([gene])].copy()
    chromosome = data2.layers[para[0]][:,0].copy().astype(np.float32)
    unsplice = data2.layers[para[1]][:,0].copy().astype(np.float32)
    splice = data2.layers[para[2]][:,0].copy().astype(np.float32)
    detail = pd.DataFrame({'gene_name':gene, 'chromosome': chromosome, 
                           'unsplice':unsplice, 'splice':splice})
    return(detail)

def downsampling_embedding(c, u, s, embedding,
                           para,
                           target_amount,
                           step,
                           n_neighbors,
                           embeddingnames=None,
                           expression_scale=None,
                           projection_neighbor_choice='embedding',
                          pca_n_components=None,
                           umap_n=None,
                           umap_n_components=None):
   
        
    
    if step is not None:
        idx_downSampling_embedding = sampling_embedding(embedding,
                    para=para,
                    target_amount=target_amount,
                    step=step)
    else:
        idx_downSampling_embedding=range(0,embedding.shape[0]) # all cells

    
    def transfer(c, u, s, expression_scale):
        

        if expression_scale=='log':
            c = np.log(c + 0.000001)
            u = np.log(u + 0.000001)
            s = np.log(s + 0.000001)
            
        elif expression_scale=='2power':
            c = 2**(c)
            u = 2**(u)
            s = 2**(s)
    
        elif expression_scale=='power10':
            c = c**10
            u = u**10
            s = s**10
            
        return (c, u, s)

    c, u, s =transfer(c, u, s, expression_scale)
    

    if projection_neighbor_choice=='gene':
        embedding_downsampling = s[idx_downSampling_embedding]
        
    elif projection_neighbor_choice=='pca': # not use
        from sklearn.decomposition import PCA
        #print('using pca projection_neighbor_choice')
        embedding_downsampling_0 = mdata.mod[rna_mod].layers['Ms'].iloc[idx_downSampling_embedding]
        pca=PCA(n_components=pca_n_components)
        pca.fit(embedding_downsampling_0)
        embedding_downsampling = pca.transform(embedding_downsampling_0)[:,range(pca_n_components)]
        
    elif projection_neighbor_choice=='pca_norm':
        from sklearn.decomposition import PCA
        #print('pca_norm')
        embedding_downsampling_0 = s[idx_downSampling_embedding]
        
        pca=PCA(n_components=pca_n_components)
        pca.fit(embedding_downsampling_0)
        embedding_downsampling_trans = pca.transform(embedding_downsampling_0)[:,range(pca_n_components)]
        embedding_downsampling_trans_norm=(embedding_downsampling_trans - embedding_downsampling_trans.min(0)) / embedding_downsampling_trans.ptp(0)#normalize
        embedding_downsampling_trans_norm_mult10=embedding_downsampling_trans_norm*10 #optional
        embedding_downsampling=embedding_downsampling_trans_norm_mult10**5 # optional
        
    elif projection_neighbor_choice=='embedding':
        embedding_downsampling = embedding[idx_downSampling_embedding]

    elif projection_neighbor_choice =='umap':
        import umap
        embedding_downsampling_0 = s[idx_downSampling_embedding]
        
        def get_umap(df,n_neighbors=umap_n, min_dist=0.1, n_components=umap_n_components, metric='euclidean'): 
            fit = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric
            )
            embed = fit.fit_transform(df);
            return(embed)
        embedding_downsampling=get_umap(embedding_downsampling_0)

    n_neighbors = min(int((embedding_downsampling.shape[0])/4), n_neighbors)
    if n_neighbors==0:
        n_neighbors=1
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embedding_downsampling) 
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
    return(embedding_downsampling, idx_downSampling_embedding, embedding_knn)

def downsampling(data_df, gene_list, downsampling_ixs):
    '''
    Guangyu
    '''
    data_df_downsampled=pd.DataFrame()
    for gene in gene_list:
        data_df_one_gene=data_df[data_df['gene_name']==gene]
        data_df_one_gene_downsampled = data_df_one_gene.iloc[downsampling_ixs]
        data_df_downsampled=data_df_downsampled.append(data_df_one_gene_downsampled)
    return(data_df_downsampled)
