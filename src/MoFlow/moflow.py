import os
import glob
import pandas as pd
import numpy as np
import random
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from torch.utils.data import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm
import pkg_resources
import warnings

from multiprocessing import Manager

from .sampling import *

def cosine_similarity(cnv, unv, snv, vc, vu, vs, rna_only=False):
    """Cost function
    Return:
        list of cosine distance and a list of the index of the next cell
    """
    if rna_only:
        nv = torch.stack([unv, snv], dim=-1)
        v = torch.stack([vu, vs], dim=-1)

        cosine = F.cosine_similarity(nv, v, dim=-1)

        cosine_max, cosine_max_idx = torch.max(cosine, dim=0)
        return 1 - cosine_max , cosine_max_idx


    else:
        nv = torch.stack([cnv, unv, snv], dim=-1)
        v = torch.stack([vc, vu, vs], dim=-1)

        cosine = F.cosine_similarity(nv, v, dim=-1)

        cosine_max, cosine_max_idx = torch.max(cosine, dim=0)
        return 1 - cosine_max, cosine_max_idx



def get_loss(c, u, s, param, indices, 
             direction=None, rna_only=False, warm=False
             ):
    
    if rna_only:
        alpha, beta, gamma = param
        n = c.shape[0]
        
        vu = alpha - beta*u
        vs = beta*u - gamma*s
        
        indices_ = indices[:, 1:].T
        
        cnv, unv, snv = c[indices_] - c, u[indices_] - u, s[indices_] - s   
        cosine_, cosine_idx= cosine_similarity(cnv, unv, snv, vc, vu, vs, rna_only=True)
        
        velo = [vc, vu, vs]
        loss = torch.mean(cosine_)
        
        stateon = (vc>=0).type(torch.float32)
        stateoff = (vc<0).type(torch.float32)
        states = [stateon, stateoff]

        return loss, velo, param, states
    else:
        alpha_c, alpha, beta, gamma = param

        n = c.shape[0]
        vcopen = alpha_c*(1-c)
        vcclose = alpha_c*(-c)

        vu = alpha*c - beta*u
        vs = beta*u - gamma*s
        indices_ = indices[:, 1:].T

        cnv, unv, snv = c[indices_] - c, u[indices_] - u, s[indices_] - s   
        

        vc = torch.stack([vcopen, vcclose], dim=1)

        if warm:
            if direction=='on':
                vu = torch.clamp(vu, 0., None)
                vs = torch.clamp(vs, 0., None)
            elif direction=='off':
                vu = torch.clamp(vu, None, 0.)
                vs = torch.clamp(vs, None, 0.)
            else:
                _ = (u<s)
                vu[_] = torch.clamp(vu[_], None, 0.)
                vs[_] = torch.clamp(vs[_], None, 0.)
                vu[~_] = torch.clamp(vu[~_], 0., None)
                vs[~_] = torch.clamp(vs[~_], 0., None)
                
            cosine_, cosine_idx= cosine_similarity(cnv, unv, snv, vc, vu, vs, rna_only=True)
            cnv_, unv_, snv_ = cnv[cosine_idx, torch.arange(n)], unv[cosine_idx, torch.arange(n)], snv[cosine_idx, torch.arange(n)]
            nv = torch.stack([cnv_, unv_, snv_], dim=-1)
            
            vopen = torch.stack([vcopen, vu, vs], dim=-1)
            vclose = torch.stack([vcclose, vu, vs], dim=-1)

            cosine1 = 1-F.cosine_similarity(nv, vopen, dim=-1)
            cosine2 = 1-F.cosine_similarity(nv, vclose, dim=-1)

            cosineloss = torch.stack([cosine1, cosine2], dim=-1)
            cosineloss, _ = torch.min(cosineloss, dim=1) 

            vc = vc[torch.arange(n), _]
            loss = torch.mean(cosine1 + cosine2)/2 #choice between vc open or close is not crucial in warmup phase
            
        else:
            cosine_, cosine_idx= cosine_similarity(cnv, unv, snv, vc, vu, vs, rna_only=True)
            cnv_, unv_, snv_ = cnv[cosine_idx, torch.arange(n)], unv[cosine_idx, torch.arange(n)], snv[cosine_idx, torch.arange(n)]
            nv = torch.stack([cnv_, unv_, snv_], dim=-1)
            vopen = torch.stack([vcopen, vu, vs], dim=-1)
            vclose = torch.stack([vcclose, vu, vs], dim=-1)

            cosine1 = 1-F.cosine_similarity(nv, vopen, dim=-1)
            cosine2 = 1-F.cosine_similarity(nv, vclose, dim=-1)

            cosineloss = torch.stack([cosine1, cosine2], dim=-1)

            cosineloss, _ = torch.min(cosineloss, dim=1)

            vc = vc[torch.arange(n), _]
            loss = torch.mean(cosineloss)

        param = [alpha_c, alpha, beta, gamma] 

        vc = vc[indices_]
        vc = torch.mean(vc, dim=0)
        velo = [vc, vu, vs]

        stateon = (vc>=0).type(torch.float32)
        stateoff = (vc<0).type(torch.float32)
        states = [stateon, stateoff]
        
        return loss, velo, param, states, direction


class DNN_layer(nn.Module):

    """Define network structure.
    """

    def __init__(self, rna_only=False, 
                 direction_='complete'):
        super().__init__()
        self.rna_only = rna_only
        self.direction = direction_
        
        if self.rna_only:
            self.l1 = nn.Linear(2, 100)
            self.l2 = nn.Linear(100, 100)
            self.l3 = nn.Linear(100, 3)
        
        else:
            
            self.param = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 48),
                nn.ReLU(),
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Sigmoid()
            )
            
            self.param_c = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 48),
                nn.ReLU(),
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        if self.direction == 'on':
            self.alpha0 = nn.Parameter(torch.tensor(2.0))
            self.gamma0 = nn.Parameter(torch.tensor(1.0))
            
        else:
            self.alpha0 = nn.Parameter(torch.tensor(1.0))
            self.gamma0 = nn.Parameter(torch.tensor(1.0))
            

    def forward(self, c, u, s, indices, epoch=None):
        if self.rna_only:
            
            input = torch.stack([u, s], dim=1)
            
            x = self.l1(input)
            x = F.leaky_relu(x)
            x = self.l2(x)
            x = F.leaky_relu(x)
            x = self.l3(x)
            output = torch.sigmoid(x)
            alpha = output[:,0]
            beta = output[:,1]
            gamma = output[:,2]
            c = torch.ones_like(u)
            
        else:
            n = c.shape[0]
                    
            input = torch.stack([c, u, s], dim=1)
            
            output= self.param(input)
            self.alpha0.data.clamp_(1., 10.)
            self.gamma0.data.clamp_(.5, 1.)
            alpha = output[:,0]*self.alpha0
            beta = output[:,1]
            gamma = output[:,2]*self.gamma0

            alpha_c = self.param_c(input)[:, 0]

            param = [alpha_c, alpha, beta, gamma]
            
            if epoch > 20:
                warm=False
            else:
                warm=True
            
            loss, velo, param, states, direction = get_loss(
                c, u, s, param, indices, self.direction, self.rna_only, warm )
            vc, vu, vs = velo
            alpha_c, alpha, beta, gamma = param

        
        return loss, vc, vu, vs, alpha_c, alpha, beta, gamma, states
          

    def load(self, model_name=None):

        model_name = model_name + '.pt'
        model_path = pkg_resources.resource_stream( __name__,os.path.join('model/', model_name)).name
        checkpoint = torch.load(model_path)

        self.l1 = checkpoint["l1"]
        self.l2 = checkpoint["l2"]
        self.l3 = checkpoint["l3"]
            
class DNN_module(nn.Module):
    '''
    calculate loss function
    load network "DNN_layer"
    predict splice_predict and unsplice_predict
    '''
    def __init__(self, module, n_neighbors = None, rna_only=False, device=None):
        super().__init__()
        self.module = module
        self.n_neighbors = n_neighbors
        self.rna_only=rna_only
        if device is None:
            device = torch.device('cpu')
        else:
            self.device = device

    def velocity_calculate(self, 
                           c,
                           u,
                           s,
                           embedding1,
                           embedding2, 
                        epoch=None,
                        test=False):
        '''
        add embedding
        for real dataset
        calculate loss function
        predict unsplice_predict splice_predict from network 
        '''
        
        #generate neighbor indices and expr dataframe

        
        if self.rna_only:
            points = np.array([u.cpu().numpy(), s.cpu().numpy()]).transpose()
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        
            distances, indices = nbrs.kneighbors(points) 
            indices = torch.tensor(indices)
            
        else:
            points = np.array([c.cpu().numpy(), u.cpu().numpy(), s.cpu().numpy()]).transpose()
            distance_matrix = cdist(points, points, metric='mahalanobis')
            indices = np.argsort(distance_matrix, axis=1)[:, :self.n_neighbors]

        loss, vc, vu, vs, alpha_c, alpha, beta, gamma, states = self.module(
            c, u, s, indices, epoch)
        
        if torch.isnan(vc).any():
            print('vc has nan')
        if torch.isinf(vc).any():
            print('vc has inf')
        
        return loss, vc, vu, vs, alpha_c, alpha, beta, gamma, states


    def summary_para_validation(self, cost_mean): 
        loss_df = pd.DataFrame({'cost': cost_mean}, index=[0])
        return(loss_df)


    def summary_para(self, c, u, s,
                     vc, vu, vs,
                    alpha_c,
                    alpha, beta, gamma, 
                     stateon, stateoff
                    ): 
        output_df = pd.merge(pd.DataFrame(c, columns=['c']),pd.DataFrame(u, columns=['u']), left_index=True, right_index=True) 
        
        output_df['s'] = s
        output_df['vc'] = vc
        output_df['vu'] = vu
        output_df['vs'] = vs
        output_df['alpha_c'] = alpha_c
        output_df['alpha'] = alpha
        output_df['beta'] = beta
        output_df['gamma'] = gamma
        output_df['stateon'] = stateon
        output_df['stateoff'] = stateoff

        return output_df



class ltModule(pl.LightningModule):
    '''
    train network using "DNN_module"
    '''
    def __init__(self, 
                backbone=None, 
                learning_rate=None,
                optimizer='Adam',
                cost_type='smooth',
                average_cost_window_size=10,
                smooth_weight=0.9
                ):
        super().__init__()
        self.backbone = backbone
        self.validation_loss = []
        self.learning_rate=learning_rate
        self.optimizer=optimizer
        self.save_hyperparameters()
        self.get_loss=1000
        self.cost_type=cost_type
        self.average_cost_window_size=average_cost_window_size # will be used only when cost_tpye.isin(['average', 'median'])
        self.cost_window=[]
        self.smooth_weight=smooth_weight
        
    def save(self, model_path):
        self.backbone.module.save(model_path)    # save network

    def load(self, model_name=None):
        self.backbone.module.load(model_name)   # load network of cellDancer

    def configure_optimizers(self):     # define optimizer
        if self.optimizer=="Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=10**(-8), weight_decay=0.004, amsgrad=False)

        elif self.optimizer=="SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.8)


        return optimizer

    def training_step(self, batch, batch_idx):
        '''
        traning network
        batch: [] output returned from realDataset.__getitem__
        
        '''

        cs, us, ss, embedding1s, embedding2s, gene_names, _ = batch
        c, u, s, embedding1, embedding2, gene_name= cs[0], us[0], ss[0], embedding1s[0], embedding2s[0], gene_names[0]
        
        cost, vc, vu, vs, alpha_c, alpha, beta, gamma, states = self.backbone.velocity_calculate( \
                c, u, s, embedding1, embedding2, self.current_epoch)

        if self.cost_type=='average': # keep the window len <= check_val_every_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.mean(torch.stack(self.cost_window))
            self.log("loss", self.get_loss)
            
        elif self.cost_type=='median': # keep the window len <= check_val_every_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.median(torch.stack(self.cost_window))
            self.log("loss", self.get_loss)
            
        elif self.cost_type=='smooth':
            if self.get_loss==1000:
                self.get_loss=cost
            smoothed_val = cost * self.smooth_weight + (1 - self.smooth_weight) * self.get_loss  # calculate smoothed value
            self.get_loss = smoothed_val  
            self.log("loss", self.get_loss)
        else:
            self.get_loss = cost
            self.log("loss", self.get_loss) 

        return cost


    def validation_step(self, batch, batch_idx):
        '''
        predict unsplice_predict, splice_predict on the training dataset
        '''
        if self.current_epoch!=0:
            cost = self.get_loss.data.cpu().numpy()
            self.validation_loss.append(cost)

    def test_step(self, batch, batch_idx):
        cs, us, ss, embedding1s, embedding2s, gene_names, _ = batch
        c, u, s, embedding1, embedding2, gene_name= cs[0], us[0], ss[0], embedding1s[0], embedding2s[0], gene_names[0]
        
        cost, vc, vu, vs, alpha_c, alpha, beta, gamma, states = \
        self.backbone.velocity_calculate(
            c, u, s, 
            embedding1, embedding2, self.current_epoch, test=True)
        
        stateon, stateoff= states
        self.test_df= self.backbone.summary_para(
            c.data.cpu().numpy(), u.data.cpu().numpy(), s.data.cpu().numpy(),
            vc.data.cpu().numpy(), vu.data.cpu().numpy(), vs.data.cpu().numpy(),
                    alpha_c.data.cpu().numpy(), 
                    alpha.data.cpu().numpy(), beta.data.cpu().numpy(), gamma.data.cpu().numpy(), 
            stateon.data.cpu().numpy(), stateoff.data.cpu().numpy(),
        )
        self.test_df.insert(0, "gene_name", gene_name)
        self.test_df.insert(0, "cellIndex", self.test_df.index)
        self.test_df.insert(len(self.test_df.columns), "epoch", self.current_epoch)
        self.test_df.insert(len(self.test_df.columns), "loss", cost.item())

class getItem(Dataset): 
    '''
    load training and test data
    '''
    def __init__(self, c, u, s, embedding,
                 rna_only=False,
                 data_fit=None, data_predict=None, 
                 datastatus="predict_dataset", gene_name=None,
                 permutation_ratio=0.1, norm_data=True, norm_cell_distribution=False): 
        self.c = c
        self.u = u
        self.s = s
        self.embedding = embedding
        self.data_fit=data_fit
        self.data_predict=data_predict
        self.rna_only=rna_only
        self.datastatus=datastatus
        self.permutation_ratio=permutation_ratio
        self.norm_data=norm_data
        self.norm_max_unsplice=None
        self.norm_max_splice=None
        self.norm_cell_distribution=norm_cell_distribution
        self.gene_name = gene_name
        

        
        gpdist = pairwise_distances(
                embedding)
        subset_cells = s > 0.1*np.percentile(s, 99)
        subset_cells = np.where(subset_cells)[0]
        if len(subset_cells) > 3000:
            rng = np.random.default_rng(2021)
            subset_cells = rng.choice(subset_cells, 3000, replace=False)
        local_pdist = gpdist[np.ix_(subset_cells, subset_cells)]
        dists = (np.ravel(local_pdist[np.triu_indices_from(local_pdist, k=1)])
                 .reshape(-1, 1))
        self.local_std = np.std(dists)
        
    def __len__(self):
        return len(self.gene_name) # gene count

    def __getitem__(self, idx):
        gene_name = self.gene_name[idx]
        if self.datastatus=="fit_dataset":
            # random sampling in each epoch
            data_fitting=self.data_fit
            if self.permutation_ratio==1:
                data=data_fitting
            elif (self.permutation_ratio<1) & (self.permutation_ratio>0):
                data=np.random.choice(data_fitting, 
                                      np.round(len(data_fitting)*self.permutation_ratio).astype(int),
                                      replace=False)
            else:
                print('sampling ratio is wrong!')
            
            c = self.c[data, idx]
            u = self.u[data, idx]
            s = self.s[data, idx]
            
            
            embedding = self.embedding[data]
            embedding1 = embedding[:, 0]
            embedding2 = embedding[:, 1]
            
            
        elif self.datastatus=="predict_dataset":
            data=self.data_predict
            c = self.c[data, idx]
            u = self.u[data, idx]
            s = self.s[data, idx]
                        
            embedding = self.embedding[data]
            embedding1 = embedding[:, 0]
            embedding2 = embedding[:, 1]
            
        c = c.astype(np.float32)
        u = u.astype(np.float32)
        s = s.astype(np.float32)
        
        cmax = np.float32(max(c)) 
        umax = np.float32(max(u)) 
        smax = np.float32(max(s))

        u = u/umax
        s = s/smax
        
        if self.rna_only:
            c = np.ones_like(u)
        else:
            c = c/cmax

        # add embedding
        embedding1 = np.array(embedding1.astype(np.float32))
        embedding2 = np.array(embedding2.astype(np.float32))


        return c, u, s, embedding1, embedding2, gene_name, self.local_std



class feedData(pl.LightningDataModule):
    '''
    load training and test data
    '''
    def __init__(self, c, u, s, embedding,
                 rna_only=False, data_fit=None, data_predict=None, gene_name=None,
                 permutation_ratio=1, norm_data=True, norm_cell_distribution=False):
        super().__init__()

        self.fit_dataset = getItem(c, u, s, embedding,
                                   rna_only=rna_only,
                                   data_fit=data_fit, data_predict=data_predict,
                                   gene_name=gene_name,
                                   datastatus="fit_dataset", permutation_ratio=permutation_ratio, norm_data=norm_data, norm_cell_distribution=norm_cell_distribution)
        
        self.predict_dataset = getItem(c, u, s, embedding,
                                       rna_only=rna_only, 
                                       data_fit=data_fit, data_predict=data_predict,
                                       gene_name=gene_name,
                                       datastatus="predict_dataset", permutation_ratio=permutation_ratio,norm_data=norm_data)

    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.fit_dataset = Subset(self.fit_dataset, indices)
        temp.predict_dataset = Subset(self.predict_dataset, indices)
        return temp

    def train_dataloader(self):
        return DataLoader(self.fit_dataset,num_workers=0)
    def val_dataloader(self):
        return DataLoader(self.fit_dataset,num_workers=0)
    def test_dataloader(self):
        return DataLoader(self.predict_dataset,num_workers=0,)


def build_datamodule(c, u, s, embedding,
                     rna_only=False,
                     speed_up=True,
                     norm_data=True,
                     permutation_ratio=None, 
                     norm_cell_distribution=False, 
                     gene_list=None,
                     gene_name=None,
                     downsample_method='neighbors',
                     n_neighbors_downsample=10,
                     step=(200,200),
                     downsample_target_amount=None):

    '''
    get data from target gene
    gene is annotated as index using gene_list
    set fitting data, data to be predicted, and sampling ratio when fitting
    '''
    step_i=step[0]
    step_j=step[1]
    
    if not gene_list is None:
        c = c[:, gene_list]
        u = u[:, gene_list]
        s = s[:, gene_list]    
    unsampled_idx = np.arange(c.shape[0])
     
    if speed_up:
        _, sampling_idx, _ = downsampling_embedding(c, u, s, embedding, 
                            para=downsample_method,
                            target_amount=downsample_target_amount,
                            step=(step_i,step_j),
                            n_neighbors=n_neighbors_downsample,
                            projection_neighbor_choice='embedding')
        
        feed_data = feedData(c, u, s, embedding, 
                             rna_only=rna_only, 
                             data_fit = sampling_idx, data_predict=unsampled_idx,
                             gene_name = gene_name, 
                             permutation_ratio=permutation_ratio, norm_data=norm_data,
                             norm_cell_distribution=norm_cell_distribution) # default 
    else:
        feed_data = feedData(c, u, s, embedding,
                             rna_only=rna_only, 
                             data_fit=unsampled_idx, data_predict=unsampled_idx,
                             gene_name=gene_name, 
                             permutation_ratio=permutation_ratio, norm_data=norm_data,
                             norm_cell_distribution=norm_cell_distribution) 

    return(feed_data)

class MOFlow():
    def __init__(self,
                 adata, 
                 adata_atac=None,
                 rna_only=False,
                 gene_list=None,
                 min_epoches=20,
                 max_epoches=200, 
                 check_val_every_n_epoch=10,
                 patience=3,
                 learning_rate=0.001,
                 n_neighbors=40,
                 embed='X_umap',
                 permutation_ratio=0.125,
                 speed_up=True,
                 norm_data=True,
                 norm_cell_distribution=True,
                 save_path=None,
                 folder_name=None,
                 device=None):
        """Velocity estimation for each cell.
        
        Arguments
        ---------
        adata: `anndata.AnnData`
            anndata that contains the unspliced abundance, spliced abundance, embedding space, and cell type information. 
        adata_atac: `anndata.AnnData`
            anndata that contains the chromatin accessibility.
        gene_list: optional, `list` (default: None)
            Gene list for velocity estimation. `None` if to estimate the velocity of all genes.
        min_epoches: optional, `int` (default: 20)
            Minimum epoches to update the network
        max_epoches: optional, `int` (default: 200)
            Stop to update the network once this number of epochs is reached.
        check_val_every_n_epoch: optional, `int` (default: 10)
            Check loss every n train epochs.
        patience: optional, `int` (default: 3)
            Number of checks with no improvement after which training will be stopped.
        permutation_ratio: optional, `float` (default: 0.125)
            Sampling ratio of cells in each epoch when training each gene.
        speed_up: optional, `bool` (default: True)
            `True` if speed up by downsampling cells. `False` if to use all cells to train the model.
        norm_data: optional, `bool` (default: True)
            `True` if normalize unsplice (and splice) reads by dividing max value of unspliced (and spliced) reads.
        norm_cell_distribution: optional, `bool` (default: True)
            `True` if the bias of cell distribution is to be removed on embedding space (many cells share the same position of unspliced (and spliced) reads).
        n_jobs: optional, `int` (default: -1)
            The maximum number of concurrently running jobs.
        save_path: optional, `str` (default: None)
            Path to save the result of velocity estimation.
        save_path: optional, `str` (default: None)
            folder name to save the result of velocity estimation.
        Returns
        -------
        loss_df: `pandas.DataFrame`
            The record of loss.
        """
        
        # set output dir


        if save_path is None:
            save_path=os.getcwd()
        if folder_name is None:
            folder_name = 'MoFlow_velocity_'

        os.makedirs(os.path.join(save_path,folder_name), exist_ok = True)
        save_path=os.path.join(save_path,folder_name)
        os.makedirs(os.path.join(save_path,'TEMP'), exist_ok = True)
        
        self.save_path = save_path
        if os.path.exists(save_path):
            [os.remove(f) for f in glob.glob(f'{self.save_path}/TEMP/*.csv')]
            
        print('Using '+ save_path +' as the output path.')
        
        
        # set gene_list if not given
        if gene_list is None:
            gene_list=adata.var_names
        else:
            adata = adata[:, adata.var_names.isin(gene_list)]
            if not rna_only:
                adata_atac = adata_atac[:, adata_atac.var_names.isin(gene_list)]
            gene_list = list(list(set(adata.var_names).intersection(set(gene_list))))
        gene_list_ = np.arange(len(gene_list))
        self.G = len(gene_list)
        self.gene_list = gene_list_ #as index
        self.gene_name = adata.var_names #gene name
        self.gene_list_batch = None
        
        self.rna_only = rna_only
        self.speed_up = speed_up
        self.norm_data = norm_data
        self.permutation_ratio = permutation_ratio
        self.norm_cell_distribution = norm_cell_distribution
        self.min_epoches = min_epoches
        self.max_epoches = max_epoches
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.patience = patience
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate

        if device is None:
            self.accelerator="cpu"
            self.devices="auto"
        else:
            self.accelerator="gpu"
            self.devices="-1"

        u, s = adata.layers['Mu'], adata.layers['Ms'] 
        if rna_only:
            c = np.ones_like(u)
        else:
            c = adata_atac.layers['Mc']

        self.c = c.toarray() if sparse.issparse(c) else c
        self.u = u.toarray() if sparse.issparse(u) else u
        self.s = s.toarray() if sparse.issparse(s) else s
        try:
            self.embedding = adata.obsm[embed]
        except KeyError:
            print("Embedding not found! Set to None.")
            self.embedding = np.nan*np.ones((adata.n_obs, 2))
            
    def _train_thread(self,
                      datamodule, 
                      data_indices,
                      model_save_path=None):
        try:
            seed = 0
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            #decide model
            selected_data = datamodule.subset(data_indices)

            c, u, s, embedding1, embedding2, gene_name, local_std = selected_data.predict_dataset.__getitem__(0)
            embedding = np.stack((embedding1, embedding2), axis=1)



            # iniate network (DNN_layer) and loss function (DynamicModule)

            model_ = select_initial_net(c, u, s, gene_name, self.rna_only, embedding, local_std)

            if model_ == 'low_quality':
                return gene_name
            elif self.rna_only:
                direction_, margin_ = None, None
            else:
                print(gene_name, model_)
                model_, direction_, margin_ = model_


            backbone = DNN_module(DNN_layer(self.rna_only, direction_),
                                n_neighbors=self.n_neighbors, rna_only=self.rna_only)
            model = ltModule(backbone=backbone, learning_rate=self.learning_rate)
            #if self.rna_only:
            #    model.load(model_)


            early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.0, patience=self.patience, mode='min')

            if self.check_val_every_n_epoch is None:
                # not use early stop
                trainer = pl.Trainer(
                    max_epochs=self.max_epoches, 
                    min_epochs = self.min_epoches,
                    reload_dataloaders_every_n_epochs=1, 
                    log_every_n_steps=self.max_epoches+1,
                    logger = False,
                    enable_checkpointing = False,
                    enable_model_summary=False,
                    accelerator=self.accelerator,
                    devices=self.devices,
                    enable_progress_bar=False,
                    )
            else:
                # use early stop
                trainer = pl.Trainer(
                    max_epochs=self.max_epoches, 
                    min_epochs = self.min_epoches,
                    reload_dataloaders_every_n_epochs=1, 
                    log_every_n_steps=self.max_epoches+1,
                    logger = False,
                    enable_checkpointing = False,
                    check_val_every_n_epoch = self.check_val_every_n_epoch,
                    enable_model_summary=False,
                    callbacks=[early_stop_callback],
                    accelerator=self.accelerator,
                    devices=self.devices,
                    enable_progress_bar=False,
                    )

            if self.max_epoches > 0:
                trainer.fit(model, selected_data)   # train network

            trainer.test(model, selected_data, verbose=False)    # predict

            if(model_save_path != None):
                model.save(model_save_path)

            loss = model.validation_loss
            moflow_df = model.test_df


            if(model_save_path != None):
                model.save(model_save_path)

            moflow_df.insert(0, "direction", direction_)
            moflow_df.insert(0, "slope", margin_)
            
            moflow_df.to_csv(os.path.join(self.save_path, 'TEMP', ('moflow_estimation_'+gene_name+'.csv')), index=False)
                
            return None
        except:
            return gene_name


            
            
    def velocity(self, adata, 
                 model_save_path=None, n_jobs=-1, file_path=None, file_name=None):
        """
        n_jobs: optional, `int` (default: -1)
            The maximum number of concurrently running jobs.
        """
        
        #buring
        
        gene_idx_buring = [self.gene_list[0]]
        
        datamodule=build_datamodule(self.c, self.u, self.s, self.embedding,
                                    self.rna_only, self.speed_up,
                                    self.norm_data,
                                    self.permutation_ratio, self.norm_cell_distribution,
                                    gene_list=gene_idx_buring,
                                    gene_name=self.gene_name[0])

        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._train_thread)(
                datamodule = datamodule,
                data_indices=[data_index])
            for data_index in range(0,len(gene_idx_buring)))
        
        # clean directory
        #shutil.rmtree(os.path.join(save_path,'TEMP'))
        #os.mkdir(os.path.join(save_path,'TEMP'))
        
        unpredicted_gene_lst = []
        
        id_ranges=list()
        if n_jobs==-1:
            interval=os.cpu_count()
        else:
            interval=n_jobs
        for i in range(0, self.G, interval):
            idx_start=i
            if self.G < i+interval:
                idx_end = self.G
            else:
                idx_end=i+interval
            id_ranges.append((idx_start,idx_end))


        print('Arranging genes for parallel job.')
        if len(id_ranges)==1:
            if id_ranges==1:
                print(self.G,' gene was arranged to ',len(id_ranges),' portion.')
            else:
                print(self.G,' genes were arranged to ',len(id_ranges),' portion.')
        else: 
            print(self.G,' genes were arranged to ',len(id_ranges),' portions.')

        unpredicted_gene_lst=list()

        for id_range in tqdm(id_ranges, desc="Velocity Estimation", total=len(id_ranges), position=1, leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

            self.gene_list_batch=self.gene_list[id_range[0]:id_range[1]]
            gene_name_batch=self.gene_name[id_range[0]:id_range[1]]
            
            datamodule=build_datamodule(self.c, self.u, self.s, self.embedding,
                                        self.rna_only, self.speed_up, 
                                        self.norm_data, 
                                        self.permutation_ratio, self.norm_cell_distribution,
                                        gene_list=self.gene_list_batch, gene_name=gene_name_batch)

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self._train_thread)(
                datamodule = datamodule,
                data_indices=[data_index])
                for data_index in range(0,len(self.gene_list_batch)))
            gene_name_lst=[x for x in result if x is not None]
            
            for i in gene_name_lst:
                unpredicted_gene_lst.append(i)
        if len(unpredicted_gene_lst)!=0:
            self.gene_name = set(self.gene_name) - set(unpredicted_gene_lst)


        
                
        def combine_csv(save_path, files):
            with open(save_path,"wb") as fout:
                # first file:
                with open(files[0], "rb") as f:
                    fout.write(f.read())
                # the rest:    
                for filepath in files[1:]:
                    with open(filepath, "rb") as f:
                        next(f)
                        fout.write(f.read())
            return(pd.read_csv(save_path))
        
        moflow_df = os.path.join(self.save_path,'TEMP', "moflow_estimation*.csv")
        moflow_df_files = glob.glob(moflow_df)
        moflow_df=combine_csv(os.path.join(self.save_path, "moflow_estimation.csv"), moflow_df_files)
        
        moflow_df = moflow_df[moflow_df['gene_name'].isin(self.gene_name)]
        moflow_df = moflow_df.dropna()
        
        c_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='c')
        u_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='u')
        s_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='s')
        vc_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='vc')
        vu_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='vu')
        vs_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='vs')
        alpha_c_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='alpha_c')
        alpha_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='alpha')
        beta_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='beta')
        gamma_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='gamma')
        stateon_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='stateon')
        stateoff_ = moflow_df.pivot_table(index='cellIndex', columns='gene_name', values='stateoff')
        direction_ = moflow_df[['gene_name', 'direction']].drop_duplicates().set_index('gene_name')
        slope_ = moflow_df[['gene_name', 'slope']].drop_duplicates().set_index('gene_name')
        loss = moflow_df[['gene_name', 'loss']].drop_duplicates().set_index('gene_name')
        epoch = moflow_df[['gene_name', 'epoch']].drop_duplicates().set_index('gene_name')
        
        os.makedirs(file_path, exist_ok=True)
        
        gene_name_ = c_.columns.values
        adata = adata[:, adata.var_names.isin(gene_name_)]
        
        adata.layers['Mc'] = c_
        adata.layers['c'] = c_
        adata.layers['u'] = u_
        adata.layers['s'] = s_
        
        adata.layers['alpha'] = alpha_
        adata.layers['beta'] = beta_
        adata.layers['gamma'] = gamma_
        
        adata.layers['alpha_c'] = alpha_c_

        adata.layers['velo_c'] = vc_
        adata.layers['velo_u'] = vu_
        adata.layers['velo_s'] = vs_
        adata.layers['stateon'] = stateon_
        adata.layers['stateoff'] = stateoff_

        adata.var['loss'] = loss
        adata.var['epoch'] = epoch
        adata.var['direction'] = direction_
        adata.var['slope'] = slope_
        
        if file_name is not None:
            print('write data')
            adata.write(f"{file_path}/{file_name}")   
            print('Result is perfectly saved at ' + file_path + '/' + file_name)

            





def select_initial_net(
    c, u, s, gene_name, rna_only, embedding, local_std, fit_slope=True):
    '''
    check if right top conner has cells
    circle.pt is the model for single kinetic
    branch.pt is multiple kinetic
    '''
    s_max=np.max(s)
    u_max = np.max(u)
    c_max = np.max(c)
    s_max_90per = 0.9*s_max
    u_max_90per = 0.9*u_max
    
    if s_max <=1e-4:
        return 'low_quality'
    if u_max <=1e-4:
        return 'low_quality'
    if c_max <1e-4:
        return 'low_quality'

    cells_corner = (s>s_max_90per) & (u>u_max_90per)
    if np.sum(cells_corner) > 0.001*u.shape[0]: 
        model_r= 'circle'
    else:
        model_r = 'branch'
    

    if rna_only:
        return model_r
    else:
        
        w_non_zero = ((c >= 0.1 * np.max(c)) &
                      (u >= 0.1 * np.max(u)) &
                      (s >= 0.1 * np.max(s)))
        u_non_zero = u[w_non_zero]
        s_non_zero = s[w_non_zero]
        if len(u_non_zero) < 10:
            low_quality = True
            return 'low_quality'

        # GMM
        w_low = ((np.percentile(s_non_zero, 30) <= s_non_zero) &
                 (s_non_zero <= np.percentile(s_non_zero, 40)))
        if np.sum(w_low) < 10:
            fit_gmm = False
            partial = True
        else:

            pdist = pairwise_distances(
                embedding[w_non_zero, :][w_low, :])
            dists = (np.ravel(pdist[np.triu_indices_from(pdist, k=1)])
                     .reshape(-1, 1))
            model = GaussianMixture(n_components=2, covariance_type='tied',
                                    random_state=2021).fit(dists)
            mean_diff = np.abs(model.means_[1][0] - model.means_[0][0])
            criterion1 = mean_diff > local_std 
            criterion2 = np.all(model.weights_[1] > 0.2)

            if criterion1 and criterion2:
                partial = False
            else:
                partial = True

        # steady-state slope
        wu = u >= np.percentile(u_non_zero, 95)
        ws = s >= np.percentile(s_non_zero, 95)
        ss_u = u[wu | ws]
        ss_s = s[wu | ws]
        if np.all(ss_u == 0) or np.all(ss_s == 0):
            return 'low_quality'
        gamma = np.dot(ss_u, ss_s) / np.dot(ss_s, ss_s)

        # thickness of phase portrait
        u_norm = u_non_zero / np.max(u)
        s_norm = s_non_zero / np.max(s)
        exp = np.hstack((np.reshape(u_norm, (-1, 1)),
                         np.reshape(s_norm, (-1, 1))))
        U, S, Vh = np.linalg.svd(exp)
        thickness = S[1]

        # slope-based direction decision
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = u /s
        non_naninf = (~np.isnan(slope))&(~np.isinf(slope))
        slope = slope[non_naninf]
        on = slope >= gamma
        off = slope < gamma
        if len(ss_u) < 10 or len(u_non_zero) < 10:
            fit_slope = False
            direction = 'complete'
        
        else:
            slope_ = u_non_zero / s_non_zero
            on_ = slope_ >= gamma
            off_ = slope_ < gamma
            on_dist = np.sum((u_non_zero[on_] - gamma * s_non_zero[on_])**2)
            off_dist = np.sum((gamma * s_non_zero[off_] - u_non_zero[off_])**2)

            if thickness < 1.5:
                narrow = True
            else:
                narrow = False

            if on_dist > 10 * off_dist:
                direction = 'on'
                partial = True
            elif off_dist > 10  * on_dist:
                direction = 'off'
                partial = True
            else:
                if partial is True:
                    if on_dist > 3 * off_dist:
                        direction = 'on'
                    elif off_dist > 3 * on_dist:
                        direction = 'off'
                    else:
                        if narrow:
                            direction = 'on'
                        else:
                            direction = 'complete'
                            partial = False
                else:
                    if narrow:
                        direction = ('off'
                                          if off_dist > 2  * on_dist
                                          else 'on')
                        partial = True
                    else:
                        direction = 'complete'

        # model pre-determination
        if direction == 'on':
            model_ = 1
        elif direction == 'off':
            model_ = 2
        else: #complete
            c_high = c >= np.mean(c) + 2 * np.std(c)
            c_high = c_high[non_naninf]
            if np.sum(c_high) < 10:
                c_high = c >= np.mean(c) + np.std(c)
                c_high = c_high[non_naninf]
            if np.sum(c_high) < 10:
                c_high = c >= np.percentile(c, 90)
                c_high = c_high[non_naninf]
            if np.sum(c[non_naninf][c_high] == 0) > 0.5*np.sum(c_high):
                return 'low_quality'
            c_high_on = np.sum(c_high & on)
            c_high_off = np.sum(c_high & off)
            if c_high_on > c_high_off:
                model_ = 1
            else:
                model_ = 2
        
    
    
    return(model_r, direction, np.median(slope))


