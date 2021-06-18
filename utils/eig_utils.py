import numpy as np
import pyflann

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

eig_config = {
    'use_spectral': True,
    'use_all': False,
    'use_sparse': True,
    'sigma': 0.1,
    'eig_k': 128,
    'graph_k': 32,
    'use_min_connect': False
}
eig_config_256_min = {
    'use_spectral': True,
    'use_all': False,
    'use_sparse': True,
    'sigma': 0.1,
    'eig_k': 256,
    'graph_k': 32,
    'use_min_connect': True,
    'min_connect': 5e-2,
}
eig_config_32 = {
    'use_spectral': True,
    'use_all': False,
    'use_sparse': True,
    'sigma': 0.1,
    'eig_k': 32,
    'graph_k': 8,
    'use_min_connect': False
}
eig_config_32_min = {
    'use_spectral': True,
    'use_all': False,
    'use_sparse': True,
    'sigma': 0.1,
    'eig_k': 32,
    'graph_k': 8,
    'use_min_connect': True,
    'min_connect': 5e-2,
}
eig_config_hard = {
    'use_spectral': True,
    'use_all': False,
    'use_sparse': True,
    'sigma': 0.2,
    'eig_k': 128,
    'graph_k': 32,
    'use_min_connect': False,
    'min_connect': 5e-2,
}
name2config = {
    'default': eig_config,
    'small': eig_config_32,
    'small_min': eig_config_32_min,
    'large_min': eig_config_256_min,
    'hard': eig_config_hard,
}

def build_graph(xs,config):
    n=xs.shape[0]
    k=config['graph_k']
    # eig_k=min(config['eig_k'],n)
    if config['use_all']:
        dist_sq=np.sum((xs[:,None,:]-xs[None,:,:])**2,2)
        adj=np.exp(-dist_sq/config['sigma']**2)
        degree=np.sum(adj,1)
        adj = (np.diag(1 / np.sqrt(degree))) @ \
              adj @ (np.diag(1 / np.sqrt(degree)))
    else:
        # construct sparse graph
        flann=pyflann.FLANN()
        idxs,dist_sq=flann.nn(xs, xs, k+1, algorithm='linear')

        weights=np.exp(-dist_sq/config['sigma']**2)
        ########################################
        if config['use_min_connect']:
            weights[weights<config['min_connect']]=config['min_connect']

        if config['use_sparse']:
            ids0=np.repeat(np.arange(n)[:,None],k+1,1).flatten()
            ids1=idxs.flatten()
            data=weights.flatten()
            adj=coo_matrix((data,(ids0.astype(np.int32),ids1.astype(np.int32))),shape=(n,n))
            adj.setdiag(np.zeros(n))
            adj=(adj.T+adj)/2
            degree = np.asarray(np.sum(adj, 1))[:, 0]
            degree_sqrt_inv = 1/np.sqrt(degree)
            adj = adj.multiply(degree_sqrt_inv[:,None])
            adj = adj.multiply(degree_sqrt_inv[None,:])
        else:
            adj=np.zeros([n,n],np.float64)
            adj[np.arange(n)[:,None],idxs]=weights
            # adj[idxs,np.arange(n)[:,None]]=weights
            adj = (adj.T + adj) / 2
            degree=np.sum(adj,1)
            adj = (np.diag(1 / np.sqrt(degree))) @ \
                  adj @ (np.diag(1 / np.sqrt(degree)))

    if config['use_sparse']:
        degree = np.asarray(np.sum(adj, 1))[:,0]
        laplacian = -adj
        laplacian.setdiag(degree)
        eig_val,eig_vec=eigsh(laplacian,min(config['eig_k'],laplacian.shape[0]-1),which='SA')
    else:
        degree = np.sum(adj, 1)
        laplacian = -adj
        laplacian[np.arange(n),np.arange(n)]=degree
        eig_val,eig_vec=np.linalg.eigh(laplacian)
        eig_val,eig_vec=eig_val[:config['eig_k']],eig_vec[:,:config['eig_k']]

    return eig_val, eig_vec