import os
import time

import numpy as np
import cv2
import torch

from torch.utils.data import Dataset

from utils.base_utils import read_pickle, hpts_to_pts, pts_to_hpts, \
    compute_dR_dt, np_skew_symmetric, epipolar_distance_mean
from utils.filter_utils import sift_feats_stats, get_sift_corr_feats, sp2_feats_stats


def generate_virtual_positive_corr(F):
    step = 0.1
    xx,yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
    # Points in first image before projection
    pts1 = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
    # Points in second image before projection
    pts2 = np.float32(pts1)
    pts1, pts2 = pts1.reshape(1,-1,2), pts2.reshape(1,-1,2)
    pts1, pts2 = cv2.correctMatches(F.reshape(3, 3), pts1, pts2)
    return np.concatenate([pts1[0],pts2[0]],1).astype(np.float32)


class RelPoseFeatsDataset(Dataset):
    def __init__(self, root_dir, ext_name, match_name, pair_info_fn,
                 epipolar_thresh, use_eig, eig_name, use_feats=True, is_train=False):
        self.root_dir=root_dir
        self.match_name=match_name
        self.ext_name=ext_name
        self.eig_name=eig_name
        self.pair_infos=read_pickle(pair_info_fn)
        self.epipolar_thresh=epipolar_thresh
        self.use_eig=use_eig
        self.use_feats=use_feats
        self.is_train=is_train

    def __getitem__(self, index):
        np.random.seed((index+int(time.time()))%(2**16))
        if self.is_train:
            index=index % (len(self.pair_infos))
        pair_info=self.pair_infos[index]
        seq_name, id0, id1 = pair_info
        matches_info=np.load(os.path.join(self.root_dir,self.ext_name,self.match_name,seq_name,f'{id0}-{id1}.npy'))
        matches=matches_info[:,:2].astype(np.int32)

        results={}
        if self.use_eig:
            eig=np.load(os.path.join(self.root_dir,self.ext_name,self.match_name,self.eig_name,seq_name,f'{id0}-{id1}.npy'))
            eig_val, eig_vec = eig[0], eig[1:]
            results['eig_val']=eig_val
            results['eig_vec']=eig_vec

        # randomly swap
        if np.random.random()<0.5:
            id0, id1 = id1, id0
            matches=matches[:,(1,0)]

        # load data
        data0=np.load(os.path.join(self.root_dir,self.ext_name,seq_name,f'{id0}.npz'))
        data1=np.load(os.path.join(self.root_dir,self.ext_name,seq_name,f'{id1}.npz'))
        kps0,desc0,K0,R0,t0=data0['kps'][:,:2],data0['desc'],data0['K'],data0['R'],data0['t']
        kps1,desc1,K1,R1,t1=data1['kps'][:,:2],data1['desc'],data1['K'],data1['R'],data1['t']
        kps0=hpts_to_pts(pts_to_hpts(kps0) @ np.linalg.inv(K0).T)
        kps1=hpts_to_pts(pts_to_hpts(kps1) @ np.linalg.inv(K1).T)

        # gen vxs
        R,t=compute_dR_dt(R0,t0,R1,t1)
        F=np_skew_symmetric(t) @ R
        vxs=generate_virtual_positive_corr(F)
        vxs[np.isnan(vxs)]=0
        vxs[np.isinf(vxs)]=0
        xs=np.concatenate([kps0[matches[:,0]],kps1[matches[:,1]]],1)
        ys=epipolar_distance_mean(xs[:,:2],xs[:,2:],F)<self.epipolar_thresh
        results.update({'xs': xs.astype(np.float32), 'ys': ys.astype(np.float32), 'vxs': vxs.astype(np.float32),
                        'R': R.astype(np.float32), 't': t.astype(np.float32),})
        ############process feats#############
        if self.use_feats:
            if self.ext_name.startswith('sift'):
                desc0, desc1 = desc0[matches[:,0]], desc1[matches[:,1]]
                kps_info0, kps_info1 = data0['kps'][matches[:,0]], data1['kps'][matches[:,1]]
                resp0, resp1 = kps_info0[:, 2], kps_info1[:, 2]
                size0, size1 = kps_info0[:, 3], kps_info1[:, 3]
                angle0, angle1 = kps_info0[:, 4], kps_info1[:, 4]
                image_feats, geom_feats = get_sift_corr_feats(desc0,desc1,size0,size1,angle0,angle1,resp0,resp1,sift_feats_stats)
            elif self.ext_name.startswith('superpoint'):
                desc0, desc1 = desc0[matches[:,0]], desc1[matches[:,1]]
                kps_info0, kps_info1 = data0['kps'][matches[:,0]], data1['kps'][matches[:,1]]
                image_feats = desc0 - desc1 # n,256
                scores = np.stack([kps_info0[:, 2], kps_info1[:, 2]], 1)  # n,2
                scores = (scores-sp2_feats_stats['scores_mean'])/sp2_feats_stats['scores_std']
                if self.match_name.startswith('superglue'):
                    sg_scores=matches_info[:,2]
                    sg_scores=(sg_scores-sp2_feats_stats['sg_scores_mean'])/sp2_feats_stats['sg_scores_std']
                    sg_mutual=matches_info[:,3]
                    sg_mutual=sg_mutual-0.5 # normalize to [-0.5,0.5]
                    geom_feats = np.concatenate([scores,sg_scores[:,None],sg_mutual[:,None]],1)
                else:
                    geom_feats = scores
            else:
                raise NotImplementedError
            results.update({'image_feats': image_feats, 'geom_feats': geom_feats})
        return results


    def __len__(self):
        if self.is_train:
            return 9999999
        else:
            return len(self.pair_infos)


def collate_fn(data_list):
    pns=[data['xs'].shape[0] for data in data_list]
    min_pn=np.min(pns)

    other_names = []
    output_data = {}
    candidate_other_names=['R','t','eig_val','vxs']
    candidate_sample_names=['eig_vec','geom_feats','image_feats','xs','ys']
    for name in candidate_other_names:
        if name in data_list[0]:
            other_names.append(name)
            output_data[name]=[]

    for name in candidate_sample_names:
        if name in data_list[0]:
            output_data[name]=[]

    for pn, data in zip(pns,data_list):
        if pn==min_pn:
            for name in candidate_sample_names:
                if name in output_data:
                    output_data[name].append(np.expand_dims(data[name],0))
        else:
            idxs=np.arange(pn)
            np.random.shuffle(idxs)
            idxs=idxs[:min_pn]
            for name in candidate_sample_names:
                if name in output_data:
                    output_data[name].append(np.expand_dims(data[name][idxs],0))

        for name in other_names:
            output_data[name].append(np.expand_dims(data[name],0))

    for k, v in output_data.items():
        if k=='ys':
            v=np.concatenate(v,0).astype(np.float32)
            output_data[k] = torch.from_numpy(v)
        else:
            output_data[k] = torch.from_numpy(np.concatenate(v,0))

    return output_data