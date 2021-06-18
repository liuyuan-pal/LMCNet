import os

import cv2
import numpy as np
import torch

from network.lmcnet import LMCNet
from utils.base_utils import pts_to_hpts, hpts_to_pts
from utils.filter_utils import get_sift_corr_feats, sp2_feats_stats
from utils.eig_utils import build_graph, name2config

class LMCNetFilter:
    default_cfg={
        "eig_dir": 'data/eig_cache'
    }
    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        if not os.path.exists(self.cfg['eig_dir']):
            os.system(f'mkdir -p {self.cfg["eig_dir"]}')

        self.network=LMCNet(cfg).cuda()
        restore_model=torch.load(os.path.join('data/model', cfg['name'], 'model_best.pth'))
        if 'step' in restore_model: print(f'loading step {restore_model["step"]} ...')
        state_dict = restore_model['network_state_dict']
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def prepare_input(self,xs,image_feats,geom_feats,all_name):
        # all_name "dataset_name-seq_name-id0-id1-det_name-desc_name-match_name"
        fn = f'{self.cfg["eig_dir"]}/{all_name}'
        if os.path.exists(fn):
            eig=np.load(fn)
            eig_val, eig_vec = eig[0], eig[1:,:]
        else:
            eig_val, eig_vec = build_graph(xs, name2config[self.cfg['eig_type']])
            np.save(fn,np.concatenate([eig_val[None,:],eig_vec],0))

        xs=torch.from_numpy(xs.astype(np.float32)).unsqueeze(0).cuda()
        eig_val=torch.from_numpy(eig_val.astype(np.float32)).unsqueeze(0).cuda()
        eig_vec=torch.from_numpy(eig_vec.astype(np.float32)).unsqueeze(0).cuda()
        image_feats=torch.from_numpy(image_feats.astype(np.float32)).unsqueeze(0).cuda()
        geom_feats=torch.from_numpy(geom_feats.astype(np.float32)).unsqueeze(0).cuda()
        data={'xs': xs, 'eig_val': eig_val, 'eig_vec': eig_vec,
              'image_feats': image_feats, 'geom_feats': geom_feats}
        return data

    def compute_logits(self, xs, image_feats, geom_feats, name):
        data = self.prepare_input(xs, image_feats, geom_feats, name)
        with torch.no_grad():
            logits = self.network(data)['logits'].cpu().numpy()[0]
        return logits

    def prepare_call(self,kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs):
        matches=matches_info[:,:2].astype(np.int32)
        x0=hpts_to_pts(pts_to_hpts(kps0[matches[:,0]][:,:2]) @ np.linalg.inv(K0).T)
        x1=hpts_to_pts(pts_to_hpts(kps1[matches[:,1]][:,:2]) @ np.linalg.inv(K1).T)
        xs=np.concatenate([x0,x1],1)

        det_name=kwargs['det_name']
        match_name=kwargs['match_name']
        if det_name.startswith('sift'):
            resp0,resp1=kps0[matches[:,0]][:,2],kps1[matches[:,1]][:,2]
            size0,size1=kps0[matches[:,0]][:,3],kps1[matches[:,1]][:,3]
            angle0,angle1=kps0[matches[:,0]][:,4],kps1[matches[:,1]][:,4]
            desc0,desc1=desc0[matches[:,0]],desc1[matches[:,1]]
            image_feats,geom_feats=get_sift_corr_feats(desc0,desc1,size0,size1,angle0,angle1,resp0,resp1)
        elif det_name.startswith('superpoint'):
            desc0, desc1 = desc0[matches[:, 0]], desc1[matches[:, 1]]
            kps_info0, kps_info1 = kps0[matches[:, 0]], kps1[matches[:, 1]]
            image_feats = desc0 - desc1  # n,256
            scores = np.stack([kps_info0[:, 2], kps_info1[:, 2]], 1)  # n,2
            scores = (scores - sp2_feats_stats['scores_mean']) / sp2_feats_stats['scores_std']
            if match_name.startswith('superglue'):
                sg_scores = matches_info[:, 2]
                sg_scores = (sg_scores - sp2_feats_stats['sg_scores_mean']) / sp2_feats_stats['sg_scores_std']
                sg_mutual = matches_info[:, 3]
                sg_mutual = sg_mutual - 0.5  # normalize to [-0.5,0.5]
                geom_feats = np.concatenate([scores, sg_scores[:, None], sg_mutual[:, None]], 1)
            else:
                geom_feats = scores
        else:
            print(det_name)
            raise NotImplementedError
        return xs, image_feats, geom_feats

    def __call__(self, kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs):
        xs, image_feats, geom_feats = self.prepare_call(kps0, kps1, matches_info, img0, img1, name, K0, K1, desc0, desc1, **kwargs)
        logits=self.compute_logits(xs,image_feats,geom_feats,name)
        logits[logits>=10]=10
        prob=np.exp(logits)/(1+np.exp(logits))
        return prob>self.cfg['prob_thresh']

class LMFFilter:
    def __init__(self,cfg):
        self.eig_type=cfg['eig_type']
        self.thresh=cfg['thresh']
        self.eta=cfg['eta']

    def __call__(self, kps0, kps1, matches, img0, img1, name, K0, K1, *args, **kwargs):
        kps0, kps1 = kps0[:, :2], kps1[:, :2]
        x0=hpts_to_pts(pts_to_hpts(kps0[matches[:,0]]) @ np.linalg.inv(K0).T)
        x1=hpts_to_pts(pts_to_hpts(kps1[matches[:,1]]) @ np.linalg.inv(K1).T)
        xs=np.concatenate([x0,x1],1).astype(np.float32)

        if name=='do it!':
            eig_val, eig_vec = build_graph(xs,name2config[self.eig_type])
        elif name.startswith('detrac_'):
            _, ext_name, match_name, seq_name, id0, id1 = name.split('/')
            fn=f'data/detrac_train_cache/{ext_name}-{match_name}-corr/{self.eig_type}/{seq_name}/{id0}-{id1}.npy'
            if os.path.exists(fn):
                eig = np.load(fn)
                eig_val, eig_vec = eig[0], eig[1:, :]
            else:
                eig_val, eig_vec = build_graph(xs,name2config[self.eig_type])
                if not os.path.exists(f'data/detrac_train_cache/{ext_name}-{match_name}-corr/{self.eig_type}/{seq_name}'):
                    os.system(f'mkdir data/detrac_train_cache/{ext_name}-{match_name}-corr/{self.eig_type}/{seq_name} -p')
                np.save(fn, np.concatenate([eig_val[None, :], eig_vec], 0))
        else:
            if name.startswith('sun3d'):
                dataset_name, det_name, desc_name, matcher_name, seq_name, id0, id1 = name.split('/')
                # if dataset_name.endswith('_subset'): dataset_name=dataset_name[:-7]
            else:
                dataset_name,det_name,desc_name,matcher_name,seq_name,id0,id1=name.split('-')
            fn=f'data/yfcc_eval_eig_cache/{dataset_name}-{det_name}-{desc_name}-' \
               f'{matcher_name}-{self.eig_type}/{seq_name}-{id0}-{id1}.npy'

            if os.path.exists(fn):
                eig=np.load(fn)
                eig_val, eig_vec = eig[0], eig[1:,:]
            else:
                eig_val, eig_vec = build_graph(xs,name2config[self.eig_type])
                if not os.path.exists(f'data/yfcc_eval_eig_cache/{dataset_name}-{det_name}-'
                                      f'{desc_name}-{matcher_name}-{self.eig_type}'):
                    os.mkdir(f'data/yfcc_eval_eig_cache/{dataset_name}-{det_name}-{desc_name}-'
                             f'{matcher_name}-{self.eig_type}')
                np.save(fn,np.concatenate([eig_val[None,:],eig_vec],0))

        motion=x1-x0
        motion_smooth = eig_vec @ ((eig_vec.T @ motion) / (1+self.eta*eig_val)[:,None])
        motion_res=motion_smooth-motion
        motion_res_norm=np.linalg.norm(motion_res,2,1)
        if 'output_res' in kwargs and kwargs['output_res']:
            return motion_res_norm
        return motion_res_norm<self.thresh

name2filter={
    'lmcnet': LMCNetFilter,
}