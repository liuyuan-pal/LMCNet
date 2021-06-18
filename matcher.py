import torch
from pyflann import FLANN
import numpy as np

from others.superpoint import process_image
from utils.extend_utils.extend_utils_fn import \
    find_nearest_point_idx, find_first_and_second_nearest_point
from others.superglue import SuperGlue


class Matcher:
    def __init__(self, cfg):
        self.mutual_best=cfg['mutual_best']
        self.ratio_test=cfg['ratio_test']
        self.ratio=cfg['ratio']
        self.use_cuda=cfg['cuda']
        self.flann=FLANN()
        if self.use_cuda:
            self.match_fn_1=lambda desc0,desc1: find_nearest_point_idx(desc1, desc0)
            self.match_fn_2=lambda desc0,desc1: find_first_and_second_nearest_point(desc1, desc0)
        else:
            self.match_fn_1=lambda desc0,desc1: self.flann.nn(desc1, desc0, 1, algorithm='linear')
            self.match_fn_2=lambda desc0,desc1: self.flann.nn(desc1, desc0, 2, algorithm='linear')

    def match(self,desc0,desc1,*args,**kwargs):
        mask=np.ones(desc0.shape[0],dtype=np.bool)
        if self.ratio_test:
            idxs,dists = self.match_fn_2(desc0,desc1)

            dists=np.sqrt(dists) # note the distance is squared
            ratio_mask=dists[:,0]/dists[:,1]<self.ratio
            mask&=ratio_mask
            idxs=idxs[:,0]
        else:
            idxs,_=self.match_fn_1(desc0,desc1)

        if self.mutual_best:
            idxs_mutual,_=self.match_fn_1(desc1,desc0)
            mutual_mask = np.arange(desc0.shape[0]) == idxs_mutual[idxs]
            mask&=mutual_mask

        matches=np.concatenate([np.arange(desc0.shape[0])[:,None],idxs[:,None]],axis=1)
        matches=matches[mask]

        return matches

class SuperGlueMatcher:
    def __init__(self,cfg):
        self.spg=SuperGlue(cfg).eval().cuda()
        self.cfg=cfg

    def match(self,desc0,desc1,kps0,kps1,image0,image1,*args,**kwargs):
        image0,img0,scales0=process_image(image0,'cuda',(640,480),False)
        image1,img1,scales1=process_image(image1,'cuda',(640,480),False)
        kps0_xy=np.round(kps0[:,:2]/scales0)
        kps1_xy=np.round(kps1[:,:2]/scales1)
        data={
            'descriptors0':torch.from_numpy(desc0.astype(np.float32)).permute(1,0).unsqueeze(0).cuda(),
            'descriptors1':torch.from_numpy(desc1.astype(np.float32)).permute(1,0).unsqueeze(0).cuda(),
            'keypoints0':torch.from_numpy(kps0_xy.astype(np.float32)).unsqueeze(0).cuda(),
            'keypoints1':torch.from_numpy(kps1_xy.astype(np.float32)).unsqueeze(0).cuda(),
            'scores0':torch.from_numpy(kps0[:,2].astype(np.float32)).unsqueeze(0).cuda(),
            'scores1':torch.from_numpy(kps1[:,2].astype(np.float32)).unsqueeze(0).cuda(),
            'image0':img0,'image1':img1
        }

        with torch.no_grad():
            results=self.spg(data)

        matches=results['matches0'].cpu().numpy()[0]
        scores=results['matching_scores0'].cpu().numpy()[0]
        mutual=results['mutual0'].cpu().numpy()[0]
        valid=matches>-1
        matches=np.stack([np.arange(matches.shape[0]),matches],1)
        if not self.cfg['use_match_directly']:
            return matches[valid]
        else:
            matches = np.concatenate([matches, scores[:, None], mutual[:, None]], 1)
            return matches


name2matcher={
    'default': Matcher,
    'superglue': SuperGlueMatcher,
}