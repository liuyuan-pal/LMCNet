import argparse
import os
import random

import yaml
from skimage.io import imsave
from tqdm import tqdm

from metrics import evaluate_R_t, pose_metrics
from utils.base_utils import save_h5, load_h5, load_component, load_cfg, get_stem, compute_precision_recall_np

from filter import name2filter
from dataset.pose_dataset import PoseSequenceDataset, name2datalist, ScanNetDataset
from descriptor import name2desc, DummyDescriptor
from detector import name2det, SuperPointDetector
from estimator import name2estimator
from matcher import name2matcher

import numpy as np


class PoseEvaluator:
    def __init__(self, cfg_fn):
        cfg=load_cfg(cfg_fn) if type(cfg_fn)!=dict else cfg_fn
        self.descriptor=load_component(name2desc,cfg['desc'])
        self.detector=load_component(name2det,cfg['det'])
        self.matcher=load_component(name2matcher,cfg['matcher'])
        self.filter=load_component(name2filter,cfg['filter']) if 'filter' in cfg else None
        self.estimator=load_component(name2estimator,cfg['estimator'])
        self.eval_name=get_stem(cfg_fn) if type(cfg_fn)!=dict else cfg['name']

        self.desc_name=get_stem(cfg['desc'])
        self.det_name=get_stem(cfg['det'])
        self.matcher_name=get_stem(cfg['matcher'])
        self.estimator_name=get_stem(cfg['estimator'])
        self.filter_name=get_stem(cfg['filter']) if 'filter' in cfg else ''

    def get_geom_fn(self, dataset: PoseSequenceDataset):
        matcher_filter_name=f'{self.matcher_name}-{self.filter_name}' if self.filter is not None else self.matcher_name
        geom_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{matcher_filter_name}-{self.estimator_name}')
        return geom_fn

    def extract_kps_desc(self, dataset: PoseSequenceDataset):
        det_fn=dataset.cache_fn(f'{self.det_name}')
        desc_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}')
        if not os.path.exists(det_fn) and not os.path.exists(desc_fn):
            print(f'extract {self.det_name} kps and {self.desc_name} desc ...')
            kps_dict,desc_dict={},{}
            for img_id in tqdm(dataset.image_ids):
                if isinstance(self.detector, SuperPointDetector) and isinstance(dataset, ScanNetDataset):
                    img,img_fn=dataset.get_image(img_id,True)
                    kps,desc=self.detector(img,img_fn)
                else:
                    img=dataset.get_image(img_id)
                    kps,desc=self.detector(img)
                if type(self.descriptor)!=DummyDescriptor:
                    desc=self.descriptor(img,kps)
                kps_dict[img_id]=kps.astype(np.float32)   # n,2
                desc_dict[img_id]=desc.astype(np.float32) # n,128

            save_h5(kps_dict,det_fn)
            save_h5(desc_dict,desc_fn)

        elif not os.path.exists(desc_fn) and os.path.exists(det_fn):
            print(f'extract {self.desc_name} desc on {self.det_name} kps ...')
            desc_dict={}
            kps_dict=load_h5(det_fn)
            for img_id in tqdm(dataset.image_ids):
                img=dataset.get_image(img_id)
                kps=kps_dict[img_id]
                desc=self.descriptor(img,kps)
                desc_dict[img_id]=desc.astype(np.float32) # n,128

            save_h5(desc_dict,desc_fn)

        else:
            print(f'{det_fn} and {desc_fn} all exist! skip it!')

    def match_desc(self,dataset:PoseSequenceDataset):
        det_fn=dataset.cache_fn(f'{self.det_name}')
        desc_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}')
        match_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}')
        if not os.path.exists(match_fn):
            match_dict={}
            kps_dict,desc_dict=load_h5(det_fn),load_h5(desc_fn)
            print(f'match by {self.matcher_name} ...')
            for pair_id in tqdm(dataset.pair_ids):
                id0,id1=pair_id
                K0=dataset.get_K(id0)
                K1=dataset.get_K(id1)
                desc0,desc1=desc_dict[id0],desc_dict[id1]
                kps0,kps1=kps_dict[id0],kps_dict[id1]
                img0,img1=dataset.get_image(id0),dataset.get_image(id1)
                matches=self.matcher.match(desc0,desc1,kps0,kps1,img0,img1,K0,K1)
                match_dict['-'.join(pair_id)]=matches
            save_h5(match_dict,match_fn)
        else:
            print(f'{match_fn} exists! skip it!')

    def filter_matches(self,dataset:PoseSequenceDataset,dataset_name):
        if self.filter is None:
            print('there is no matches filter, skip it ...')
            return

        det_fn=dataset.cache_fn(f'{self.det_name}')
        desc_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}')
        match_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}')
        filter_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}-{self.filter_name}')
        if not os.path.exists(filter_fn):
            filter_dict={}
            kps_dict,match_dict,desc_dict=load_h5(det_fn),load_h5(match_fn),load_h5(desc_fn)
            print(f'filter by {self.filter_name} ...')
            for pair_id in tqdm(dataset.pair_ids):
                id0,id1=pair_id
                kps0,kps1=kps_dict[id0],kps_dict[id1]
                matches=match_dict['-'.join(pair_id)] # [:,:2].astype(np.int32)
                desc0,desc1=desc_dict[id0],desc_dict[id1]
                img0,img1=dataset.get_image(id0),dataset.get_image(id1)
                K0,K1=dataset.get_K(id0),dataset.get_K(id1)
                # dataset_name-seq_name-id0-id1-det_name-desc_name-match_name
                filter_cur_name = f'{dataset_name}-{dataset.seq_name}-{id0}-{id1}-' \
                                  f'{self.det_name}-{self.desc_name}-{self.matcher_name}'
                mask = self.filter(kps0, kps1, matches, img0, img1, filter_cur_name, K0, K1, desc0, desc1,
                                   det_name=self.det_name, match_name=self.matcher_name)
                filter_dict['-'.join(pair_id)]=mask

            save_h5(filter_dict,filter_fn)
        else:
            print(f'{filter_fn} exists! skip it!')

    def estimate_geom(self,dataset):
        det_fn=dataset.cache_fn(f'{self.det_name}')
        match_fn=dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}')
        filter_fn = dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}-{self.filter_name}')
        geom_fn=self.get_geom_fn(dataset)
        if not os.path.exists(geom_fn):
            match_dict=load_h5(match_fn)
            kps_dict=load_h5(det_fn)
            if self.filter is not None:
                mask_dict=load_h5(filter_fn)
            geom_dict={}
            print(f'geometry estimated by {self.estimator_name} ...')
            for pair_id in tqdm(dataset.pair_ids):
                id0,id1=pair_id
                pair_id_str='-'.join(pair_id)
                img0,img1=dataset.get_image(id0),dataset.get_image(id1)
                K0=dataset.get_K(id0)
                K1=dataset.get_K(id1)
                kps0,kps1=kps_dict[id0][:,:2],kps_dict[id1][:,:2]
                matches=match_dict[pair_id_str][:,:2].astype(np.int32)
                if self.filter is not None:
                    matches=matches[mask_dict[pair_id_str]]

                if matches.shape[0]<=8:
                    R, t = np.identity(3), np.asarray([1,0,0])[:,None]
                else:
                    pts0, pts1=np.ascontiguousarray(kps0[matches[:, 0]]), np.ascontiguousarray(kps1[matches[:, 1]])
                    _,R,t=self.estimator.pose_estimate(pts0,pts1,K0,K1,img0,img1)
                geom_dict[pair_id_str]=np.concatenate([R,t],1).astype(np.float32)
            save_h5(geom_dict,geom_fn)
        else:
            print(f'{geom_fn} exists! skip it!')

    def eval_pose(self,dataset):
        geom_fn=self.get_geom_fn(dataset)
        error_fn=f'{geom_fn}-error'
        if not os.path.exists(error_fn):
            geom_dict=load_h5(geom_fn)
            error_dict={}
            print(f'compute error for {self.eval_name} ...')
            for pair_id in tqdm(dataset.pair_ids):
                id0,id1=pair_id
                pair_id_str='-'.join(pair_id)
                R_gt, t_gt=dataset.get_pose(id0,id1)
                Rt_pr = geom_dict[pair_id_str]
                R_pr, t_pr = Rt_pr[:,:3], Rt_pr[:,3]
                R_err, t_err = evaluate_R_t(R_gt, t_gt, R_pr, t_pr)
                # R_err, t_err = evaluate_R_t_v2(R_gt, t_gt, R_pr, t_pr)
                error_dict[pair_id_str]=np.asarray([R_err,t_err],np.float32)

            save_h5(error_dict,error_fn)
        else:
            print(f'{error_fn} exists! skip it!')

    def compute_metrics(self,dataset:PoseSequenceDataset):
        # load data
        geom_fn=self.get_geom_fn(dataset)
        error_fn=f'{geom_fn}-error'
        error_dict=load_h5(error_fn)

        det_fn = dataset.cache_fn(f'{self.det_name}')
        kps_dict = load_h5(det_fn)

        match_fn = dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}')
        match_dict = load_h5(match_fn)

        if self.filter is not None:
            filter_fn = dataset.cache_fn(f'{self.det_name}-{self.desc_name}-'
                                         f'{self.matcher_name}-{self.filter_name}')
            mask_dict = load_h5(filter_fn)

        # name2error={}
        # if bug_eval:
        #     for pair_id in dataset.pair_ids:
        #         pair_id_str = '-'.join(pair_id)
        #         cur_error = error_dict[pair_id_str]
        #         id0, id1 = pair_id
        #         fn0=dataset.img_id_to_fn[int(id0)]
        #         fn1=dataset.img_id_to_fn[int(id1)]
        #         stem0=os.path.basename(fn0)[:-4]
        #         stem1=os.path.basename(fn1)[:-4]
        #         name2error['-'.join([stem0,stem1])]=cur_error

        # compute metrics
        error_array,pr_re_f1_array=[],[]
        for pair_id in dataset.pair_ids:
            pair_id_str = '-'.join(pair_id)
            id0, id1 = pair_id
            # if bug_eval:
            #     fn0=dataset.img_id_to_fn[int(id0)]
            #     fn1=dataset.img_id_to_fn[int(id1)]
            #     stem0=os.path.basename(fn0)[:-4]
            #     stem1=os.path.basename(fn1)[:-4]
            #     error_array.append(name2error['-'.join([stem0,stem1])])
            error_array.append(error_dict[pair_id_str])

            # compute precision recall
            if self.filter is not None:
                kps0,kps1=kps_dict[id0][:,:2],kps_dict[id1][:,:2]
                matches=match_dict[pair_id_str][:,:2].astype(np.int32)
                pr=mask_dict[pair_id_str]

                gt=dataset.get_mask_gt(id0,id1,kps0,kps1,matches,5)
                pr,re,f1=compute_precision_recall_np(pr,gt)
                pr_re_f1_array.append(np.asarray([pr,re,f1]))

        error_array=np.asarray(error_array) # n,2
        if self.filter is not None:
            pr_re_f1_array = np.asarray(pr_re_f1_array)
        return error_array, pr_re_f1_array

    def eval_dataset(self,dataset,dataset_name):
        self.extract_kps_desc(dataset)
        self.match_desc(dataset)
        self.filter_matches(dataset,dataset_name)
        self.estimate_geom(dataset)
        self.eval_pose(dataset)
        error_array, pr_re_f1_array=self.compute_metrics(dataset)
        return error_array, pr_re_f1_array

    def __call__(self, dataset_name):
        rt_array_all, prf_array_all = [], []
        dataset_list = name2datalist(dataset_name)
        for dataset in dataset_list:
            rt_array, prf_array = self.eval_dataset(dataset,dataset_name)
            rt_array, prf_array = np.asarray(rt_array), np.asarray(prf_array)
            rt_array_all.append(rt_array)
            if self.filter is not None:
                prf_array_all.append(prf_array)

        rt_array_all=np.concatenate(rt_array_all,0)
        if self.filter is not None:
            prf_array_all=np.concatenate(prf_array_all,0)

        msg=pose_metrics(rt_array_all,prf_array_all)
        print(msg)
        with open('data/pose_results.log','a') as f:
            f.write(f'{self.eval_name:20} {dataset_name:20} {msg}\n')

    def compute_precision_recall(self,dataset_name):
        assert(self.filter is not None)
        dataset_list=name2datalist(dataset_name)
        results=[]
        for dataset in dataset_list:
            self.extract_kps_desc(dataset)
            self.match_desc(dataset)
            self.filter_matches(dataset, dataset_name, False)

            det_fn = dataset.cache_fn(f'{self.det_name}')
            kps_dict = load_h5(det_fn)
            match_fn = dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}')
            match_dict = load_h5(match_fn)
            filter_fn = dataset.cache_fn(f'{self.det_name}-{self.desc_name}-{self.matcher_name}-{self.filter_name}')
            mask_dict = load_h5(filter_fn)
            print(f'compute precision recall for {dataset.seq_name}')
            for pair_id in tqdm(dataset.pair_ids):
                id0,id1 = pair_id
                kps0,kps1=kps_dict[id0][:,:2],kps_dict[id1][:,:2]
                matches=match_dict['-'.join(pair_id)][:,:2].astype(np.int32)
                pr=mask_dict['-'.join(pair_id)]
                gt=dataset.get_mask_gt(id0,id1,kps0,kps1,matches,5)
                pr,re,f1=compute_precision_recall_np(pr,gt)
                results.append((pr,re,f1))
        results=np.asarray(results)
        print(f'{self.eval_name} precision {np.mean(results[:,0])} recall {np.mean(results[:,1])} f1 {np.mean(results[:,2])}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/sun3d/sift_baseline.yaml')
    parser.add_argument('--name', type=str, default='sun3d')
    flags = parser.parse_args()

    evaluator=PoseEvaluator(flags.cfg)
    evaluator(flags.name)


