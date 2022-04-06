import os
import random
from abc import ABC, abstractmethod
from glob import glob

import cv2
import numpy as np
from skimage.io import imread

from utils.base_utils import compute_dR_dt, load_h5, read_pickle, save_pickle, np_skew_symmetric, epipolar_distance_mean


class PoseSequenceDataset(ABC):
    def __init__(self):
        self.cache_dir=''
        self.seq_name=''
        self.image_ids=[]
        self.pair_ids=[]

    @abstractmethod
    def get_image(self,img_id):
        return np.zeros([0,0,3],np.uint8)

    @abstractmethod
    def get_K(self,img_id):
        return np.zeros([3,3],np.float32)

    @abstractmethod
    def get_pose(self,id0,id1):
        return np.zeros([3,3],np.float32), np.zeros([3],np.float32)

    def cache_fn(self,suffix):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        return f'{self.cache_dir}/{self.seq_name}-{suffix}.h5'

    def get_F(self,id0,id1):
        R, t = self.get_pose(id0, id1)
        K0, K1 = self.get_K(id0), self.get_K(id1)
        F = np.linalg.inv(K1).T @ np_skew_symmetric(t) @ R @ np.linalg.inv(K0)
        return F

    def get_mask_gt(self,id0,id1,kps0,kps1,matches,thresh):
        K0,K1=self.get_K(id0),self.get_K(id1)
        R,t=self.get_pose(id0,id1)
        F = np.linalg.inv(K1).T @ np_skew_symmetric(t) @ R @ np.linalg.inv(K0)
        dist=epipolar_distance_mean(kps0[matches[:,0]],kps1[matches[:,1]],F)
        gt=dist<thresh
        return gt

class OANetSplitDataset(PoseSequenceDataset):
    def __init__(self, seq_name, vis_thresh=50, dataset_type='test', random_seed=1234, prefix="yfcc"):
        super().__init__()
        if prefix=='yfcc':
            self.cache_dir='data/yfcc_eval_cache'
        elif prefix=='sun3d':
            self.cache_dir='data/sun3d_eval_cache'
        else:
            raise NotImplementedError

        if prefix=='yfcc':
            self.seq_name=f'{seq_name}-{dataset_type}'
        elif prefix=='sun3d':
            self.seq_name=f'{seq_name}-{dataset_type}'
        else:
            raise NotImplementedError

        self.vis_thresh=vis_thresh
        self.random_seed=random_seed
        self.dataset_type=dataset_type
        self.seq_name=seq_name

        if prefix=='yfcc':
            seq_dir=os.path.join('data','yfcc100m',seq_name,dataset_type)
        elif prefix=='sun3d':
            seq_dir=os.path.join('data',f'sun3d_{dataset_type}',seq_name,dataset_type)
        else:
            raise NotImplementedError

        img_list_file = os.path.join(seq_dir, "images.txt")
        geo_list_file = os.path.join(seq_dir, "calibration.txt")
        vis_list_file = os.path.join(seq_dir, "visibility.txt")

        self.img_pths=[os.path.join(seq_dir,pth) for pth in np.loadtxt(img_list_file,dtype=str)]
        self.geo_list=[os.path.join(seq_dir,pth) for pth in np.loadtxt(geo_list_file,dtype=str)]
        self.vis_list=[os.path.join(seq_dir,pth) for pth in np.loadtxt(vis_list_file,dtype=str)]

        self.pair_ids=read_pickle(f'data/pairs/{self.seq_name}-te-1000-pairs.pkl')
        self.pair_ids=[(str(pair[0]),str(pair[1])) for pair in self.pair_ids]

        unique_ids = set()
        for pair in self.pair_ids:
            unique_ids.add(pair[0])
            unique_ids.add(pair[1])
        self.image_ids = list(unique_ids)

    @staticmethod
    def rectify_K(geom):
        img_size, K = geom['imsize'][0], geom['K']
        if (type(img_size)==tuple or type(img_size)==list or type(img_size)==np.ndarray) and len(img_size)==2:
            w, h = img_size[0], img_size[1]
        else:
            h=w=img_size

        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5
        K[0, 2] += cx
        K[1, 2] += cy
        return K

    def get_image(self,img_id,grey_model=False):
        if grey_model:
            img=cv2.imread(self.img_pths[int(img_id)],cv2.IMREAD_GRAYSCALE)
        else:
            img=imread(self.img_pths[int(img_id)])
        return img

    def get_K(self,img_id):
        geo = load_h5(self.geo_list[int(img_id)])
        K=self.rectify_K(geo)
        return K

    def get_pose_single(self, img_id):
        geo = load_h5(self.geo_list[int(img_id)])
        R, t = geo['R'], geo['T'][0]
        return R, t

    def get_pose(self,id0,id1):
        geo0 = load_h5(self.geo_list[int(id0)])
        geo1 = load_h5(self.geo_list[int(id1)])
        R0, t0 = geo0['R'], geo0['T'][0]
        R1, t1 = geo1['R'], geo1['T'][0]
        dR, dt = compute_dR_dt(R0,t0,R1,t1)
        return dR, dt


class OANetTrainDataset(PoseSequenceDataset):
    def __init__(self, seq_name, vis_thresh=50, pair_num=3000,random_seed=1234, prefix="yfcc"):
        super().__init__()
        if prefix=='yfcc':
            self.cache_dir='data/yfcc_train_cache'
        elif prefix=='sun3d':
            self.cache_dir='data/sun3d_train_cache'
        else:
            raise NotImplementedError

        self.vis_thresh=vis_thresh
        self.pair_num=pair_num
        self.random_seed=random_seed
        self.seq_name=seq_name

        if prefix=='yfcc':
            seq_dir=os.path.join('data','yfcc100m', seq_name, 'train')
        elif prefix=='sun3d':
            seq_dir=os.path.join('data','sun3d_train', seq_name, 'train')
        else:
            raise NotImplementedError

        img_list_file = os.path.join(seq_dir, "images.txt")
        geo_list_file = os.path.join(seq_dir, "calibration.txt")
        vis_list_file = os.path.join(seq_dir, "visibility.txt")

        self.img_pths=[os.path.join(seq_dir,pth) for pth in np.loadtxt(img_list_file,dtype=str)]
        self.geo_list=[os.path.join(seq_dir,pth) for pth in np.loadtxt(geo_list_file,dtype=str)]
        self.vis_list=[os.path.join(seq_dir,pth) for pth in np.loadtxt(vis_list_file,dtype=str)]

        self.image_ids=[str(idx) for idx in list(range(len(self.img_pths)))]
        self.pair_ids=self.pair_selection()

        unique_ids = set()
        for pair in self.pair_ids:
            unique_ids.add(pair[0])
            unique_ids.add(pair[1])
        self.image_ids = list(unique_ids)

    def pair_selection(self):
        pairs=[]
        img_num=len(self.image_ids)
        for i in range(img_num):
            vis=np.loadtxt(self.vis_list[i]).flatten().astype("float32")
            for j in range(i+1,img_num):
                if vis[j]>self.vis_thresh:
                    pairs.append((str(i),str(j)))

        np.random.seed(self.random_seed)
        pairs = [pairs[i] for i in np.random.permutation(len(pairs))[:self.pair_num]]
        return pairs

    @staticmethod
    def rectify_K(geom):
        img_size, K = geom['imsize'][0], geom['K']
        if (type(img_size)==tuple or type(img_size)==list or type(img_size)==np.ndarray) and len(img_size)==2:
            w, h = img_size[0], img_size[1]
        else:
            h=w=img_size

        cx = (w - 1.0) * 0.5
        cy = (h - 1.0) * 0.5
        K[0, 2] += cx
        K[1, 2] += cy
        return K

    def get_image(self,img_id,grey_model=False):
        if grey_model:
            img=cv2.imread(self.img_pths[int(img_id)],cv2.IMREAD_GRAYSCALE)
        else:
            img=imread(self.img_pths[int(img_id)])
        return img

    def get_K(self,img_id):
        geo = load_h5(self.geo_list[int(img_id)])
        K=self.rectify_K(geo)
        return K

    def get_pose_single(self, img_id):
        geo = load_h5(self.geo_list[int(img_id)])
        R, t = geo['R'], geo['T'][0]
        return R, t

    def get_pose(self,id0,id1):
        geo0 = load_h5(self.geo_list[int(id0)])
        geo1 = load_h5(self.geo_list[int(id1)])
        R0, t0 = geo0['R'], geo0['T'][0]
        R1, t1 = geo1['R'], geo1['T'][0]
        dR, dt = compute_dR_dt(R0,t0,R1,t1)
        return dR, dt

class ScanNetDataset(PoseSequenceDataset):
    @staticmethod
    def parse_gt():
        def process_fn(fn):
            fn_split = fn.split('/')
            fn_stem = fn_split[-1]
            fn_split[-1] = str(int(fn_stem[len('frame-'):len('frame-') + 6])) + '.jpg'
            fn_split[-2] = 'rgb'
            return '/'.join(fn_split)

        def parse_img_data(fn,K,img_data,img_id_to_fn):
            if fn in img_id_to_fn:
                fi=img_id_to_fn.index(fn)
            else:
                fi=len(img_id_to_fn)
                img_id_to_fn.append(fn)
                img_data[fn]=K
            return fi

        img_data,pair_data,img_id_to_fn={},{},[]
        with open('data/scannet_test_pairs_with_gt.txt', 'r') as f:
            for pair_id, pair in enumerate(f.readlines()):
                pair = pair.split(' ')
                fn0 = process_fn(pair[0])
                fn1 = process_fn(pair[1])
                K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
                K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
                T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)
                R = T_0to1[:3, :3]
                t = T_0to1[:3, 3:]

                img_id0=parse_img_data(fn0,K0,img_data,img_id_to_fn)
                img_id1=parse_img_data(fn1,K1,img_data,img_id_to_fn)
                pair_data['-'.join([str(img_id0),str(img_id1)])]=(R,t.flatten())
        return img_data, pair_data, img_id_to_fn

    def __init__(self):
        super().__init__()
        self.cache_dir='data/scannet_eval_cache'
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.seq_name='0'
        self.img_fn_to_K, self.pair_id_to_pose, self.img_id_to_fn=self.parse_gt()
        self.image_ids=[str(k) for k in range(len(self.img_id_to_fn))]
        self.pair_ids=[k.split('-') for k in self.pair_id_to_pose.keys()]

    def get_image(self, img_id, get_fn=False):
        fn = f'data/scannet_dataset/' + self.img_id_to_fn[int(img_id)]
        if get_fn:
            return imread(fn), fn
        else:
            return imread(fn)

    def get_K(self, img_id):
        fn=self.img_id_to_fn[int(img_id)]
        return self.img_fn_to_K[fn]

    def get_pose(self, id0, id1):
        return self.pair_id_to_pose['-'.join([id0,id1])]

class ScanNetTrainDataset(PoseSequenceDataset):
    K_default=np.asarray(
        [
            [1165.72, 0., 649.095],
            [0., 1165.74, 484.765],
            [0.,       0.,     1.],
        ],
        dtype=np.float32
    )
    def __init__(self,seq_name,seq_pair_num=None,min_interval=10,max_interval=20):
        super().__init__()
        self.root_dir=os.path.join(f'data/scannet_train_dataset/{seq_name}')
        self.image_fns=os.listdir(os.path.join(self.root_dir,'rgb'))
        img_id_int = [int(fn[:-4]) for fn in self.image_fns]
        idxs=np.argsort(img_id_int)
        self.image_fns=np.asarray(self.image_fns)[idxs]

        self.image_ids=[str(k) for k in range(len(self.image_fns))]

        self.stems = [self.image_fns[int(img_id)][:-4] for img_id in self.image_ids]
        self.pair_ids = [(self.image_ids[i],self.image_ids[j]) for i in range(len(self.image_ids))
                         for j in range(i+min_interval,min(i+max_interval,len(self.image_ids)))]
        if seq_pair_num is not None:
            random.seed(6033)
            random.shuffle(self.pair_ids)
            self.pair_ids = self.pair_ids[:seq_pair_num]

        h,w=self.get_image(self.image_ids[0]).shape[:2]
        if h==480:
            self.scaled=True
        else:
            self.scaled=False

    def get_image(self, img_id):
        return imread(os.path.join(self.root_dir,'rgb',self.image_fns[int(img_id)]))

    def get_K(self, img_id):
        # scale K
        if self.scaled:
            K=self.K_default.copy()
            scale_x = 640./1296.
            scale_y = 480./968.
            return np.diag(np.asarray([scale_x,scale_y,1.])) @ K
        else:
            return self.K_default

    def get_pose(self, id0, id1):
        Rt0=np.loadtxt(os.path.join(self.root_dir,'pose',self.stems[int(id0)]+'.txt'))
        Rt1=np.loadtxt(os.path.join(self.root_dir,'pose',self.stems[int(id1)]+'.txt'))
        # dR, dt = compute_dR_dt(Rt0[:3,:3],Rt0[:3,3],Rt1[:3,:3],Rt1[:3,3])
        dt = Rt1[:3,:3].T @ (Rt0[:3,3]-Rt1[:3,3])
        dR = Rt1[:3,:3].T @ Rt0[:3,:3]
        return dR, dt

    def get_pose_single(self, img_id):
        Rt=np.loadtxt(os.path.join(self.root_dir,'pose',self.stems[int(img_id)]+'.txt'))
        t=-Rt[:3,:3].T @ Rt[:3,3]
        return -Rt[:3,:3].T, t

def name2datalist(name):
    dataset_list=[]
    if name=='yfcc':
        seq_names = ['buckingham_palace', 'notre_dame_front_facade', 'reichstag', 'sacre_coeur']
        for seq_name in seq_names:
            dataset_list.append(OANetSplitDataset(seq_name, prefix='yfcc'))
        return dataset_list
    elif name=='sun3d':
        seq_names = ['te-brown1', 'te-brown2', 'te-brown3', 'te-brown4', 'te-brown5', 'te-hotel1', 'te-harvard1',
                     'te-harvard2', 'te-harvard3', 'te-harvard4','te-mit1', 'te-mit2', 'te-mit3', 'te-mit4', 'te-mit5']
        for seq_name in seq_names:
            dataset_list.append(OANetSplitDataset(seq_name, prefix='sun3d'))
        return dataset_list
    elif name=='scannet':
        return [ScanNetDataset()]
    else:
        raise NotImplementedError
