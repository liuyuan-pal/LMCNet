import argparse

import numpy as np
import os

from skimage.io import imsave
from tqdm import tqdm

from dataset.pose_dataset import ScanNetTrainDataset, OANetTrainDataset
from detector import name2det
from matcher import name2matcher
from utils.base_utils import load_component, get_stem, hpts_to_pts, pts_to_hpts, save_pickle, read_pickle
# from utils.draw_utils import draw_hist, draw_correspondence, get_colors_gt_pr
from utils.eig_utils import build_graph, name2config


def make_non_existed_dir(dir):
    if not os.path.exists(dir): os.mkdir(dir)

class TrainSetGenerator:
    def __init__(self, ext_fn, match_fn, eig_name, output_dir, prefix='yfcc', begin=-1, end=-1):
        self.extractor=load_component(name2det,ext_fn)
        self.matcher=load_component(name2matcher,match_fn)
        self.ext_name=get_stem(ext_fn)
        self.matcher_name=get_stem(match_fn)
        self.eig_name=eig_name
        self.output_dir=output_dir
        make_non_existed_dir(f'{self.output_dir}/{self.ext_name}')
        make_non_existed_dir(f'{self.output_dir}/{self.ext_name}/{self.matcher_name}')
        make_non_existed_dir(f'{self.output_dir}/{self.ext_name}/{self.matcher_name}/{self.eig_name}')

        self.prefix=prefix
        if prefix=='yfcc':
            with open('data/yfcc_train.txt', 'r') as f:
                lines = f.readlines()
                self.train_seq_names = [line.strip() for line in lines]
            self.seq_pair_num=8000
        elif prefix=='sun3d':
            with open('data/sun3d_train.txt', 'r') as f:
                lines = f.readlines()
                self.train_seq_names = ['-'.join(line.strip()[:-1].split('/')) for line in lines]
            self.seq_pair_num=2000
        elif prefix=='scannet':
            with open('data/scannet_train_seq_names_used.txt', 'r') as f:
                self.train_seq_names = [line.strip() for line in f.readlines()]
            if begin!=-1 and end!=-1:
                self.train_seq_names = self.train_seq_names[begin:end]
            else:
                self.train_seq_names = self.train_seq_names
            self.seq_pair_num = 2000
        else:
            raise NotImplementedError

    def get_dataset(self,seq_name):
        if self.prefix=='yfcc':
            return OANetTrainDataset(seq_name, 50, self.seq_pair_num, 1234, 'yfcc')
        elif self.prefix=='sun3d':
            return OANetTrainDataset(seq_name, 0.35, self.seq_pair_num, 1234, 'sun3d')
        elif self.prefix=='scannet':
            return ScanNetTrainDataset(seq_name, self.seq_pair_num)
        else:
            raise NotImplementedError

    def extract(self):
        for seq_name in self.train_seq_names:
            import ipdb; ipdb.set_trace()
            train_dataset=self.get_dataset(seq_name)
            make_non_existed_dir(f'{self.output_dir}/{self.ext_name}/{seq_name}')
            print(f'{self.ext_name} extract kps and desc for {seq_name} ...')
            for img_id in tqdm(train_dataset.image_ids):
                npy_fn=f'{self.output_dir}/{self.ext_name}/{seq_name}/{img_id}.npz'
                if os.path.exists(npy_fn): continue
                img=train_dataset.get_image(img_id)
                kps,desc=self.extractor(img)

                K=train_dataset.get_K(img_id)
                R,t=train_dataset.get_pose_single(img_id)
                np.savez_compressed(npy_fn, kps=kps.astype(np.float32),desc=desc.astype(np.float32),K=K,R=R,t=t)

    def revise_K(self):
        for seq_name in self.train_seq_names:
            train_dataset=self.get_dataset(seq_name)
            make_non_existed_dir(f'{self.output_dir}/{self.ext_name}/{seq_name}')
            print(f'{self.ext_name} extract kps and desc for {seq_name} ...')
            for img_id in tqdm(train_dataset.image_ids):
                npy_fn=f'{self.output_dir}/{self.ext_name}/{seq_name}/{img_id}.npz'
                data=np.load(npy_fn)
                kps=data['kps']
                desc=data['desc']
                K=train_dataset.get_K(img_id)
                R,t=train_dataset.get_pose_single(img_id)
                np.savez_compressed(npy_fn, kps=kps.astype(np.float32),desc=desc.astype(np.float32),K=K,R=R,t=t)

    @staticmethod
    def normalize_desc(desc):
        desc /= np.linalg.norm(desc, 1)
        desc[desc>0.2]=0.2
        desc /= np.linalg.norm(desc, 1)
        return desc

    def match(self):
        for seq_name in self.train_seq_names:
            train_dataset=self.get_dataset(seq_name)
            make_non_existed_dir(f'{self.output_dir}/{self.ext_name}/{self.matcher_name}/{seq_name}')
            print(f'{self.matcher_name} match for {seq_name} ...')
            for pair_id in tqdm(train_dataset.pair_ids):
                id0, id1=pair_id
                npy_fn=f'{self.output_dir}/{self.ext_name}/{self.matcher_name}/{seq_name}/{id0}-{id1}.npy'
                if os.path.exists(npy_fn): continue
                if np.sum(np.isinf(train_dataset.get_pose_single(id0)[0]))>0 or \
                    np.sum(np.isinf(train_dataset.get_pose_single(id1)[0])) > 0:
                    continue
                data0=np.load(f'{self.output_dir}/{self.ext_name}/{seq_name}/{id0}.npz')
                data1=np.load(f'{self.output_dir}/{self.ext_name}/{seq_name}/{id1}.npz')
                desc0,kps0=data0['desc'],data0['kps']
                desc1,kps1=data1['desc'],data1['kps']
                img0=train_dataset.get_image(id0)
                img1=train_dataset.get_image(id1)

                if self.ext_name.startswith('sift'):
                    # normalize desc
                    desc0=self.normalize_desc(desc0)
                    desc1=self.normalize_desc(desc1)

                matches = self.matcher.match(desc0,desc1,kps0,kps1,img0,img1)
                if self.matcher_name.startswith('superglue'):
                    np.save(npy_fn, matches.astype(np.float32))
                else:
                    np.save(npy_fn, matches.astype(np.uint16))

    def compute_eig(self,):
        for seq_name in self.train_seq_names:
            train_dataset=self.get_dataset(seq_name)
            make_non_existed_dir(f'{self.output_dir}/{self.ext_name}/{self.matcher_name}/{self.eig_name}/{seq_name}')
            print(f'eig {self.eig_name} for {seq_name} ...')
            for pair_id in tqdm(train_dataset.pair_ids):
                id0, id1=pair_id
                eig_fn=os.path.join(f'{self.output_dir}/{self.ext_name}/{self.matcher_name}/{self.eig_name}/{seq_name}/{id0}-{id1}.npy')
                if os.path.exists(eig_fn): continue
                if np.sum(np.isinf(train_dataset.get_pose_single(id0)[0])) > 0 or \
                    np.sum(np.isinf(train_dataset.get_pose_single(id1)[0])) > 0:
                    continue
                matches=np.load(f'{self.output_dir}/{self.ext_name}/{self.matcher_name}/{seq_name}/{id0}-{id1}.npy')
                matches=matches[:,:2].astype(np.int32)
                data0=np.load(f'{self.output_dir}/{self.ext_name}/{seq_name}/{id0}.npz')
                data1=np.load(f'{self.output_dir}/{self.ext_name}/{seq_name}/{id1}.npz')
                kps0,K0=data0['kps'][:,:2],train_dataset.get_K(id0)
                kps1,K1=data1['kps'][:,:2],train_dataset.get_K(id1)
                x0 = hpts_to_pts(pts_to_hpts(kps0[matches[:, 0]]) @ np.linalg.inv(K0).T)
                x1 = hpts_to_pts(pts_to_hpts(kps1[matches[:, 1]]) @ np.linalg.inv(K1).T)
                xs = np.concatenate([x0,x1],1)

                eig_val, eig_vec = build_graph(xs,name2config[self.eig_name])
                data = np.concatenate([eig_val[None, :], eig_vec], 0).astype(np.float32)
                np.save(eig_fn, data)

    def save_train_val_pair_info(self,val_num=1000):
        all_pair_infos=[]
        for seq_name in tqdm(self.train_seq_names):
            train_dataset=self.get_dataset(seq_name)
            for pair_id in train_dataset.pair_ids:
                id0, id1 = pair_id
                # R0=np.load(f'{self.output_dir}/{self.ext_name}/{seq_name}/{id0}.npz')['R']
                # R1=np.load(f'{self.output_dir}/{self.ext_name}/{seq_name}/{id1}.npz')['R']
                R0=train_dataset.get_pose_single(id0)[0]
                R1=train_dataset.get_pose_single(id1)[0]
                if np.sum(np.isinf(R0))>0 or np.sum(np.isinf(R1))>0:
                    continue
                all_pair_infos.append((seq_name,id0,id1))

        print(f'pair num {len(all_pair_infos)}')
        np.random.seed(1234)
        np.random.shuffle(all_pair_infos)
        save_pickle(all_pair_infos[:val_num],f'{self.output_dir}/val-{self.seq_pair_num}.pkl')
        save_pickle(all_pair_infos[val_num:],f'{self.output_dir}/train-{self.seq_pair_num}.pkl')

    def save_train_val_pair_info_v2(self,val_num=4000):
        all_pair_infos=[]
        for seq_name in self.train_seq_names:
            train_dataset=self.get_dataset(seq_name)
            for pair_id in train_dataset.pair_ids:
                id0, id1 = pair_id
                all_pair_infos.append((seq_name,id0,id1))

        test_pair_infos=read_pickle(f'{self.output_dir}/val.pkl')
        train_pair_infos=[]
        for pair_info in tqdm(all_pair_infos):
            if pair_info not in test_pair_infos:
                train_pair_infos.append(pair_info)

        print(len(all_pair_infos))
        print(len(train_pair_infos))
        print(len(test_pair_infos))
        save_pickle(train_pair_infos,f'{self.output_dir}/train-{self.seq_pair_num}.pkl')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext_cfg', type=str, default='configs/detector/sift.yaml')
    parser.add_argument('--match_cfg', type=str, default='configs/matcher/nn.yaml')
    parser.add_argument('--output', type=str, default='data/yfcc_train_cache')
    parser.add_argument('--eig_name', type=str, default='small_min')
    parser.add_argument('--prefix', type=str, default='yfcc')
    parser.add_argument('--begin', type=int, default=-1)
    parser.add_argument('--end', type=int, default=-1)
    flags = parser.parse_args()
    if not os.path.exists(flags.output): os.mkdir(flags.output)
    generator=TrainSetGenerator(flags.ext_cfg, flags.match_cfg, flags.eig_name, flags.output, flags.prefix, flags.begin, flags.end)
    print(f'begin {flags.begin} end {flags.end}!')
    generator.extract()
    generator.match()
    generator.compute_eig()
    generator.save_train_val_pair_info(2000)
