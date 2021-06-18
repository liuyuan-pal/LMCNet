import os

import torch
import numpy as np
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# from dataset.train_detrac_dataset import DETRACTrainDataset, TPSTrainDataset
from dataset.pose_train_dataset import RelPoseFeatsDataset, collate_fn
from network.loss import name2loss
from network.lmcnet import LMCNet
from train.train_metrics import name2metric
from train.train_tools import to_cuda, Logger, reset_learning_rate, MultiGPUWrapper, DummyLoss
from train.train_valid import ValidationEvaluator


class Trainer:
    def _init_dataset(self):
        if self.cfg['dataset_type']=='pose_feats':
            train_set = RelPoseFeatsDataset(
                self.cfg['dataset_root'],
                self.cfg['dataset_extract_name'],
                self.cfg['dataset_match_name'],
                self.cfg['train_pair_info_fn'],
                self.cfg['epipolar_inlier_thresh'],
                self.cfg['use_eig'],
                self.cfg['dataset_eig_name'],
                self.cfg['use_feats'],is_train=True
            )
            val_set = RelPoseFeatsDataset(
                self.cfg['dataset_root'],
                self.cfg['dataset_extract_name'],
                self.cfg['dataset_match_name'],
                self.cfg['val_pair_info_fn'],
                self.cfg['epipolar_inlier_thresh'],
                self.cfg['use_eig'],
                self.cfg['dataset_eig_name'],
                self.cfg['use_feats'],is_train=False
            )
        elif self.cfg['dataset_type']=='detrac':
            root_dir='data/detrac_train_cache' if 'root_dir' not in self.cfg else self.cfg['root_dir']
            train_set=DETRACTrainDataset(self.cfg['train_pair_info_fn'], root_dir,
                                         self.cfg['dataset_extract_name'],self.cfg['dataset_match_name'],
                                         self.cfg['use_eig'],self.cfg['eig_name'],True,True)
            val_set=DETRACTrainDataset(self.cfg['val_pair_info_fn'], root_dir,
                                       self.cfg['dataset_extract_name'],self.cfg['dataset_match_name'],
                                       self.cfg['use_eig'],self.cfg['eig_name'],False)
        else:
            raise NotImplementedError

        self.train_set = DataLoader(train_set, self.cfg['batch_size'], True,
                                    num_workers=16, pin_memory=False, collate_fn=collate_fn)
        self.val_set = DataLoader(val_set, self.cfg['batch_size'], False,
                                  num_workers=4, collate_fn=collate_fn)
        print(f'train set len {len(self.train_set)}')
        print(f'val set len {len(self.val_set)}')

    def _init_network(self):
        self.network=LMCNet(self.cfg).cuda()
        self.optimizer = Adam(self.network.parameters(), lr=1e-3)

        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metric:
                self.val_metrics.append(name2metric[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        if self.cfg['multi_gpus']:
            # make multi gpu network
            self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network=self.network
            self.train_losses=self.val_losses

        if 'finetune' in self.cfg and self.cfg['finetune']:
            checkpoint=torch.load(self.cfg['finetune_path'])
            self.network.load_state_dict(checkpoint['network_state_dict'])
            print(f'==> resuming from step {self.cfg["finetune_path"]}')
        self.val_evaluator=ValidationEvaluator(self.cfg)

    def __init__(self,cfg):
        self.cfg=cfg
        self.model_dir=os.path.join('data/model',cfg['name'])
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        best_para,start_step=self._load_model()
        train_iter=iter(self.train_set)

        pbar=tqdm(total=self.cfg['total_step'],bar_format='{r_bar}')
        pbar.update(start_step)
        for step in range(start_step,self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step']=step

            self.train_network.train()
            self.network.train()
            reset_learning_rate(self.optimizer,self._get_lr(step))
            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info={}
            outputs=self.train_network(train_data)
            for loss in self.train_losses:
                loss_results=loss(outputs,train_data,step)
                for k,v in loss_results.items():
                    log_info[k]=v

            loss=0
            for k,v in log_info.items():
                if k.startswith('loss'):
                    loss=loss+torch.mean(v)

            loss.backward()
            self.optimizer.step()
            if ((step+1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info,step+1,'train')

            if (step+1)%self.cfg['val_interval']==0:
                val_results, val_para=self.val_evaluator(self.network, self.val_losses + self.val_metrics, self.val_set)
                if val_para>best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para=val_para
                    # if self.cfg['save_inter_model'] and (step+1)%self.cfg['save_inter_interval']==0:
                    #     self._save_model(step + 1, best_para, os.path.join(self.model_dir,f'{step+1}.pth'))
                    self._save_model(step+1,best_para,self.best_pth_fn)
                self._log_data(val_results,step+1,'val')

            if (step+1)%self.cfg['save_interval']==0:
                # if self.cfg['save_inter_model'] and (step+1)%10000==0:
                #     self._save_model(step+1,best_para,f'{self.model_dir}/{step}.pth')
                self._save_model(step+1,best_para)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()))
            pbar.update(1)

        pbar.close()

    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.pth_fn):
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)

    def _get_lr(self,step):
        if 'lr_type' not in self.cfg or self.cfg['lr_type']=='default':
            if step<=self.cfg['lr_mid_epoch']:
                return self.cfg['lr_start']
            else:
                decay_rate=self.cfg['lr_decay_rate']
                decay_step=self.cfg['lr_decay_step']
                decay_num=(step-self.cfg['lr_mid_epoch'])//decay_step
                return max(self.cfg['lr_start']*decay_rate**decay_num,self.cfg['lr_min'])
        elif self.cfg['lr_type']=='warm_up':
            if step<=self.cfg['lr_warm_up_step']:
                return self.cfg['lr_warm_up']
            else:
                decay_rate=self.cfg['lr_decay_rate']
                decay_step=self.cfg['lr_decay_step']
                decay_num=(step-self.cfg['lr_warm_up_step'])//decay_step
                return max(self.cfg['lr_start']*decay_rate**decay_num,self.cfg['lr_min'])



