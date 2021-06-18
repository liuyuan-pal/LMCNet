import time

import torch
import numpy as np
from tqdm import tqdm

from metrics import compute_map
from train.train_tools import to_cuda

def compute_Rt_auc_20(results):
    return compute_map(results['Rt_err'],20,1)

def compute_Rt_precision_5(results):
    return np.mean(results['Rt_err']<5)

def compute_f1(results):
    return np.mean(results['f1'])

def compute_precision(results):
    return np.mean(results['precision'])

def compute_Rt_auc_20_oanet(results):
    ths = np.arange(7) * 5
    qt_acc_hist, _ = np.histogram(results['Rt_err'], ths) # use the larger error
    num_pair = float(len(results['Rt_err']))
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    qt_acc = np.cumsum(qt_acc_hist) #
    return np.mean(qt_acc[:4])

def compute_residual(results):
    return 1.0-np.mean(results['loss_res_residuals'])

def spg_loss(results):
    return 1.0-np.mean(results['loss_cls_logits'])

name2keymetric={
    'Rt_precision_5': compute_Rt_precision_5,
    'Rt_auc_20': compute_Rt_auc_20,
    'f1': compute_f1,
    'oanet_Rt_auc_20': compute_Rt_auc_20_oanet,
    'res': compute_residual,
    'spg_loss': spg_loss,
    'precision': compute_precision
}

class ValidationEvaluator:
    def __init__(self,cfg):
        self.key_metric_name=cfg['key_metric_name']
        self.key_metric=name2keymetric[self.key_metric_name]

    def __call__(self, model, losses, eval_dataset):
        model.eval()
        eval_results={}
        begin=time.time()
        for data in tqdm(eval_dataset):
            data = to_cuda(data)
            with torch.no_grad():
                outputs=model(data)
                for loss in losses:
                    loss_results=loss(outputs, data, -1)
                    for k,v in loss_results.items():
                        if type(v)==torch.Tensor:
                            v=v.detach().cpu().numpy()

                        if k in eval_results:
                            eval_results[k].append(v)
                        else:
                            eval_results[k]=[v]

        for k,v in eval_results.items():
            eval_results[k]=np.concatenate(v,axis=0)

        key_metric_val=self.key_metric(eval_results)
        eval_results[self.key_metric_name]=key_metric_val
        print('eval cost {} s'.format(time.time()-begin))
        return eval_results, key_metric_val
