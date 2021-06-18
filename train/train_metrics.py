import cv2

from network.ops import weighted_8points, get_knn_feats
import torch
import numpy as np
from metrics import evaluate_R_t


def decompose_RT_from_E(p1s, p2s, E_hat, scores):
    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask].astype(np.float64)
    p2s_good = p2s[mask].astype(np.float64)

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(np.float64)
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(E_hat, p1s_good, p2s_good)
    else:
        R, t = np.identity(3), np.ones(3)
    return R, t

class RelPoseMetrics:
    def __init__(self,cfg):
        pass

    def __call__(self, network_outputs, data_inputs, step):
        logits=network_outputs['logits'] # b,n
        R=data_inputs['R'] # b,3,3
        t=data_inputs['t'] # b,3
        xs=data_inputs['xs'] # b,n,4

        with torch.no_grad():
            E_pr=weighted_8points(xs.unsqueeze(1),logits) # b,3,3

        E_pr=E_pr.detach().cpu().numpy()
        R=R.cpu().numpy()
        t=t.cpu().numpy()
        xs=xs.cpu().numpy()
        logits=logits.detach().cpu().numpy()
        b,_=logits.shape
        R_errs,t_errs=[],[]
        # E_pr=E_pr.reshape(b,3,3)
        for bi in range(b):
            R_pr, t_pr = decompose_RT_from_E(xs[bi,:,:2],xs[bi,:,2:],E_pr[bi],logits[bi])
            # R_err,t_err=ang_diff_Rt(R[bi],t[bi],R_pr,t_pr)
            R_err, t_err=evaluate_R_t(R[bi],t[bi],R_pr,t_pr)
            R_errs.append(R_err)
            t_errs.append(t_err)

        R_errs,t_errs=np.asarray(R_errs),np.asarray(t_errs)
        Rt_errs=np.max(np.stack([R_errs,t_errs],axis=1),1)
        return {'R_err':R_errs,'t_err':t_errs,'Rt_err':Rt_errs}

name2metric={
    'pose':RelPoseMetrics,
}