import numpy as np

sift_feats_stats={
    'desc_norm_mean': 512.0016479492188,
    'desc_norm_std': 0.5519750714302063,
    'size_ratio_log_mean': 0.0056634703651070595,
    'size_ratio_log_std': 0.6931794285774231,
    'resp_mean': 0.030322134494781494,
    'resp_std': 1.5606329441070557,
    'size_mean': 3.7023561000823975,
    'size_std': 3.3257734775543213,
}
sp2_feats_stats={
    'scores_mean': 0.06332188844680786,
    'scores_std': 0.12062419205904007,
    'sg_scores_mean': 0.3228559195995331,
    'sg_scores_std': 0.3630686104297638,
}

def get_sift_corr_feats(desc0,desc1,size0,size1,angle0,angle1,
                        resp0,resp1,corr_feats_stats=sift_feats_stats):
    desc0_norm=np.linalg.norm(desc0, 2, 1)
    desc1_norm=np.linalg.norm(desc1, 2, 1)
    desc0/=desc0_norm[:,None]
    desc1/=desc1_norm[:,None]
    desc_diff = desc1 - desc0
    size_ratio_log = np.log(size1+1e-3)-np.log(size0+1e-3)
    ang_diff = np.deg2rad(angle0-angle1)
    ang_diff[ang_diff<-np.pi]=ang_diff[ang_diff<-np.pi]+2*np.pi
    ang_diff[ang_diff>np.pi]=ang_diff[ang_diff>np.pi]-2*np.pi

    size_ratio_log=(size_ratio_log-corr_feats_stats['size_ratio_log_mean'])/corr_feats_stats['size_ratio_log_std']
    desc0_norm=(desc0_norm-corr_feats_stats['desc_norm_mean'])/corr_feats_stats['desc_norm_std']
    desc1_norm=(desc1_norm-corr_feats_stats['desc_norm_mean'])/corr_feats_stats['desc_norm_std']
    resp0=(resp0-corr_feats_stats['resp_mean'])/corr_feats_stats['resp_std']
    resp1=(resp1-corr_feats_stats['resp_mean'])/corr_feats_stats['resp_std']
    size0=(size0-corr_feats_stats['size_mean'])/corr_feats_stats['size_std']
    size1=(size1-corr_feats_stats['size_mean'])/corr_feats_stats['size_std']

    image_feats=np.stack([resp0,resp1,desc0_norm,desc1_norm,size0,size1],1)
    image_feats=np.clip(image_feats,-5,5)
    image_feats=np.concatenate([image_feats,desc_diff],1)
    geom_feats=np.stack([ang_diff,np.clip(size_ratio_log,-5,5)],1)
    return image_feats, geom_feats