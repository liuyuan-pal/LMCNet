import numpy as np
import cv2

from others.superpoint import process_image
from utils.base_utils import pts_to_hpts, hpts_to_pts

def normalize_coordinates_camera(pts1, pts2, K1, K2):
    pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32)
    pts_l_norm = hpts_to_pts(pts_to_hpts(pts1) @ np.linalg.inv(K1).T)
    pts_r_norm = hpts_to_pts(pts_to_hpts(pts2) @ np.linalg.inv(K2).T)
    return pts_l_norm, pts_r_norm

class RANSACEstimator:
    def __init__(self,cfg):
        self.configs=cfg

    def pose_estimate(self, pts1, pts2, K1, K2, *args, **kwargs):
        # f_avg = (K1[0, 0] + K2[0, 0]) / 2
        pts_l_norm, pts_r_norm = normalize_coordinates_camera(pts1, pts2, K1, K2)
        E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
                                       method=cv2.RANSAC, prob=self.configs['confidence'],
                                       threshold=self.configs['thresh'])
        points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
        return mask[:,0].astype(np.bool), R_est, t_est

    def fundamental_matrix_estimate(self, pts1, pts2):
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=self.configs['thresh'],
                                         confidence=self.configs['confidence'])
        return mask[:,0].astype(np.bool), F

    def homography_estimation(self,pts1,pts2):
        H, mask = cv2.findHomography(pts1[:,None,:2],pts2[:,None,:2],method=cv2.RANSAC,ransacReprojThreshold=self.configs['thresh'])
        return H

class RescaleRANSACEstimator:
    @staticmethod
    def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
        if len(kpts0) < 5:
            return None

        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        norm_thresh = thresh / f_mean

        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
            method=cv2.RANSAC)

        if E is None:
            import ipdb; ipdb.set_trace()
        assert E is not None

        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
        return ret

    def __init__(self,cfg):
        self.resize=cfg['resize']
        self.resize_float=False if 'resize_float' in cfg and not cfg['resize_float'] else True
        self.round=True if 'round' in cfg and cfg['round'] else False

    def pose_estimate(self, pts1, pts2, K1, K2, image1, image2):
        pts1, pts2, K1, K2 = np.copy(pts1), np.copy(pts2), np.copy(K1), np.copy(K2)
        resize = (self.resize,) if isinstance(self.resize, int) else self.resize
        _,_,scales=process_image(image1,resize=resize,resize_float=self.resize_float)
        scales=np.ascontiguousarray(scales)
        pts1=pts1/scales[None,:]
        K1[0,0]/=scales[0]
        K1[0,2]/=scales[0]
        K1[1,1]/=scales[1]
        K1[1,2]/=scales[1]

        _,_,scales=process_image(image2,resize=resize,resize_float=self.resize_float)
        scales = np.ascontiguousarray(scales)
        pts2=pts2/scales[None,:]
        K2[0,0]/=scales[0]
        K2[0,2]/=scales[0]
        K2[1,1]/=scales[1]
        K2[1,2]/=scales[1]

        if self.round:
            pts1=np.round(pts1)
            pts2=np.round(pts2)
        results=self.estimate_pose(pts1,pts2,K1,K2,1.0)
        if results is not None:
            R, t, mask = results
        else:
            R, t, mask=np.eye(3), np.zeros(3), np.zeros(pts1.shape[0],np.bool)
        return mask, R, t[:,None]

class LMEDSPoseEstimator:
    def __init__(self,cfg):
        self.configs=cfg

    def pose_estimate(self, pts1, pts2, K1, K2, *args, **kwargs):
        if self.configs['model']=='essential':
            pts_l_norm, pts_r_norm = normalize_coordinates_camera(pts1, pts2, K1, K2)
            E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.LMEDS, prob=self.configs['confidence'])
            points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
        elif self.configs['model']=='fundamental':
            mask, F = self.fundamental_matrix_estimate(pts1,pts2)
            pts1, pts2 = normalize_coordinates_camera(pts1, pts2, K1, K2)
            E = K2.T @ F @ K1
            points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1, pts2, mask=mask[:, None].astype(np.uint8))
            mask=mask[:,None]
        else:
            raise NotImplementedError
        return mask[:,0].astype(np.bool), R_est, t_est

    def fundamental_matrix_estimate(self, pts1, pts2):
        F, lmeds_mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.LMEDS, confidence=self.configs['confidence'])
        return lmeds_mask[:,0].astype(np.bool), F

name2estimator={
    'ransac': RANSACEstimator,
    'rescale_ransac': RescaleRANSACEstimator,
    'lmeds': LMEDSPoseEstimator,
}