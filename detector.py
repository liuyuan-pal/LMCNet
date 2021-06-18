import cv2
import numpy as np
import torch

from others.superpoint import SuperPoint, process_image, process_image_fn

class SIFTDetector:
    def __init__(self, cfg):
        num_kp = 2000
        contrastThreshold = 1e-5
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    def __call__(self, img):
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp])  # N*4
        sizes = np.asarray([_kp.size for _kp in cv_kp])
        responses = np.asarray([_kp.response for _kp in cv_kp])
        angles = np.asarray([_kp.angle for _kp in cv_kp])
        kp=np.concatenate([kp,responses[:,None],sizes[:,None],angles[:,None]],1)
        return kp, desc

class SuperPointDetector:
    def __init__(self,cfg):
        self.sp2=SuperPoint(cfg).eval()
        self.sp2=self.sp2.cuda()
        self.cfg=cfg

    def extract_kps_desc_scores(self,img):
        resize=(self.cfg['resize'],) if isinstance(self.cfg['resize'],int) else self.cfg['resize']
        resize_float = False if 'resize_float' in self.cfg and not self.cfg['resize_float'] else True
        image, image_tensor, scales = process_image(img,'cuda',resize, resize_float)
        with torch.no_grad():
            sp_results=self.sp2({'image':image_tensor})
        kps=sp_results['keypoints'][0].detach().cpu().numpy()
        desc=sp_results['descriptors'][0].detach().cpu().numpy()
        scores=sp_results['scores'][0].detach().cpu().numpy()
        desc=desc.T
        kps*=np.asarray(scales)[None,:]
        return kps, desc, scores

    def extract_kps_desc_scores_fn(self,img_fn):
        resize=(self.cfg['resize'],) if isinstance(self.cfg['resize'],int) else self.cfg['resize']
        resize_float = False if 'resize_float' in self.cfg and not self.cfg['resize_float'] else True
        image, image_tensor, scales = process_image_fn(img_fn,'cuda',resize, resize_float)
        with torch.no_grad():
            sp_results=self.sp2({'image':image_tensor})
        kps=sp_results['keypoints'][0].detach().cpu().numpy()
        desc=sp_results['descriptors'][0].detach().cpu().numpy()
        scores=sp_results['scores'][0].detach().cpu().numpy()
        desc=desc.T
        kps*=np.asarray(scales)[None,:]
        return kps, desc, scores

    def __call__(self, img, *args, **kwargs):
        if len(args)>0:
            fn=args[0]
            kps, desc, scores = self.extract_kps_desc_scores_fn(fn)
        else:
            kps, desc, scores = self.extract_kps_desc_scores(img)
        kps=np.concatenate([kps,scores[:,None]],1)
        return kps, desc

name2det = {
    'superpoint': SuperPointDetector,
    'sift': SIFTDetector,
}