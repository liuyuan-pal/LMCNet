import os

import cv2
import h5py

import numpy as np
import pickle

import yaml
from skimage.io import imread

#######################io#########################################
def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

#####################depth and image###############################

def mask_zbuffer_to_pts(mask, zbuffer, K):
    ys,xs=np.nonzero(mask)
    zbuffer=zbuffer[ys, xs]
    u,v,f=K[0,2],K[1,2],K[0,0]
    depth = zbuffer / np.sqrt((xs - u + 0.5) ** 2 + (ys - v + 0.5) ** 2 + f ** 2) * f

    pts=np.asarray([xs, ys, depth], np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    return np.dot(pts,np.linalg.inv(K).transpose())

def mask_depth_to_pts(mask,depth,K,output_2d=False):
    hs,ws=np.nonzero(mask)
    pts_2d=np.asarray([ws,hs],np.float32).transpose()
    depth=depth[hs,ws]
    pts=np.asarray([ws,hs,depth],np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    if output_2d:
        return np.dot(pts,np.linalg.inv(K).transpose()), pts_2d
    else:
        return np.dot(pts,np.linalg.inv(K).transpose())

def read_render_zbuffer(dpt_pth,max_depth,min_depth):
    zbuffer=imread(dpt_pth)
    mask=zbuffer>0
    zbuffer=zbuffer.astype(np.float64)/2**16*(max_depth-min_depth)+min_depth
    return mask, zbuffer

def zbuffer_to_depth(zbuffer,K):
    u,v,f=K[0,2],K[1,2],K[0,0]
    x=np.arange(zbuffer.shape[1])
    y=np.arange(zbuffer.shape[0])
    x,y=np.meshgrid(x,y)
    x=np.reshape(x,[-1,1])
    y=np.reshape(y,[-1,1])
    depth = np.reshape(zbuffer,[-1,1])

    depth = depth / np.sqrt((x - u + 0.5) ** 2 + (y - v + 0.5) ** 2 + f ** 2) * f
    return np.reshape(depth,zbuffer.shape)

def project_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

#######################image processing#############################

def gray_repeats(img_raw):
    if len(img_raw.shape) == 2: img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3: img_raw = img_raw[:, :, :3]
    return img_raw

def normalize_image(img,mask=None):
    import torch
    if mask is not None: img[np.logical_not(mask.astype(np.bool))]=127
    img=(img.transpose([2,0,1]).astype(np.float32)-127.0)/128.0
    return torch.tensor(img,dtype=torch.float32)

def tensor_to_image(tensor):
    return (tensor * 128 + 127).astype(np.uint8).transpose(1,2,0)

def equal_hist(img):
    if len(img.shape)==3:
        img0=cv2.equalizeHist(img[:,:,0])
        img1=cv2.equalizeHist(img[:,:,1])
        img2=cv2.equalizeHist(img[:,:,2])
        img=np.concatenate([img0[...,None],img1[...,None],img2[...,None]],2)
    else:
        img=cv2.equalizeHist(img)
    return img

def resize_large_image(img,resize_max):
    h,w=img.shape[:2]
    max_side = max(h, w)
    if max_side > resize_max:
        ratio = resize_max / max_side
        if ratio <= 0.5: img = cv2.GaussianBlur(img, (5, 5), 1.5)
        img = cv2.resize(img, (int(round(ratio * w)), int(round(ratio * h))), interpolation=cv2.INTER_LINEAR)
        return img, ratio
    else:
        return img, 1.0

def resize_small_image(img,resize_min):
    h,w=img.shape[:2]
    min_side = min(h, w)
    if min_side < resize_min:
        ratio = resize_min / min_side
        img = cv2.resize(img, (int(round(ratio * w)), int(round(ratio * h))), interpolation=cv2.INTER_LINEAR)
        return img, ratio
    else:
        return img, 1.0
    
############################geometry######################################
def round_coordinates(coord,h,w):
    coord=np.round(coord).astype(np.int32)
    coord[coord[:,0]<0,0]=0
    coord[coord[:,0]>=w,0]=w-1
    coord[coord[:,1]<0,1]=0
    coord[coord[:,1]>=h,1]=h-1
    return coord

def get_img_patch(img,pt,size):
    h,w=img.shape[:2]
    x,y=pt.astype(np.int32)
    xmin=max(0,x-size)
    xmax=min(w-1,x+size)
    ymin=max(0,y-size)
    ymax=min(h-1,y+size)
    patch=np.full([size*2,size*2,3],127,np.uint8)
    patch[ymin-y+size:ymax-y+size,xmin-x+size:xmax-x+size]=img[ymin:ymax,xmin:xmax]
    return patch

def perspective_transform(pts, H):
    tpts = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1) @ H.transpose()
    tpts = tpts[:, :2] / np.abs(tpts[:, 2:]) # todo: why only abs? this one is correct
    return tpts

def get_rot_m(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32) # rn+1,3,3

def get_rot_m_batch(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32).transpose([2,0,1])

def compute_F(K1, K2, R, t):
    """

    :param K1: [3,3]
    :param K2: [3,3]
    :param R:  [3,3]
    :param t:  [3,1]
    :return:
    """
    A = K1 @ R.T @ t # [3,1]
    C = np.asarray([[0,-A[2,0],A[1,0]],
                    [A[2,0],0,-A[0,0]],
                    [-A[1,0],A[0,0],0]])
    F = (np.linalg.inv(K2)).T @ R @ K1.T @ C
    return F

def compute_angle(rotation_diff):
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return angular_distance

def load_h5(filename):
    dict_to_load = {}
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]#.value
    return dict_to_load

def save_h5(dict_to_save, filename):
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])

def pts_to_hpts(pts):
    return np.concatenate([pts,np.ones([pts.shape[0],1])],1)

def hpts_to_pts(hpts):
    return hpts[:,:2]/hpts[:,2:]

def np_skew_symmetric(v):
    M = np.asarray([
        [0, -v[2], v[1],],
        [v[2], 0, -v[0],],
        [-v[1], v[0], 0,],
    ])

    return M


def point_line_dist(hpts,lines):
    """
    :param hpts: n,3 or n,2
    :param lines: n,3
    :return:
    """
    if hpts.shape[1]==2:
        hpts=np.concatenate([hpts,np.ones([hpts.shape[0],1])],1)
    return np.abs(np.sum(hpts*lines,1))/np.linalg.norm(lines[:,:2],2,1)

def epipolar_distance(x0, x1, F):
    """

    :param x0: [n,2]
    :param x1: [n,2]
    :param F:  [3,3]
    :return:
    """

    hkps0 = np.concatenate([x0, np.ones([x0.shape[0], 1])], 1)
    hkps1 = np.concatenate([x1, np.ones([x1.shape[0], 1])], 1)

    lines1 = hkps0 @ F.T
    lines0 = hkps1 @ F

    dist10 = point_line_dist(hkps0, lines0)
    dist01 = point_line_dist(hkps1, lines1)

    return dist10, dist01

def epipolar_distance_mean(x0, x1, F):
    return np.mean(np.stack(epipolar_distance(x0,x1,F),1),1)

def compute_dR_dt(R0, t0, R1, t1):
    # Compute dR, dt
    dR = np.dot(R1, R0.T)
    dt = t1 - np.dot(dR, t0)
    return dR, dt

def compute_precision_recall_np(pr,gt,eps=1e-5):
    tp=np.sum(gt & pr)
    fp=np.sum((~gt) & pr)
    fn=np.sum(gt & (~pr))
    precision=(tp+eps)/(fp+tp+eps)
    recall=(tp+eps)/(tp+fn+eps)
    if precision<1e-3 or recall<1e-3:
        f1=0.0
    else:
        f1=(2*precision*recall+eps)/(precision+recall+eps)

    return precision, recall, f1


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_stem(path,suffix_len=5):
    return os.path.basename(path)[:-suffix_len]

def load_component(component_func,component_cfg_fn):
    component_cfg=load_cfg(component_cfg_fn)
    return component_func[component_cfg['type']](component_cfg)

def get_pixel_val(img,pts,interpolation=cv2.INTER_LINEAR):
    # img [h,w,k] pts [n,2]
    if len(pts)<32767:
        pts=pts.astype(np.float32)
        return cv2.remap(img,pts[:,None,0],pts[:,None,1],borderMode=cv2.BORDER_CONSTANT,borderValue=0,interpolation=interpolation)[:,0]
        pn=len(pts)
        sl=int(np.ceil(np.sqrt(pn)))
        tmp_img=np.zeros([sl*sl,2],np.float32)
        tmp_img[:pn]=pts
        tmp_img=tmp_img.reshape([sl,sl,2])
        tmp_img=cv2.remap(img,tmp_img[:,:,0],tmp_img[:,:,1],borderMode=cv2.BORDER_CONSTANT,borderValue=0,interpolation=interpolation)
        return tmp_img.flatten()[:pn]