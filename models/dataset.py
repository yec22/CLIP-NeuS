import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os, copy
import json
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from .clip_encoder import CLIP_Encoder
from PIL import Image
import torchvision

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    return c2w

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.clip = CLIP_Encoder(self.device)
        self.stage = conf.get_int('stage')

        self.data_dir = conf.get_string('data_dir')
        self.prior_dir = os.path.join(self.data_dir, 'prior')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        self.select_sparse_view = conf.get_list('training_views')
        
        if self.stage == 1:
            self.images_lis = sorted(glob(os.path.join(self.prior_dir, '*_rgb.png')))
        if self.stage == 2:
            self.images_lis = sorted(glob(os.path.join(self.prior_dir, '*_rgb.png')))
            # self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.images_lis = [self.images_lis[i] for i in self.select_sparse_view]
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0

        if self.stage == 1:
            self.masks_lis = sorted(glob(os.path.join(self.prior_dir, '*_mask.png')))
        if self.stage == 2:
            self.masks_lis = sorted(glob(os.path.join(self.prior_dir, '*_mask.png')))
            # self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_lis = [self.masks_lis[i] for i in self.select_sparse_view]
        self.masks_np = np.stack([cv.imread(im_name)[:, :, 0] for im_name in self.masks_lis]) / 256.0
        self.masks_np = self.masks_np > 0.5

        self.H = self.images_np[0].shape[0]
        self.W = self.images_np[0].shape[1]

        with open(os.path.join(self.prior_dir, 'meta_data.json'), 'r') as f:
            self.resize_camera = json.load(f)
        self.resize_camera = self.resize_camera["frames"]
        self.resize_camera = [self.resize_camera[i] for i in self.select_sparse_view]

        self.gt_clip_feat = []
        for img_idx, mask_idx in zip(self.images_lis, self.masks_lis):
            img = cv.imread(img_idx, -1)
            masks = (cv.imread(mask_idx, -1) / 256.0).astype(bool)
            img[~masks] = 0.0
            # img = cv.resize(img, (self.W // 4, self.H // 4), interpolation=cv.INTER_CUBIC)
            img = torchvision.transforms.ToTensor()(img)

            with torch.no_grad():
                self.gt_clip_feat.append(self.clip.CLIP_Encode(img).squeeze().float())

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.select_sparse_view]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.select_sparse_view]

        self.intrinsics_all = []
        self.pose_all = []

        self.random_pose_num = len(sorted(glob(os.path.join(self.data_dir, 'image/*.png'))))
        self.random_world_mat =  [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.random_pose_num)]
        self.random_scale_mat = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.random_pose_num)]

        for i, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, self.world_mats_np)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            resize_intrinsics = np.array(self.resize_camera[i]["intrinsics"]).astype(np.float32)
            if self.stage == 1:
                self.intrinsics_all.append(torch.from_numpy(resize_intrinsics).float())
            if self.stage == 2:
                self.intrinsics_all.append(torch.from_numpy(resize_intrinsics).float())
                # self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rand_pose_list = []
        for rand_scale_mat, rand_world_mat in zip(self.random_scale_mat, self.random_world_mat):
            P = rand_world_mat @ rand_scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.rand_pose_list.append(torch.from_numpy(pose).float())
        
        self.global_intrinsic = self.intrinsics_all[0].to(self.device)
        self.global_intrinsic_inv = torch.inverse(self.global_intrinsic)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()                # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).unsqueeze(-1).cpu()   # [n_images, H, W, 1]
        if self.stage == 1:
            self.normal_lis = sorted(glob(os.path.join(self.prior_dir, '*_normal.png')))
            self.normal_lis = [self.normal_lis[i] for i in self.select_sparse_view]
            self.normal_np = self.read_normal()
            self.normals = torch.from_numpy(self.normal_np.astype(np.float32)).cpu()                # [n_images, H, W, 3]

            self.depth_lis = sorted(glob(os.path.join(self.prior_dir, '*_depth.npy')))
            self.depth_lis = [self.depth_lis[i] for i in self.select_sparse_view]
            self.depth_np = np.stack([np.load(im_name) for im_name in self.depth_lis])
            self.depths = torch.from_numpy(self.depth_np.astype(np.float32)).unsqueeze(-1).cpu()    # [n_images, H, W, 1]

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.rand_pose_list = torch.stack(self.rand_pose_list).to(self.device)
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        self.camera_avg_dist = self.get_avg_dist(self.pose_all)

        print('images', self.images.shape)
        print('masks', self.masks.shape)
        print('intrinsics', self.intrinsics_all.shape)
        print('poses', self.pose_all.shape)

        print('Load data: End')

    def read_normal(self):
        data_list = []
        for i, im_name in enumerate(self.normal_lis):
            normal = cv.imread(im_name)
            normal = cv.cvtColor(normal, cv.COLOR_BGR2RGB) # BGR -> RGB
            normal = (normal / 255.0 - 0.5) * 2.0 # [0, 255] -> [-1, 1]
            
            ex_i = np.linalg.inv(self.pose_all[i])
            normal_world = self.get_world_normal(normal.reshape(-1, 3), ex_i).reshape(self.H, self.W, 3)
            
            data_list.append(normal_world)

        return np.stack(data_list)

    def get_world_normal(self, normal, extrin):
        '''
        Args:
            normal: N*3
            extrinsics: 4*4, world to camera
        Return:
            normal: N*3, in world space 
        '''
        extrinsics = copy.deepcopy(extrin)        
        assert extrinsics.shape[0] == 4
        normal = normal.transpose()
        extrinsics[:3, 3] = np.zeros(3)  # only rotation, no translation
        normal_world = np.matmul(np.linalg.inv(extrinsics),
                                np.vstack((normal, np.ones((1, normal.shape[1])))))[:3]
        normal_world = normal_world.transpose((1, 0))

        return normal_world

    def get_avg_dist(self, pose_list):
        cam_dist = 0.
        for pose in pose_list:
            cam_dist += torch.norm(pose[:3, 3], p=2, dim=-1).item()
        cam_avg_dist = cam_dist / pose_list.shape[0]
        return cam_avg_dist

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 1
        if self.stage == 1:
            normal = self.normals[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
            depth = self.depths[img_idx][(pixels_y, pixels_x)]      # batch_size, 1
        if self.stage == 2:
            normal = torch.zeros_like(color).to(color.device)
            depth = torch.zeros_like(mask).to(mask.device)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1], normal, depth], dim=-1).cuda()    # batch_size, 14

    def gen_rays_given_pose(self, pose, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cuda()  # W, H, 3
        p = torch.matmul(self.global_intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)