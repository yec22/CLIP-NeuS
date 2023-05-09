import os
import time
import logging
import argparse
from math import sqrt
import numpy as np
import cv2 as cv
import trimesh
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, Cascaded_SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from torchvision import transforms


class TVLoss(torch.nn.Module):
    """
    TV loss
    """

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = self._tensor_size(x[:, 1:, :, :])
        count_w = self._tensor_size(x[:, :, 1:, :])
        h_tv = torch.pow((x[:, 1:, :, :] - x[:, :h_x - 1, :, :]), 2).sum()
        w_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :w_x - 1, :]), 2).sum()
        return (h_tv + w_tv) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask):
        target = target * 50 + 0.5
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.prediction_depth = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        
        depth_error = (self.prediction_depth - target) * mask
        depth_error = depth_error.reshape(1, -1).permute(1, 0)
        depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / (mask.sum() + 1e-5)

        return depth_loss

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, ckpt=''):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = self.conf.get_int('train.start_iter')

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.clip_loss_freq = self.conf.get_int('train.clip_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.clip_stop = self.conf.get_int('train.stop_clip')

        # Weights
        self.color_weight = self.conf.get_float('train.color_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.clip_weight = self.conf.get_float('train.clip_weight')
        self.depth_tv_weight = self.conf.get_float('train.depth_tv_weight')
        self.normal_tv_weight = self.conf.get_float('train.normal_tv_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        self.depth_weight = self.conf.get_float('train.depth_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.stage = self.conf.get_int('model.cascaded_network.stage')

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = Cascaded_SDFNetwork(**self.conf['model.cascaded_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if ckpt != '':
            latest_model_name = ckpt

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def get_angular_error(self, normals_source, normals_target, mask):
        '''Get angular error betwee predicted normals and ground truth normals
        Args:
            normals_source, normals_target: N*3
            mask: N*1 (optional, default: None)
        Return:
            angular_error: float
        '''
        inner = (normals_source * normals_target).sum(dim=-1, keepdim=True)
        norm_source =  torch.linalg.norm(normals_source, dim=-1, ord=2, keepdim=True)
        norm_target = torch.linalg.norm(normals_target, dim=-1, ord=2, keepdim=True)
        angles = torch.arccos(inner/((norm_source*norm_target) + 1e-5))
        assert not torch.isnan(angles).any()
        if mask.ndim == 1:
            mask =  mask.unsqueeze(-1)
        assert angles.ndim == mask.ndim
        
        angular_error = F.l1_loss(angles*mask, torch.zeros_like(angles), reduction='sum') / (mask.sum() + 1e-5)
        return angular_error

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask, normal_gt, depth_gt = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 13], data[:, 13: 14]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            normal_pred = render_out['normal_map']
            depth_pred = render_out['depth_map'].unsqueeze(-1)
            
            depth_pred = depth_pred.reshape(int(sqrt(self.batch_size)), int(sqrt(self.batch_size)), 1).permute(2, 0, 1)
            depth_gt = depth_gt.reshape(int(sqrt(self.batch_size)), int(sqrt(self.batch_size)), 1).permute(2, 0, 1)
            depth_mask = mask.clone().reshape(int(sqrt(self.batch_size)), int(sqrt(self.batch_size)), 1).permute(2, 0, 1)

            # Loss
            clip_loss, normal_tv_loss, depth_tv_loss = None, None, None
            normal_loss, depth_loss = None, None
            if (self.iter_step + 1) % self.clip_loss_freq == 0 and self.iter_step < self.clip_stop and self.clip_weight > 0:
                rand_idx = torch.randint(low = 0, high = self.dataset.random_pose_num, size=[1]).item()
                rand_pose = self.dataset.rand_pose_list[rand_idx]
                gt_idx = torch.randint(low = 0, high = self.dataset.n_images, size=[1]).item()
                render_img, render_normal, render_depth, normal_patch, depth_patch = self.render_image_from_pose(gt_idx, rand_pose, 5, self.iter_step + 1)
                
                h, w, _ = render_img.shape
                trans = transforms.Compose([
                    transforms.CenterCrop(h),
                ])
                render_img = trans(render_img).permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
                render_img = F.interpolate(render_img, size=(224, 224), mode='bilinear').squeeze()

                render_clip = self.dataset.clip.CLIP_Encode(render_img).squeeze().float()
                gt_clip = self.dataset.gt_clip_feat[gt_idx]
                
                clip_loss = 1 - torch.cosine_similarity(gt_clip, render_clip, dim=-1)

                normal_tv_loss = TVLoss()(normal_patch)
                depth_tv_loss = TVLoss()(depth_patch)

            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            if self.stage == 1:
                normal_loss = self.get_angular_error(normal_pred, normal_gt, mask)
                depth_loss = ScaleAndShiftInvariantLoss()(depth_pred, depth_gt, depth_mask)
            if self.stage == 2:
                normal_loss = 0.0
                depth_loss = 0.0

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss * self.color_weight +\
                eikonal_loss * self.igr_weight +\
                mask_loss * self.mask_weight +\
                normal_loss * self.normal_weight +\
                depth_loss * self.depth_weight
            
            if clip_loss is not None:
                loss += clip_loss * self.clip_weight
            
            if normal_tv_loss is not None:
                loss += normal_tv_loss * self.normal_tv_weight
            
            if depth_tv_loss is not None:
                loss += depth_tv_loss * self.depth_tv_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} normal_loss = {} depth_loss = {}'.format(self.iter_step, loss, normal_loss, depth_loss))
                if clip_loss is not None:
                    print('iter:{:8>d} clip_loss = {} lr={}'.format(self.iter_step, clip_loss, self.optimizer.param_groups[0]['lr']))
                if normal_tv_loss is not None:
                    print('iter:{:8>d} normal_tv_loss = {} lr={}'.format(self.iter_step, normal_tv_loss, self.optimizer.param_groups[0]['lr']))
                if depth_tv_loss is not None:
                    print('iter:{:8>d} depth_tv_loss = {} lr={}'.format(self.iter_step, depth_tv_loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_image_from_pose(self, idx, pose, resolution_level, iter):
        """
        render novel view given camera pose.
        """
        rays_o, rays_d = self.dataset.gen_rays_given_pose(pose, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine, out_depth_fine, out_normal_fine = [], [], []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'])
            out_depth_fine.append(render_out['depth_map'])
            out_normal_fine.append(render_out['normal_map'])
            
            torch.cuda.empty_cache()
            del render_out
            

        img_fine = torch.cat(out_rgb_fine, dim=0).reshape([H, W, 3])
        vis_img = (img_fine.detach().cpu().numpy() * 256).clip(0, 255).astype(np.uint8)

        img_normal = torch.cat(out_normal_fine, dim=0)
        normal_map = img_normal.detach().cpu().numpy()
        rot = np.linalg.inv(pose[:3, :3].cpu().numpy())
        normal_map = np.matmul(rot[None, :, :], normal_map[:, :, None]).reshape([H, W, 3])
        vis_normal_map = (normal_map * 128 + 128).clip(0, 255).astype(np.uint8)

        img_depth = torch.cat(out_depth_fine, dim=0).reshape([H, W])
        vis_img_depth = img_depth.detach().cpu().numpy()
        mi = np.min(vis_img_depth[vis_img_depth > 0]) # get minimum positive depth (ignore background)
        ma = np.max(vis_img_depth)
        vis_img_depth = (vis_img_depth - mi) / (ma - mi + 1e-6) # normalize to 0~1
        vis_img_depth = (255 * vis_img_depth).astype(np.uint8)
        vis_img_depth = cv.applyColorMap(vis_img_depth, cv.COLORMAP_JET)

        patch_size = 4
        patch_x = torch.randint(low=patch_size, high=W-patch_size-1, size=[64])
        patch_y = torch.randint(low=patch_size, high=H-patch_size-1, size=[64])
        patch_normal, patch_depth = [], []
        for i in range(len(patch_x)):
            patch_normal.append(img_normal.reshape([H, W, 3])[patch_y[i]-patch_size:patch_y[i]+patch_size, patch_x[i]-patch_size:patch_x[i]+patch_size])
            patch_depth.append(img_depth.unsqueeze(-1)[patch_y[i]-patch_size:patch_y[i]+patch_size, patch_x[i]-patch_size:patch_x[i]+patch_size])
        patch_normal = torch.stack(patch_normal, dim=0)
        patch_depth = torch.stack(patch_depth, dim=0)
        
        os.makedirs(os.path.join(self.base_exp_dir, 'clip_novel_view'), exist_ok=True)
        if iter % self.report_freq == 0:
            cv.imwrite(os.path.join(self.base_exp_dir, 'clip_novel_view','{:0>8d}_clip.png'.format(self.iter_step)),
                           np.concatenate([vis_img, vis_img_depth, vis_normal_map, self.dataset.image_at(idx, resolution_level=resolution_level)]))

        return img_fine, img_normal.reshape([H, W, 3]), img_depth, patch_normal, patch_depth


    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_reso_{}.ply'.format(self.iter_step, resolution)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

def set_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')

    args = parser.parse_args()
    set_seed(0)

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.ckpt)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode == "render":
        pose = torch.tensor([[0.30679902, 0.5421411, -0.78227705, 2.742191],
                            [-0.8241403, 0.5624584, 0.06658301, -0.33522356],
                            [0.47609568, 0.6242784, 0.61936206, -2.0749938 ],
                            [0.        , 0.       , 0.        , 1.        ]], dtype=torch.float32)
        runner.render_image_from_pose(0, pose.cuda(), 5, 0)
