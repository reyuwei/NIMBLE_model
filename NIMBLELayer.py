'''
    NIMBLE: A Non-rigid Hand Model with Bones and Muscles[SIGGRAPH-22]
    https://reyuwei.github.io/proj/nimble
'''

import torch
import trimesh
from utils import *

class NIMBLELayer(torch.nn.Module):
    __constants__ = [
        'use_pose_pca', 'shape_ncomp', 'pose_ncomp', 'pm_dict'
    ]
    def __init__(self, pm_dict, tex_dict, device, shape_ncomp=20, pose_ncomp=30, tex_ncomp=10, use_pose_pca=True, nimble_mano_vreg=None):
        super(NIMBLELayer, self).__init__()
        self.device = device

        identity_rot = torch.eye(3).to(self.device)
        self.register_buffer("identity_rot", identity_rot)
        
        self.shape_ncomp = shape_ncomp
        self.pose_ncomp = pose_ncomp
        self.tex_ncomp = tex_ncomp
        self.use_pose_pca = use_pose_pca
        self.tex_size = 1024

        self.bone_v_sep = pm_dict['bone_v_sep']
        self.skin_v_sep = pm_dict['skin_v_sep']

        self.register_buffer("th_verts", pm_dict['vert'].squeeze())
        self.register_buffer("th_faces", pm_dict['face'].squeeze())
        self.register_buffer("sw", pm_dict['all_sw'].squeeze())
        self.register_buffer("pbs", pm_dict['all_pbs'].squeeze())
        self.register_buffer("jreg_mano", pm_dict['jreg_mano'].squeeze())
        self.register_buffer("jreg_bone", pm_dict['jreg_bone'].squeeze())
        self.register_buffer("shape_basis", pm_dict['shape_basis'].squeeze())
        self.register_buffer("shape_pm_std", pm_dict['shape_pm_std'].squeeze())
        self.register_buffer("shape_pm_mean", pm_dict['shape_pm_mean'].squeeze())
        self.register_buffer("pose_basis", pm_dict['pose_basis'].squeeze())
        self.register_buffer("pose_mean", pm_dict['pose_mean'].squeeze())
        self.register_buffer("pose_pm_std", pm_dict['pose_pm_std'].squeeze())
        self.register_buffer("pose_pm_mean", pm_dict['pose_pm_mean'].squeeze())

        self.register_buffer("tex_diffuse_basis", tex_dict['diffuse']['basis'].squeeze())
        self.register_buffer("tex_diffuse_mean", tex_dict['diffuse']['mean'].squeeze())
        self.register_buffer("tex_diffuse_std", tex_dict['diffuse']['std'].squeeze())
        self.register_buffer("tex_normal_basis", tex_dict['normal']['basis'].squeeze())
        self.register_buffer("tex_normal_mean", tex_dict['normal']['mean'].squeeze())
        self.register_buffer("tex_normal_std", tex_dict['normal']['std'].squeeze())
        self.register_buffer("tex_spec_basis", tex_dict['spec']['basis'].squeeze())
        self.register_buffer("tex_spec_mean", tex_dict['spec']['mean'].squeeze())
        self.register_buffer("tex_spec_std", tex_dict['spec']['std'].squeeze())

        self.register_buffer("bone_f", pm_dict['bone_f'])
        self.register_buffer("muscle_f", pm_dict['muscle_f'])
        self.register_buffer("skin_f", pm_dict['skin_f'])

        self.skin_v_surface_mask = pm_dict['skin_v_surface_mask'].type(torch.bool)
        self.skin_v_node_weight = dis_to_weight(pm_dict['skin_v_gd'], 30, 50)

        if nimble_mano_vreg is not None:
            self.register_buffer("nimble_mano_vreg_fidx", nimble_mano_vreg['lmk_faces_idx'])
            self.register_buffer("nimble_mano_vreg_bc", nimble_mano_vreg['lmk_bary_coords'])
        else:
            assert "nimble_mano_vreg is None!!" 

        # Kinematic chain params
        kinetree = JOINT_PARENT_ID_DICT
        self.kintree_parents = []
        for i in range(STATIC_JOINT_NUM):
            self.kintree_parents.append(kinetree[i])

    @property
    def bone_v(self):
        bone_v = self.th_verts[:,:self.bone_v_sep,:]
        return bone_v
   
    @property
    def muscle_v(self):
        muscle_v = self.th_verts[:,self.bone_v_sep:self.skin_v_sep,:]
        return muscle_v
  
    @property
    def skin_v(self):
        skin_v = self.th_verts[:,self.skin_v_sep:,:]
        return skin_v
    

    def nimble_to_mano(self, verts, is_surface=False):
        skin_f = self.skin_f
        if not is_surface:
            skin_v = verts[:,self.skin_v_sep:,:]
        else:
            skin_v = verts

        nimble_mano = torch.cat([vertices2landmarks(skin_v, skin_f.squeeze(), self.nimble_mano_vreg_fidx[i],  self.nimble_mano_vreg_bc[i]).unsqueeze(0) for i in range(20)])
        nimble_mano_v = nimble_mano.mean(0)
        return nimble_mano_v

    def compute_warp(self, batch_size, points, skinning_weights, full_trans_mat):
        if points.shape[0] != batch_size:
            points = points.repeat(batch_size, 1, 1)
        if skinning_weights.shape[0] != batch_size:
            skinning_weights = skinning_weights.repeat(batch_size, 1, 1)

        th_T = torch.einsum('bijk,bkt->bijt',full_trans_mat, skinning_weights.permute(0, 2, 1))
        th_rest_shape_h = torch.cat([points.transpose(2, 1),
                                     torch.ones((batch_size, 1, points.shape[1]), dtype=skinning_weights.dtype,
                                                device=skinning_weights.device), ], 1)
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        return th_verts

    def generate_hand_shape(self, betas, normalized=True):
        # beta : B, N
        batch_size, shape_ncomp = betas.shape
        assert self.shape_ncomp == shape_ncomp

        if normalized:
            betas_real = betas * self.shape_pm_std[:shape_ncomp].reshape(1, -1) + self.shape_pm_mean[:shape_ncomp].reshape(1, -1)
        else:
            betas_real = betas
        th_v_shaped = (self.shape_basis[:shape_ncomp].T @ betas_real.T).view(-1, 3, batch_size).permute(2, 0, 1) + self.th_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        
        jreg_bone_joints = torch.matmul(self.jreg_bone, th_v_shaped[:, :self.bone_v_sep])
        return th_v_shaped, jreg_bone_joints

    def generate_full_pose(self, theta, normalized=True, with_root=True):
        # theta : B, N
        batch_size = theta.shape[0]

        if with_root:
            real_theta = theta[:, 3:]
            root_rot = theta[:, :3]
        else:
            real_theta = theta
            root_rot = torch.zeros([batch_size, 3]).to(theta.device)

        pose_ncomp = real_theta.shape[-1]
        if normalized:
            theta_real_denorm = real_theta * self.pose_pm_std[:pose_ncomp].reshape(1, -1) + self.pose_pm_mean[:pose_ncomp].reshape(1, -1)
        else:
            theta_real_denorm = real_theta

        full_pose = (self.pose_basis[:pose_ncomp].T @ theta_real_denorm.T).T + self.pose_mean.unsqueeze(0).repeat(batch_size, 1)
        full_pose = torch.cat([root_rot, full_pose], dim=1).view(batch_size, -1, 3)

        return full_pose

    def generate_texture(self, alpha, normalized=True):
        if alpha is None:
            return self.tex_mean.unsqueeze(0).repeat(batch_size, 1)
            
        batch_size = alpha.shape[0]
        assert self.tex_ncomp == alpha.shape[1]

        if normalized:
            alpha_real_d = alpha * self.tex_diffuse_std[:self.tex_ncomp].reshape(1, -1)
            alpha_real_n = alpha * self.tex_normal_std[:self.tex_ncomp].reshape(1, -1)
            alpha_real_s = alpha * self.tex_spec_std[:self.tex_ncomp].reshape(1, -1)

        x_d = (self.tex_diffuse_basis[:, :self.tex_ncomp] @ alpha_real_d.T).T + self.tex_diffuse_mean.unsqueeze(0).repeat(batch_size, 1)
        x_d = x_d.reshape(batch_size, self.tex_size, self.tex_size, 3)

        x_n = (self.tex_normal_basis[:, :self.tex_ncomp] @ alpha_real_n.T).T + self.tex_normal_mean.unsqueeze(0).repeat(batch_size, 1)
        x_n = x_n.reshape(batch_size, self.tex_size, self.tex_size, 3)

        x_s = (self.tex_spec_basis[:, :self.tex_ncomp] @ alpha_real_s.T).T + self.tex_spec_mean.unsqueeze(0).repeat(batch_size, 1)
        x_s = x_s.reshape(batch_size, self.tex_size, self.tex_size, 3)

        x = torch.cat([x_d, x_n, x_s], dim=-1)
        x = torch.clamp(x, min=0, max=1)
        return x

    def forward(self, pose_param, shape_param, texture_param, handle_collision=True):
        """
        Takes points in R^3 and first applies relevant pose and shape blend shapes.
        Then performs skinning.
        """
        if self.use_pose_pca:
            full_pose = self.generate_full_pose(pose_param, normalized=True, with_root=False).view(-1, 20, 3)
        else:
            full_pose = pose_param.view(-1, 20, 3)

        th_v_shaped, jreg_joints = self.generate_hand_shape(shape_param,normalized=True)

        mesh_v, bone_joints = self.forward_full(th_v_shaped, full_pose, None, jreg_joints, self.sw, self.pbs)
        
        skin_v = mesh_v[:, self.skin_v_sep:, :]

        tex_img = self.generate_texture(texture_param)

        if handle_collision:
            skin_v = self.handle_collision(mesh_v)
            mesh_v[:, self.skin_v_sep:, :] = skin_v

        muscle_v = mesh_v[:,self.bone_v_sep:self.skin_v_sep,:]
        bone_v = mesh_v[:,:self.bone_v_sep,:]

        return skin_v, muscle_v, bone_v, bone_joints, tex_img


    def forward_full(self, points, pose, root_trans, joints, skinning_weight, pose_bs=None, global_scale=None):
        batch_size = pose.shape[0]

        # Convert axis-angle representation to rotation matrix rep.
        th_pose_map, th_rot_map = th_posemap_axisang_2output(pose.view(batch_size, -1))
        th_full_pose = pose.view(batch_size, -1, 3)
        root_rot = batch_rodrigues(th_full_pose[:, 0]).view(batch_size, 3, 3)

        th_j = joints

        if pose_bs is not None:
        # th_pose_map: 1, 19*3
        # points: B, N, 3
        # template_muscle_tet_pose_bs: N, 3, 25*3
        # with pose blend shape
            points_pose_bs = points + torch.matmul(
                pose_bs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        else:
            points_pose_bs = points

        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(STATIC_JOINT_NUM - 1):
            i_val_joint = int(i + 1)
            if i_val_joint in JOINT_ID_BONE_DICT:
                i_val_bone = JOINT_ID_BONE_DICT[i_val_joint]
                joint_rot = th_rot_map[:, (i_val_bone - 1) * 9:i_val_bone * 9].contiguous().view(batch_size, 3, 3)
            else:
                joint_rot = self.identity_rot.repeat(batch_size, 1, 1)

            joint_j = th_j[:, i_val_joint, :].contiguous().view(batch_size, 3, 1)
            parent = self.kintree_parents[i_val_joint]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))

            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))

        th_results_global = th_results
        th_results2 = torch.zeros((batch_size, 4, 4, STATIC_JOINT_NUM),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(STATIC_JOINT_NUM):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)
        

        skinning_weight = skinning_weight.reshape(1, -1, STATIC_JOINT_NUM)
        th_verts = self.compute_warp(batch_size, points_pose_bs, skinning_weight, th_results2)

        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # global scaling
        if global_scale is not None:
            center_joint = th_jtr[:, ROOT_JOINT_IDX].unsqueeze(1)
            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint

            verts_scale = global_scale.expand(th_verts.shape[0], th_verts.shape[1])
            verts_scale = verts_scale.unsqueeze(2).repeat(1, 1, 3)
            th_verts = th_verts * verts_scale
            th_verts = th_verts + center_joint

            j_scale = global_scale.expand(th_jtr.shape[0], th_jtr.shape[1])
            j_scale = j_scale.unsqueeze(2).repeat(1, 1, 3)
            th_jtr = th_jtr * j_scale
            th_jtr = th_jtr + center_joint

        # global translation
        if root_trans is not None:
            root_position = root_trans.view(batch_size, 1, 3)
            center_joint = th_jtr[:, ROOT_JOINT_IDX].unsqueeze(1)
            offset = root_position - center_joint
        
            th_jtr = th_jtr + offset
            th_verts = th_verts + offset

        return th_verts, th_jtr


    def mesh_collision(self, floating_verts, floating_verts_normals, steady_verts, steady_faces):
        ### go to trimesh
        batch_size = floating_verts.shape[0]
        for i in range(batch_size):
            mesh_muscle = trimesh.Trimesh(steady_verts[i].detach().cpu().numpy(),
                                        steady_faces.squeeze().detach().cpu().numpy())
            skin_in_muscle = mesh_muscle.contains(floating_verts[i].detach().cpu().numpy())
            skin_surf_in_muscle = self.skin_v_surface_mask & torch.from_numpy(skin_in_muscle).to(self.device)
            if skin_surf_in_muscle.sum() <= 1:
                continue
            inside_verts = floating_verts[i][skin_surf_in_muscle].reshape(-1, 3)
            inside_verts_normal = floating_verts_normals[i][skin_surf_in_muscle].reshape(-1, 3)

            ## moving target using ray-triangle hit
            locations, index_ray, index_tri = mesh_muscle.ray.intersects_location(inside_verts.squeeze().detach().cpu().numpy(), 
                                            inside_verts_normal.squeeze().detach().cpu().numpy())
            locations = locations + 2* inside_verts_normal.squeeze().detach().cpu().numpy()[index_ray] # outside 2 mm
            index_ray = torch.from_numpy(index_ray).to(self.device)
            offset = torch.zeros_like(inside_verts)
            offset[index_ray] = torch.from_numpy(locations).float().to(self.device) - inside_verts[index_ray]

            ## move
            skin_v_offset = torch.zeros_like(floating_verts[i])
            skin_v_offset[skin_surf_in_muscle] = offset

            hard_result = floating_verts[i] + skin_v_offset
            soft_result = floating_verts[i] + (self.skin_v_node_weight.unsqueeze(-1) * skin_v_offset).sum(1)
            final_result = 0.7 * hard_result + 0.3 * soft_result
            floating_verts[i] = final_result
            
        return floating_verts


    def handle_collision(self, th_verts):
        muscle_v = th_verts[:,self.bone_v_sep:self.skin_v_sep,:]
        skin_v = th_verts[:,self.skin_v_sep:,:]
        interp_meshes_skin = Meshes(skin_v, self.skin_f.repeat(skin_v.shape[0], 1, 1))

        skin_v = self.mesh_collision(skin_v, interp_meshes_skin.verts_normals_padded(), muscle_v, self.muscle_f)
       
        return skin_v
        