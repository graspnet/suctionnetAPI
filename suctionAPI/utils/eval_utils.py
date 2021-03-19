import os
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat
import skimage
from .rotation import batch_viewpoint_to_matrix, matrix_to_dexnet_params, viewpoint_to_matrix
import torch
import torch.nn as nn
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile


k = 15.6
g = 9.8
radius = 0.01
wrench_thre = k * radius * np.pi


def get_scene_name(num):
    return ('scene_%04d' % (num,))

def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points

def parse_posevector(posevector):
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def transform_points(points, trans):
    '''
    Input:
        points: (N, 3)
        trans: (4, 4)
    Output:
        points_trans: (N, 3)
    '''
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    points_trans = points_[:,:3]
    return points_trans

def compute_point_distance(A, B):
    '''
    Input:
        A: (N, 3)
        B: (M, 3)
    Output:
        dists: (N, M)
    '''
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def compute_closest_points(A, B):
    '''
    Input:
        A: (N, 3)
        B: (M, 3)
    Output:
        indices: (N,) closest point index in B for each point in A
    '''
    dists = compute_point_distance(A, B)
    indices = np.argmin(dists, axis=-1)
    return indices

def voxel_sample_points(points, voxel_size=0.008):
    '''
    Input:
        points: (N, 3)
    Output:
        points: (n, 3)
    '''
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points)
    return points

def topk_suctions(suctions, k=10):
    '''
    Input:
        suctions: (N, 17)
        k: int
    Output:
        topk_suctions: (k, 17)
    '''
    assert(k > 0)
    suction_confidence = suctions[:, 0]
    indices = np.argsort(-suction_confidence)
    topk_indices = indices[:min(k, len(suctions))]
    topk_suctions = suctions[topk_indices]
    return topk_suctions

def get_suction_score(suction, model_points, align_mat, camera_pose):
    suction_point = suction[4:7]
    direction = suction[1:4]
    g_direction = np.array([[0, 0, -1]], dtype=np.float32)
    g_direction = transform_points(g_direction, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    g_direction = g_direction / np.linalg.norm(g_direction)
    center = model_points.mean(axis=0)
    
    smoothness_score = get_smoothness_score(suction_point, direction, model_points)
    wrench_score = get_wrench_score(suction_point, direction, center, g_direction)
    return smoothness_score, wrench_score


def get_smoothness_score(suction_point, direction, model_points):
    radius = 0.01
    num_split = 72
    radian_bins = []
    x = []
    y = []
    ideal_vertices = []
    
    for i in range(num_split):
        radian = 2 * np.pi * i / num_split - np.pi
        radian_bins.append(radian)
        x.append(radius * np.cos(radian + np.pi / num_split))
        y.append(radius * np.sin(radian + np.pi / num_split))
        ideal_vertices.append(np.array([x[-1], y[-1], 0]))
    
    radian_bins.append(np.pi)

    ideal_lengths = []
    for j in range(1, len(ideal_vertices)):
        ideal_lengths.append(np.linalg.norm(ideal_vertices[j] - ideal_vertices[j-1]))
    ideal_lengths.append(np.linalg.norm(ideal_vertices[0] - ideal_vertices[-1]))

    suction_failed = False

    new_z = direction
    new_z = new_z / np.linalg.norm(new_z)
    new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
    new_y = new_y / np.linalg.norm(new_y)

    new_x = np.cross(new_y, new_z)
    new_x = new_x / np.linalg.norm(new_x)

    new_x = np.expand_dims(new_x, axis=1)
    new_y = np.expand_dims(new_y, axis=1)
    new_z = np.expand_dims(new_z, axis=1)

    new_coords = np.concatenate((new_x, new_y, new_z), axis=-1)
    rot_matrix = new_coords

    translated_points = model_points - suction_point[np.newaxis, :]
    transformed_points = np.dot(translated_points, rot_matrix)

    transformed_xy = transformed_points[:, 0:2]
    dist = np.linalg.norm(transformed_xy, axis=-1)
    
    mask_above = transformed_points[:, 2] > 0.005
    mask_wi_radius = dist < radius
    collision_mask = mask_above & mask_wi_radius
    collision = np.any(collision_mask)

    if collision:
        suction_failed = True
    idx = np.where((dist > radius - 0.001) & (dist < radius + 0.001))[0]
    transformed_points = transformed_points[idx]
    if transformed_points.shape[0] == 0:
        suction_failed = True
    else:
        transformed_points[:, 2] -= 10 
    topview_radian = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
    bin_points_list = []
    real_vertices = []
    
    for j in range(1, len(radian_bins)):
        l_limit = radian_bins[j - 1]
        r_limit = radian_bins[j]

        bin_points = transformed_points[np.where((topview_radian > l_limit) & (topview_radian < r_limit))]
        if bin_points.shape[0] == 0:
            suction_failed = True
            break
        bin_points_list.append(bin_points)
        real_vertices.append(np.array([x[j-1], y[j-1], bin_points[:, 2].max()]))

    
    if suction_failed:
        quality = 0
    else:
        real_lengths = []
        for j in range(1, len(real_vertices)):
            real_lengths.append(np.linalg.norm(real_vertices[j] - real_vertices[j-1]))
        
        real_lengths.append(np.linalg.norm(real_vertices[0] - real_vertices[-1]))

        assert len(ideal_lengths) == len(real_lengths)

        deform_ratios = []
        for i in range(len(ideal_lengths)):
            deform_ratios.append(ideal_lengths[i] / real_lengths[i])
        
        deform = min(deform_ratios)
        
        vertices = np.vstack(real_vertices)
        A = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        ones = np.ones((A.shape[0], 1))
        A = np.hstack((A, ones))
        b = vertices[:, 2]

        w, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
        s = (1.0 / A.shape[0]) * np.square(np.linalg.norm(np.dot(A, w) - b)) * 100000
        fit = np.exp(-s)
        quality = fit * deform
    
    return quality


def get_wrench_score(suction_point, direction, center, g_direction):     
    gravity = g_direction * g

    suction_axis = viewpoint_to_matrix(direction)
    suction2center = (center - suction_point)[np.newaxis, :]
    coord = np.matmul(suction2center, suction_axis)

    gravity_proj = np.matmul(gravity, suction_axis)

    torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]
    torque_z = -gravity_proj[0, 0] * coord[0, 1] + gravity_proj[0, 1] * coord[0, 0]

    torque_max = max(abs(torque_z), abs(torque_y))
    score = 1 - min(torque_max / wrench_thre, 1) 

    return score


def collision_detection(suction_list, model_list, scene_points, outlier=0.1):
    '''
    Input:
        suction_list: [(k1,17), (k2,17), ..., (kn,17)] in camera coordinate
        model_list: [(N1, 3), (N2, 3), ..., (Nn, 3)] in camera coordinate
        scene_points: (Ns, 3) in camera coordinate
    Output:
        collsion_mask_list: [(k1,), (k2,), ..., (kn,)]
    '''
    height = 0.1
    radius = 0.01
    
    collision_mask_list = list()
    num_models = len(model_list)
    empty_mask_list = list()

    for i in range(num_models):
        if len(suction_list[i][0]) == 0:
            collision_mask_list.append(list())
            empty_mask_list.append(list())
            continue

        model = model_list[i]
        suctions = suction_list[i]
        suction_points = suctions[:, 4:7]
        suction_directions = suctions[:, 1:4]
        
        # crop scene, remove outlier
        xmin, xmax = model[:,0].min(), model[:,0].max()
        ymin, ymax = model[:,1].min(), model[:,1].max()
        zmin, zmax = model[:,2].min(), model[:,2].max()
        xlim = ((scene_points[:,0] > xmin-outlier) & (scene_points[:,0] < xmax+outlier))
        ylim = ((scene_points[:,1] > ymin-outlier) & (scene_points[:,1] < ymax+outlier))
        zlim = ((scene_points[:,2] > zmin-outlier) & (scene_points[:,2] < zmax+outlier))
        workspace = scene_points[xlim & ylim & zlim]
        
        target = (workspace[np.newaxis,:,:] - suction_points[:,np.newaxis,:])
        
        suction_poses = batch_viewpoint_to_matrix(suction_directions) # suction to camera coordinate
        
        target = np.matmul(target, suction_poses)
        target_yz = target[..., 1:3]
        target_r = np.linalg.norm(target_yz, axis=-1)
        mask1 = target_r < radius
        mask2 = ((target[..., 0] > 0.01) & (target[..., 0] < height))
        
        collision_mask = np.any(mask1 & mask2, axis=-1)
        collision_mask_list.append(collision_mask)

    return collision_mask_list

def eval_suction(suction_group, models, dense_models, poses, align_mat, camera_pose, table=None):
    '''
        models: in model coordinate
        poses: from model to camera coordinate
        table: in camera coordinate
    '''
    num_models = len(models)
    ## suction nms
    suction_group = suction_group.nms(0.02, 181.0/180*np.pi)

    ## assign suctions to object
    # merge and sample scene
    model_trans_list = list()
    dense_model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)

        dense_model = dense_models[i]
        dense_model_trans = transform_points(dense_model, poses[i])
        dense_model_trans_list.append(dense_model_trans)

    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)
    
    # assign suctions
    indices = compute_closest_points(suction_group.translations(), scene)
    model_to_suction = seg_mask[indices]
    suction_list = list()
    for i in range(num_models):
        suction_i = suction_group.suction_group_array[model_to_suction==i]
        if len(suction_i) == 0:
            suction_list.append(np.array([[]]))
            continue
        suction_i = topk_suctions(suction_i, k=10)
        suction_list.append(suction_i)

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])
    
    collision_mask_list = collision_detection(suction_list, model_trans_list, scene, outlier=0.05)
        
    # evaluate suctions
    # get suction scores
    smoothness_score_list = list()
    wrench_score_list = list()

    for i in range(num_models):
        collision_mask = collision_mask_list[i]
        suctions = suction_list[i]
        smoothness_scores = list()
        wrench_scores = list()

        if len(suction_list[i][0]) == 0:
            continue

        if len(collision_mask) != len(suctions):
            print('collision_mask len:', len(collision_mask))
            print('suctions len:', len(suctions))
        
        num_suctions = len(suctions)
        for suction_id in range(num_suctions):
            if suctions[suction_id] is None:
                smoothness_scores.append(0.)
                wrench_scores.append(0.)
                continue

            if collision_mask[suction_id]:
                smoothness_scores.append(0.)
                wrench_scores.append(0.)
                continue
            
            suction = suctions[suction_id]
            smoothness_score, wrench_score = get_suction_score(suction, dense_model_trans_list[i], align_mat, camera_pose)
            smoothness_scores.append(smoothness_score)
            wrench_scores.append(wrench_score)
            
        smoothness_score_list.append(np.array(smoothness_scores))
        wrench_score_list.append(np.array(wrench_scores))
    
    return suction_list, smoothness_score_list, wrench_score_list, collision_mask_list

