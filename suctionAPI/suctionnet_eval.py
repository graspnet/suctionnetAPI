
__author__ = 'hwcao'
__version__ = '1.0'


import numpy as np
import os
import sys
import scipy.io as scio
import skimage
import argparse
import cv2
import time
import pickle
import open3d as o3d
from PIL import Image
from suctionnet import SuctionNet
from suction import SuctionGroup
from utils.eval_utils import get_scene_name, create_table_points, parse_posevector,\
     transform_points, compute_point_distance, voxel_sample_points, eval_suction
from utils.xmlhandler import xmlReader


class SuctionNetEval(SuctionNet):
    def __init__(self, root, dense_root, camera, split, save_root):
        super(SuctionNetEval, self).__init__(root, dense_root, camera, split)
        self.save_root = save_root

    def get_scene_models(self, scene_id, ann_id):
        '''
            return models in model coordinate
        '''
        model_dir = os.path.join(self.root, 'models')
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()

        obj_list = []
        model_list = []
        dense_model_list = []
        
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            obj_list.append(obj_idx)
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
            points = np.array(model.points)
            model_list.append(points)

            poisson_dir = os.path.join(self.dense_root, '%03d'%obj_idx + '.npz')     
            data = np.load(poisson_dir)
            points = data['points']
            dense_model_list.append(points)
            
        return model_list, dense_model_list, obj_list

    def get_model_poses(self, scene_id, ann_id):
        '''
            pose_list: object pose from model to camera coordinate
            camera_pose: from camera to world coordinate
            align_mat: from world to table-horizontal coordinate
        '''
        scene_dir = os.path.join(self.root, 'scenes')
        camera_poses_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'camera_poses.npy')
        camera_poses = np.load(camera_poses_path)
        camera_pose = camera_poses[ann_id]
        align_mat_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'cam0_wrt_table.npy')
        align_mat = np.load(align_mat_path)
        scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), self.camera, 'annotations', '%04d.xml'% (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, mat = parse_posevector(posevector)
            obj_list.append(obj_idx)
            pose_list.append(mat)
        return obj_list, pose_list, camera_pose, align_mat
    
    def debug(self, scene_id, dump_folder):
        visu_num = 50

        model_list, dense_model_list, _ = self.get_scene_models(scene_id, ann_id=0)
        table = create_table_points(1.0, 0.05, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.008)
        
        for ann_id in range(256):
            suction_group = SuctionGroup().from_npy(os.path.join(dump_folder,'scene_%d'%scene_id, self.camera, 'suction_info', '%04d.npy' % (ann_id)))
            suction_points = suction_group.translations()
            suction_normals = suction_group.directions()
            
            suction_index = np.random.choice(suction_points.shape[0], visu_num)
            suction_points = suction_points[suction_index]
            suction_normals = suction_normals[suction_index]

            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            model_trans_list = list()
            for i in range(len(model_list)):
                model = model_list[i]
                model_trans = transform_points(model, pose_list[i])
                model_trans_list.append(model_trans)

            arrow_list = []

            for idx in range(len(suction_points)):
                suction_point = suction_points[idx]
                suction_normal = suction_normals[idx]
                suction_normal = suction_normal / np.linalg.norm(suction_normal)
                
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.0015, 
                                                                cylinder_height=0.005, cone_height=0.004)
                arrow_points = np.asarray(arrow.vertices)

                new_z = suction_normal
                new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
                new_y = new_y / np.linalg.norm(new_y)
                new_x = np.cross(new_y, new_z)

                R = np.c_[new_x, np.c_[new_y, new_z]]
                arrow_points = np.dot(R, arrow_points.T).T + suction_point[np.newaxis,:]
                arrow.vertices = o3d.utility.Vector3dVector(arrow_points)

                arrow_list.append(arrow)
            
            o3d.visualization.draw_geometries([*model_trans_list, *arrow_list, table_trans])

    def eval_scene(self, scene_id, split, dump_folder, threshold_list):
        TOP_K = 50

        model_list, dense_model_list, _ = self.get_scene_models(scene_id, ann_id=0)
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.01)
        model_sampled_list = list()
        
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.005)
            model_sampled_list.append(model_sampled)
            
        scene_accuracy = []
        for ann_id in range(256):
            
            suction_group = SuctionGroup().from_npy(os.path.join(dump_folder, split, 'scene_%04d'%scene_id, self.camera, 'suction', '%04d.npz' % (ann_id)))
            
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            suction_list, smoothness_score_list, wrench_score_list, collision_mask_list = eval_suction(suction_group, model_sampled_list, 
                                                                                                dense_model_list, pose_list, align_mat, 
                                                                                                camera_pose, table=table_trans)
            # concat into scene level
            # remove empty
            suction_list = [x for x in suction_list if len(x[0])!= 0]
            smoothness_score_list = [x for x in smoothness_score_list if len(x)!=0]
            wrench_score_list = [x for x in wrench_score_list if len(x)!= 0]
            collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

            suction_list, smoothness_score_list, wrench_score_list, collision_mask_list = np.concatenate(suction_list), \
                                                                                        np.concatenate(smoothness_score_list), \
                                                                                        np.concatenate(wrench_score_list), \
                                                                                        np.concatenate(collision_mask_list)
            
            # sort in scene level
            suction_confidence = suction_list[:, 0]
            indices = np.argsort(-suction_confidence)
            suction_list, smoothness_score_list, wrench_score_list, collision_mask_list = suction_list[indices], \
                                                                                        smoothness_score_list[indices], \
                                                                                        wrench_score_list[indices], \
                                                                                        collision_mask_list[indices]

            suction_accuracy = np.zeros((TOP_K,len(threshold_list)))
            for threshold_idx, threshold in enumerate(threshold_list):
                for k in range(0,TOP_K):
                    # scores[k,fric_idx] is the average score for top k suctions with coefficient of friction at fric
                    if k+1 > len(wrench_score_list):
                        suction_accuracy[k, threshold_idx] = np.sum(((wrench_score_list * smoothness_score_list)>=threshold).astype(np.float32))/(k+1)
                    else:
                        suction_accuracy[k, threshold_idx] = np.sum(((wrench_score_list[0:k+1] * smoothness_score_list[0:k+1])>=threshold).astype(np.float32))/(k+1)

            print('Mean Accuracy for suctions in view {} under threshold {}'.format(ann_id, threshold_list[0]), np.mean(suction_accuracy[:,0])) # 0.2
            print('Mean Accuracy for suctions in view {} under threshold {}'.format(ann_id, threshold_list[1]), np.mean(suction_accuracy[:,1])) # 0.4
            print('Mean Accuracy for suctions in view {} under threshold {}'.format(ann_id, threshold_list[2]), np.mean(suction_accuracy[:,2])) # 0.6
            print('Mean Accuracy for suctions in view {} under threshold {}'.format(ann_id, threshold_list[3]), np.mean(suction_accuracy[:,3])) # 0.8
            print('\rMean Accuracy for scene:{} ann:{} ='.format(scene_id, ann_id),np.mean(suction_accuracy[:, :]), end='\n')
            scene_accuracy.append(suction_accuracy)
        
        return scene_accuracy

    def parallel_eval_scenes(self, scene_ids, dump_folder, threshold_list, proc=2):
        from multiprocessing import Pool
        p = Pool(processes = proc)
        res_list = []
        for scene_id in scene_ids:
            if scene_id >=100 and scene_id < 130:
                split = 'test_seen'
            elif scene_id >=130 and scene_id < 160:
                split = 'test_similar'
            elif scene_id >= 160:
                split = 'test_novel'
            else:
                split = 'train'
            res_list.append(p.apply_async(self.eval_scene, (scene_id, split, dump_folder, threshold_list)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_all(self, dump_folder, threshold_list, proc=2):
        res = np.array(self.parallel_eval_scenes(scene_ids=list(range(100, 190)), dump_folder=dump_folder, threshold_list=threshold_list, proc=proc))
        save_dir = os.path.join(self.save_root, 'accuracy')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, self.camera+'.npy')
        np.save(save_file, res)
        ap = [np.mean(res), np.mean(res[0:30]), np.mean(res[30:60]), np.mean(res[60:90])]
        print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(self.camera, ap[0], ap[1], ap[2], ap[3]))
        return res, ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=None, help='Dataset path. [default: None]')
    parser.add_argument('--dense_root', default=None, help='Upsampled point cloud path. [default: None]')
    parser.add_argument('--pred_root', default=None, help='Suction prediction path. [default: None]')
    parser.add_argument('--save_root', default=None, help='Path to save evaluation results, if None, it will be the same with --pred_root. [default: None]')
    parser.add_argument('--camera', default='kinect', help='Which camera data the prediction uses. [default: kinect]')
    parser.add_argument('--split', default='test', help='Which split to perform evaluation. Could be test_seen, test_similar, test_novel or test (all the above). [default: test]')
    args = parser.parse_args()
    
    data_root = args.data_root
    dense_root = args.dense_root
    pred_root = args.pred_root
    save_root = args.pred_root if args.save_root is None else args.save_root
    camera = args.camera
    split = args.split
    threshold_list = [0.2, 0.4, 0.6, 0.8]
    suction_eval = SuctionNetEval(data_root, dense_root, camera, split, save_root)

    res, ap = suction_eval.eval_all(dump_folder=pred_root, threshold_list=threshold_list, proc=30)

