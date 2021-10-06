
__author__ = 'hwcao'
__version__ = '1.0'


import numpy as np
import os
import open3d as o3d
from .suctionnet import SuctionNet
from .suction import SuctionGroup
from .utils.eval_utils import get_scene_name, create_table_points, parse_posevector,\
     transform_points, voxel_sample_points, eval_suction
from .utils.xmlhandler import xmlReader


class SuctionNetEval(SuctionNet):
    '''
    Class for evaluation on SuctionNet dataset.
    
    **Input:**
    
    - root: string of root path for the dataset.
    
    - camera: string of type of the camera.
    
    - split: string of the date split.
    '''

    def __init__(self, root, camera, split='all'):
        super(SuctionNetEval, self).__init__(root, camera, split)

    def get_scene_models(self, scene_id, ann_id):
        '''
        **Input:**
        
        - scene_id: int of the scen index.
        
        - ann_id: int of the annotation index.
        
        **Output:**

        - model_list: list of model point clouds

        - dense_model_list: list of dense model point clouds created by the create_dense_point_cloud() fuction in suctionnetAPI

        - obj_list: list of object indexes
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

            poisson_dir = os.path.join(self.root, 'dense_point_clouds', '%03d'%obj_idx + '.npz')     
            data = np.load(poisson_dir)
            points = data['points']
            dense_model_list.append(points)
            
        return model_list, dense_model_list, obj_list

    def get_model_poses(self, scene_id, ann_id):
        '''
        **Input:**
        
        - scene_id: int of the scen index.
        
        - ann_id: int of the annotation index.
        
        **Output:**
        
        - obj_list: list of int of object index.
        
        - pose_list: list of 4x4 matrices of object poses.
        
        - camera_pose: 4x4 matrix of the camera pose relative to the first frame.
        
        - align mat: 4x4 matrix of camera relative to the table.
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
    
    def eval_scene(self, scene_id, split, dump_folder):
        '''
        **Input:**
        
        - scene_id: int of the scene index.
        
        - split: string of the data split
        
        - dump_folder: string of the folder that saves the dumped npy files.
        
        **Output:**
        
        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        threshold_list = [0.2, 0.4, 0.6, 0.8]
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

            print('Mean Accuracy for suctions in view {} under threshold {}: {:.6f}'.format(ann_id, threshold_list[0], np.mean(suction_accuracy[:,0]))) # 0.2
            print('Mean Accuracy for suctions in view {} under threshold {}: {:.6f}'.format(ann_id, threshold_list[1], np.mean(suction_accuracy[:,1]))) # 0.4
            print('Mean Accuracy for suctions in view {} under threshold {}: {:.6f}'.format(ann_id, threshold_list[2], np.mean(suction_accuracy[:,2]))) # 0.6
            print('Mean Accuracy for suctions in view {} under threshold {}: {:.6f}'.format(ann_id, threshold_list[3], np.mean(suction_accuracy[:,3]))) # 0.8
            print('\rMean Accuracy for scene:{} ann:{} ='.format(scene_id, ann_id),np.mean(suction_accuracy[:, :]), end='\n')
            scene_accuracy.append(suction_accuracy)
        
        return scene_accuracy

    def parallel_eval_scenes(self, scene_ids, dump_folder, proc=2):
        '''
        **Input:**
        
        - scene_ids: list of int of scene index.
        
        - dump_folder: string of the folder that saves the npy files.
        
        - proc: int of the number of processes to use to evaluate.
        
        **Output:**
        
        - scene_acc_list: list of the scene accuracy.
        '''
        
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
            res_list.append(p.apply_async(self.eval_scene, (scene_id, split, dump_folder)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list
    
    def eval_seen(self, dump_folder, proc = 2):
        '''
        
        **Input:**
        
        - dump_folder: string of the folder that saves the npy files.
        
        - proc: int of the number of processes to use to evaluate.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap_top50: float of the AP of the top 50 suctions for seen split.
       
        - ap_top1: float of the AP of the top 1 suctions for seen split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(100, 130)), dump_folder = dump_folder, proc = proc))
        # ap = np.mean(res)
        # print('\nEvaluation Result:\n----------\n{}, AP Seen={}'.format(self.camera, ap))
        # return res, ap

        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Suctions:\n----------\n{}, AP Seen={:6f}'.format(self.camera, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 Seen={:6f}'.format(self.camera, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 Seen={:6f}'.format(self.camera, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2])
        print('----------\n{}, AP0.6 Seen={:6f}'.format(self.camera, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 Seen={:6f}'.format(self.camera, ap_top50_0dot8))

        ap_top1 = np.mean(res[:, :, :1, :])
        print('\nEvaluation Result of Top 1 Suction:\n----------\n{}, AP Seen={:6f}'.format(self.camera, ap_top1))

        ap_top1_0dot2 = np.mean(res[..., :1, 0])
        print('----------\n{}, AP0.2 Seen={:6f}'.format(self.camera, ap_top1_0dot2))

        ap_top1_0dot4 = np.mean(res[..., :1, 1])
        print('----------\n{}, AP0.4 Seen={:6f}'.format(self.camera, ap_top1_0dot4))

        ap_top1_0dot6 = np.mean(res[..., :1, 2])
        print('----------\n{}, AP0.6 Seen={:6f}'.format(self.camera, ap_top1_0dot6))

        ap_top1_0dot8 = np.mean(res[..., :1, 3])
        print('----------\n{}, AP0.8 Seen={:6f}'.format(self.camera, ap_top1_0dot8))

        return res, ap_top50, ap_top1

    def eval_similar(self, dump_folder, proc = 2):
        '''
        **Input:**
        
        - dump_folder: string of the folder that saves the npy files.
        
        - proc: int of the number of processes to use to evaluate.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap_top50: float of the AP of the top 50 suctions for similar split.
        
        - ap_top1: float of the AP of the top 1 suctions for similar split.
        '''
        
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(130, 160)), dump_folder = dump_folder, proc = proc))
        # ap = np.mean(res)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Similar={}'.format(self.camera, ap, ap))
        # return res, ap

        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Suctions:\n----------\n{}, AP Similar={:6f}'.format(self.camera, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 Simila={:6f}'.format(self.camera, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 Similar={:6f}'.format(self.camera, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2])
        print('----------\n{}, AP0.6 Similar={:6f}'.format(self.camera, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 Similar={:6f}'.format(self.camera, ap_top50_0dot8))

        ap_top1 = np.mean(res[:, :, :1, :])
        print('\nEvaluation Result of Top 1 Suction:\n----------\n{}, AP Similar={:6f}'.format(self.camera, ap_top1))

        ap_top1_0dot2 = np.mean(res[..., :1, 0])
        print('----------\n{}, AP0.2 Similar={:6f}'.format(self.camera, ap_top1_0dot2))

        ap_top1_0dot4 = np.mean(res[..., :1, 1])
        print('----------\n{}, AP0.4 Similar={:6f}'.format(self.camera, ap_top1_0dot4))

        ap_top1_0dot6 = np.mean(res[..., :1, 2])
        print('----------\n{}, AP0.6 Similar={:6f}'.format(self.camera, ap_top1_0dot6))

        ap_top1_0dot8 = np.mean(res[..., :1, 3])
        print('----------\n{}, AP0.8 Similar={:6f}'.format(self.camera, ap_top1_0dot8))

        return res, ap_top50, ap_top1

    def eval_novel(self, dump_folder, proc = 2):
        '''
        **Input:**
        
        - dump_folder: string of the folder that saves the npy files.
        
        - proc: int of the number of processes to use to evaluate.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap_top50: float of the AP of top 50 suctions for novel split.
        
        - ap_top1: float of the AP of top 1 suction for novel split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(160, 190)), dump_folder = dump_folder, proc = proc))
        # ap = np.mean(res)
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Novel={}'.format(self.camera, ap, ap))
        
        ap_top50 = np.mean(res[:, :, :50, :])
        print('\nEvaluation Result of Top 50 Suctions:\n----------\n{}, AP Novel={:6f}'.format(self.camera, ap_top50))

        ap_top50_0dot2 = np.mean(res[..., :50, 0])
        print('----------\n{}, AP0.2 Novel={:6f}'.format(self.camera, ap_top50_0dot2))

        ap_top50_0dot4 = np.mean(res[..., :50, 1])
        print('----------\n{}, AP0.4 Novel={:6f}'.format(self.camera, ap_top50_0dot4))

        ap_top50_0dot6 = np.mean(res[..., :50, 2])
        print('----------\n{}, AP0.6 Novel={:6f}'.format(self.camera, ap_top50_0dot6))

        ap_top50_0dot8 = np.mean(res[..., :50, 3])
        print('----------\n{}, AP0.8 Novel={:6f}'.format(self.camera, ap_top50_0dot8))

        ap_top1 = np.mean(res[:, :, :1, :])
        print('\nEvaluation Result of Top 1 Suction:\n----------\n{}, AP Novel={:6f}'.format(self.camera, ap_top1))

        ap_top1_0dot2 = np.mean(res[..., :1, 0])
        print('----------\n{}, AP0.2 Novel={:6f}'.format(self.camera, ap_top1_0dot2))

        ap_top1_0dot4 = np.mean(res[..., :1, 1])
        print('----------\n{}, AP0.4 Novel={:6f}'.format(self.camera, ap_top1_0dot4))

        ap_top1_0dot6 = np.mean(res[..., :1, 2])
        print('----------\n{}, AP0.6 Novel={:6f}'.format(self.camera, ap_top1_0dot6))

        ap_top1_0dot8 = np.mean(res[..., :1, 3])
        print('----------\n{}, AP0.8 Novel={:6f}'.format(self.camera, ap_top1_0dot8))

        return res, ap_top50, ap_top1

    def eval_all(self, dump_folder, proc=2):
        '''
        **Input:**
        
        - dump_folder: string of the folder that saves the npy files.
        
        - proc: int of the number of processes to use to evaluate.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap_top50: float of the AP of top 50 suctions for all split.
        
        - ap_top1: float of the AP of top 1 suctions for all split.
        '''

        res = np.array(self.parallel_eval_scenes(scene_ids=list(range(100, 190)), dump_folder=dump_folder, proc=proc))
        # ap = [np.mean(res), np.mean(res[0:30]), np.mean(res[30:60]), np.mean(res[60:90])]
        # print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(self.camera, ap[0], ap[1], ap[2], ap[3]))
        
        ap_top50 = [np.mean(res[:, :, :50, :]), np.mean(res[0:30, :, :50, :]), np.mean(res[30:60, :, :50, :]), np.mean(res[60:90, :, :50, :])]
        print('\nEvaluation Result of Top 50 Suctions:\n----------\n{}, AP={:6f}, AP Seen={:6f}, AP Similar={:6f}, AP Novel={:6f}'.format(self.camera, ap_top50[0], ap_top50[1], ap_top50[2], ap_top50[3]))

        ap_top50_0dot2 = [np.mean(res[..., :50, 0]), np.mean(res[0:30, :, :50, 0]), np.mean(res[30:60, :, :50, 0]), np.mean(res[60:90, :, :50, 0])]
        print('----------\n{}, AP0.2={:6f}, AP0.2 Seen={:6f}, AP0.2 Similar={:6f}, AP0.2 Novel={:6f}'.format(self.camera, ap_top50_0dot2[0], ap_top50_0dot2[1], ap_top50_0dot2[2], ap_top50_0dot2[3]))

        ap_top50_0dot4 = [np.mean(res[..., :50, 1]), np.mean(res[0:30, :, :50, 1]), np.mean(res[30:60, :, :50, 1]), np.mean(res[60:90, :, :50, 1])]
        print('----------\n{}, AP0.4={:6f}, AP0.4 Seen={:6f}, AP0.4 Similar={:6f}, AP0.4 Novel={:6f}'.format(self.camera, ap_top50_0dot4[0], ap_top50_0dot4[1], ap_top50_0dot4[2], ap_top50_0dot4[3]))

        ap_top50_0dot6 = [np.mean(res[..., :50, 2]), np.mean(res[0:30, :, :50, 2]), np.mean(res[30:60, :, :50, 2]), np.mean(res[60:90, :, :50, 2])]
        print('----------\n{}, AP0.6={:6f}, AP0.6 Seen={:6f}, AP0.6 Similar={:6f}, AP0.6 Novel={:6f}'.format(self.camera, ap_top50_0dot6[0], ap_top50_0dot6[1], ap_top50_0dot6[2], ap_top50_0dot6[3]))

        ap_top50_0dot8 = [np.mean(res[..., :50, 3]), np.mean(res[0:30, :, :50, 3]), np.mean(res[30:60, :, :50, 3]), np.mean(res[60:90, :, :50, 3])]
        print('----------\n{}, AP0.8={:6f}, AP0.8 Seen={:6f}, AP0.8 Similar={:6f}, AP0.8 Novel={:6f}'.format(self.camera, ap_top50_0dot8[0], ap_top50_0dot8[1], ap_top50_0dot8[2], ap_top50_0dot8[3]))

        ap_top1 = [np.mean(res[:, :, :1, :]), np.mean(res[0:30, :, :1, :]), np.mean(res[30:60, :, :1, :]), np.mean(res[60:90, :, :1, :])]
        print('\nEvaluation Result of Top 1 Suction:\n----------\n{}, AP={:6f}, AP Seen={:6f}, AP Similar={:6f}, AP Novel={:6f}'.format(self.camera, ap_top1[0], ap_top1[1], ap_top1[2], ap_top1[3]))

        ap_top1_0dot2 = [np.mean(res[..., :1, 0]), np.mean(res[0:30, :, :1, 0]), np.mean(res[30:60, :, :1, 0]), np.mean(res[60:90, :, :1, 0])]
        print('----------\n{}, AP0.2={:6f}, AP0.2 Seen={:6f}, AP0.2 Similar={:6f}, AP0.2 Novel={:6f}'.format(self.camera, ap_top1_0dot2[0], ap_top1_0dot2[1], ap_top1_0dot2[2], ap_top1_0dot2[3]))

        ap_top1_0dot4 = [np.mean(res[..., :1, 1]), np.mean(res[0:30, :, :1, 1]), np.mean(res[30:60, :, :1, 1]), np.mean(res[60:90, :, :1, 1])]
        print('----------\n{}, AP0.4={:6f}, AP0.4 Seen={:6f}, AP0.4 Similar={:6f}, AP0.4 Novel={:6f}'.format(self.camera, ap_top1_0dot4[0], ap_top1_0dot4[1], ap_top1_0dot4[2], ap_top1_0dot4[3]))

        ap_top1_0dot6 = [np.mean(res[..., :1, 2]), np.mean(res[0:30, :, :1, 2]), np.mean(res[30:60, :, :1, 2]), np.mean(res[60:90, :, :1, 2])]
        print('----------\n{}, AP0.6={:6f}, AP0.6 Seen={:6f}, AP0.6 Similar={:6f}, AP0.6 Novel={:6f}'.format(self.camera, ap_top1_0dot6[0], ap_top1_0dot6[1], ap_top1_0dot6[2], ap_top1_0dot6[3]))

        ap_top1_0dot8 = [np.mean(res[..., :1, 3]), np.mean(res[0:30, :, :1, 3]), np.mean(res[30:60, :, :1, 3]), np.mean(res[60:90, :, :1, 3])]
        print('----------\n{}, AP0.8={:6f}, AP0.8 Seen={:6f}, AP0.8 Similar={:6f}, AP0.8 Novel={:6f}'.format(self.camera, ap_top1_0dot8[0], ap_top1_0dot8[1], ap_top1_0dot8[2], ap_top1_0dot8[3]))

        return res, ap_top50, ap_top1

    def get_precision_topk_all(self, dump_file, k=50):
        '''
        **Input:**
        
        - dump_file: string of the file that saves the evaluation results of all data splits.
        
        - k: int of the top number of suctions.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap: float of the AP of top k suctions for all split.
        
        '''

        res = np.load(dump_file)
        
        ap = [np.mean(res[:, :, :k, :]), np.mean(res[0:30, :, :k, :]), np.mean(res[30:60, :, :k, :]), np.mean(res[60:90, :, :k, :])]
        print('\nEvaluation Result of Top {} Suctions:\n----------\n{}, AP={:6f}, AP Seen={:6f}, AP Similar={:6f}, AP Novel={:6f}'.format(k, self.camera, ap[0], ap[1], ap[2], ap[3]))

        ap_0dot2 = [np.mean(res[..., :k, 0]), np.mean(res[0:30, :, :k, 0]), np.mean(res[30:60, :, :k, 0]), np.mean(res[60:90, :, :k, 0])]
        print('----------\n{}, AP0.2={:6f}, AP0.2 Seen={:6f}, AP0.2 Similar={:6f}, AP0.2 Novel={:6f}'.format(self.camera, ap_0dot2[0], ap_0dot2[1], ap_0dot2[2], ap_0dot2[3]))

        ap_0dot4 = [np.mean(res[..., :k, 1]), np.mean(res[0:30, :, :k, 1]), np.mean(res[30:60, :, :k, 1]), np.mean(res[60:90, :, :k, 1])]
        print('----------\n{}, AP0.4={:6f}, AP0.4 Seen={:6f}, AP0.4 Similar={:6f}, AP0.4 Novel={:6f}'.format(self.camera, ap_0dot4[0], ap_0dot4[1], ap_0dot4[2], ap_0dot4[3]))

        ap_0dot6 = [np.mean(res[..., :k, 2]), np.mean(res[0:30, :, :k, 2]), np.mean(res[30:60, :, :k, 2]), np.mean(res[60:90, :, :k, 2])]
        print('----------\n{}, AP0.6={:6f}, AP0.6 Seen={:6f}, AP0.6 Similar={:6f}, AP0.6 Novel={:6f}'.format(self.camera, ap_0dot6[0], ap_0dot6[1], ap_0dot6[2], ap_0dot6[3]))

        ap_0dot8 = [np.mean(res[..., :k, 3]), np.mean(res[0:30, :, :k, 3]), np.mean(res[30:60, :, :k, 3]), np.mean(res[60:90, :, :k, 3])]
        print('----------\n{}, AP0.8={:6f}, AP0.8 Seen={:6f}, AP0.8 Similar={:6f}, AP0.8 Novel={:6f}'.format(self.camera, ap_0dot8[0], ap_0dot8[1], ap_0dot8[2], ap_0dot8[3]))

        return res, ap

    def get_precision_topk_seen(self, dump_file, k=50):
        '''
        **Input:**
        
        - dump_file: string of the file that saves the evaluation results of seen data splits.
        
        - k: int of the top number of suctions.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap: float of the AP of top k suctions for seen split.
        
        '''

        res = np.load(dump_file)
        
        ap = np.mean(res[:, :, :k, :])
        print('\nEvaluation Result of Top {} Suctions:\n----------\n{}, AP Seen={:6f}'.format(k, self.camera, ap))

        ap_0dot2 = np.mean(res[..., :k, 0])
        print('----------\n{}, AP0.2 Seen={:6f}'.format(self.camera, ap_0dot2))

        ap_0dot4 = np.mean(res[..., :k, 1])
        print('----------\n{}, AP0.4 Seen={:6f}'.format(self.camera, ap_0dot4))

        ap_0dot6 = np.mean(res[..., :k, 2])
        print('----------\n{}, AP0.6 Seen={:6f}'.format(self.camera, ap_0dot6))

        ap_0dot8 = np.mean(res[..., :k, 3])
        print('----------\n{}, AP0.8 Seen={:6f}'.format(self.camera, ap_0dot8))

        return res, ap

    def get_precision_topk_similar(self, dump_file, k=50):
        
        '''
        **Input:**
        
        - dump_file: string of the file that saves the evaluation results of similar data splits.
        
        - k: int of the top number of suctions.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap: float of the AP of top k suctions for similar split.
        
        '''

        res = np.load(dump_file)
        
        ap = np.mean(res[:, :, :k, :])
        print('\nEvaluation Result of Top {} Suctions:\n----------\n{}, AP Similar={:6f}'.format(k, self.camera, ap))

        ap_0dot2 = np.mean(res[..., :k, 0])
        print('----------\n{}, AP0.2 Similar={:6f}'.format(self.camera, ap_0dot2))

        ap_0dot4 = np.mean(res[..., :k, 1])
        print('----------\n{}, AP0.4 Similar={:6f}'.format(self.camera, ap_0dot4))

        ap_0dot6 = np.mean(res[..., :k, 2])
        print('----------\n{}, AP0.6 Similar={:6f}'.format(self.camera, ap_0dot6))

        ap_0dot8 = np.mean(res[..., :k, 3])
        print('----------\n{}, AP0.8 Similar={:6f}'.format(self.camera, ap_0dot8))

        return res, ap

    def get_precision_topk_novel(self, dump_file, k=50):
        
        '''
        **Input:**
        
        - dump_file: string of the file that saves the evaluation results of novel data splits.
        
        - k: int of the top number of suctions.
        
        **Output:**
        
        - res: numpy array of the detailed accuracy.
        
        - ap: float of the AP of top k suctions for novel split.
        
        '''

        res = np.load(dump_file)
        
        ap = np.mean(res[:, :, :k, :])
        print('\nEvaluation Result of Top {} Suctions:\n----------\n{}, AP Novel={:6f}'.format(k, self.camera, ap))

        ap_0dot2 = np.mean(res[..., :k, 0])
        print('----------\n{}, AP0.2 Novel={:6f}'.format(self.camera, ap_0dot2))

        ap_0dot4 = np.mean(res[..., :k, 1])
        print('----------\n{}, AP0.4 Novel={:6f}'.format(self.camera, ap_0dot4))

        ap_0dot6 = np.mean(res[..., :k, 2])
        print('----------\n{}, AP0.6 Novel={:6f}'.format(self.camera, ap_0dot6))

        ap_0dot8 = np.mean(res[..., :k, 3])
        print('----------\n{}, AP0.8 Novel={:6f}'.format(self.camera, ap_0dot8))

        return res, ap


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', default=None, help='Dataset path. [default: None]')
    # parser.add_argument('--dense_root', default=None, help='Upsampled point cloud path. [default: None]')
    # parser.add_argument('--pred_root', default=None, help='Suction prediction path. [default: None]')
    # parser.add_argument('--save_root', default=None, help='Path to save evaluation results, if None, it will be the same with --pred_root. [default: None]')
    # parser.add_argument('--camera', default='kinect', help='Which camera data the prediction uses. [default: kinect]')
    # parser.add_argument('--split', default='test', help='Which split to perform evaluation. Could be test_seen, test_similar, test_novel or test (all the above). [default: test]')
    # args = parser.parse_args()
    
    # data_root = args.data_root
    # dense_root = args.dense_root
    # pred_root = args.pred_root
    # save_root = args.pred_root if args.save_root is None else args.save_root
    # camera = args.camera
    # split = args.split

    data_root = '/DATA2/Benchmark/graspnet'
    dense_root = '/DATA1/hanwen/grasping/models_poisson'
    pred_root = '/DATA2/Benchmark/suction/inference_results/deeplabV3plus'
    save_root = '/DATA2/Benchmark/suction/inference_results/deeplabV3plus_test_API'
    # camera = 'kinect'
    camera = 'realsense'
    split = 'test'

    
    suction_eval = SuctionNetEval(data_root, dense_root, camera, split, save_root)

    res, ap = suction_eval.eval_all(dump_folder=pred_root, proc=30)

