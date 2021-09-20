__author__ = 'hwcao'
__version__ = '1.0'

# TODO
# check_data_completeness (wrench), showObjSuction, showSceneSuction, show6DPose, loadSuctionLabels, loadSuction

# Interface for accessing the SuctionNet-1Billion dataset.
# Description and part of the codes modified from MSCOCO api

# SuctionNet is an open project for general object suction grasping that is continuously enriched.
# Currently we release SuctionNet-1Billion, a large-scale benchmark for general object suction grasping,
# as well as other related areas (e.g. 6D pose estimation, unseen object segmentation, etc.).
# suctionnetapi is a Python API that # assists in loading, parsing and visualizing the
# annotations in SuctionNet. Please visit https://graspnet.net/ for more information on SuctionNet,
# including for the data, paper, and tutorials. The exact format of the annotations
# is also described on the website. For example usage of the suctionnetapi
# please see suctionnetapi_demo.ipynb. In addition to this API, please download both
# the SuctionNet images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *grasping* and *6d pose* annotations. In the case of
# 6d poses not all functions are defined (e.g. collisions are undefined).

# The following API functions are defined:
#  SuctionNet             - SuctionNet api class that loads SuctionNet annotation file and prepare data structures.
#  getSceneIds          - Get scene ids that satisfy given filter conditions.
#  getObjIds            - Get obj ids that satisfy given filter conditions.
#  getDataIds           - Get data ids that satisfy given filter conditions.
#  loadSuctionLabels      - Load suction labels with the specified object ids.
#  loadObjModels        - Load object 3d mesh model with the specified object ids.
#  loadCollisionLabels  - Load collision labels with the specified scene ids.
#  loadSuction            - Load suction labels with the specified scene and annotation id.
#  loadData             - Load data path with the specified data ids.
#  showObjSuction         - Save visualization of the suction pose of specified object ids.
#  showSceneSuction       - Save visualization of the suction pose of specified scene ids.
#  show6DPose           - Save visualization of the 6d pose of specified scene ids, project obj models onto pointcloud
# Throughout the API "ann"=annotation, "obj"=object, and "img"=image.

# SuctionNet Toolbox.      version 1.0
# Data, paper, and tutorials available at:  https://graspnet.net/
# Code written by Hanwen Cao, 2021.
# Licensed under the none commercial CC4.0 license [see https://graspnet.net/about]

import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
import cv2
import trimesh

from suction import Suction, SuctionGroup
from utils.utils import transform_points, parse_posevector, generate_scene_model, create_table_cloud, get_model_suctions, \
    plot_sucker
from utils.xmlhandler import xmlReader
from utils.rotation import viewpoint_to_matrix

TOTAL_SCENE_NUM = 190

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class SuctionNet():
    def __init__(self, root, dense_root, camera='kinect', split='train'):
        '''
        **input**:

        - camera: string of type of camera: kinect or realsense

        - split: string of type of split of dataset: "all", "train", "test", "test_seen", "test_similar" or "test_novel"

        - sceneIDs: a list of scene ids of the dataset, split should be "user_define"
        '''

        assert camera in ['kinect', 'realsense'], 'camera should be kinect or realsense'
        assert split in ['all', 'train', 'test', 'test_seen', 'test_similar', 'test_novel'], 'split should be all/train/test/test_seen/test_similar/test_novel'
        self.root = root
        self.dense_root = dense_root
        self.camera = camera
        self.split = split
        self.collisionLabels = {}

        if split == 'all':
            self.sceneIds = list(range(TOTAL_SCENE_NUM))
        elif split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))

        self.rgbPath = []
        self.depthPath = []
        self.segLabelPath = []
        self.metaPath = []
        self.sceneName = []
        self.annId = []

        for i in tqdm(self.sceneIds, desc='Loading data path...'):
            for img_num in range(256):
                self.rgbPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.segLabelPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metaPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.sceneName.append('scene_'+str(i).zfill(4))
                self.annId.append(img_num)

        self.objIds = self.getObjIds(self.sceneIds)

    def __len__(self):
        return len(self.depthPath)

    def check_data_completeness(self):
        '''
        Check whether the dataset files are complete.

        **Output:**

        - bool, True for complete, False for not complete.
        '''
        error_flag = False
        for obj_id in tqdm(range(88), 'Checking Models'):
            if not os.path.exists(os.path.join(self.root, 'models','%03d' % obj_id, 'nontextured.ply')):
                error_flag = True
                print('No nontextured.ply For Object {}'.format(obj_id))
            if not os.path.exists(os.path.join(self.root, 'models','%03d' % obj_id, 'textured.sdf')):
                error_flag = True
                print('No textured.sdf For Object {}'.format(obj_id))
            if not os.path.exists(os.path.join(self.root, 'models','%03d' % obj_id, 'textured.obj')):
                error_flag = True
                print('No textured.obj For Object {}'.format(obj_id))
        for obj_id in tqdm(range(88), 'Checking Seal Labels'):
            if not os.path.exists(os.path.join(self.root, 'seal_label', '%03d_labels.npz' % obj_id)):
                error_flag = True
                print('No Seal Label For Object {}'.format(obj_id))
        for sceneId in self.sceneIds:
            scene_reader = xmlReader(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, 'kinect', 'annotations', '%04d.xml'% 0))
            posevectors = scene_reader.getposevectorlist()
            for posevector in posevectors:
                obj_id, _ = parse_posevector(posevector)
                if not os.path.exists(os.path.join(self.root, 'wrench_label', 'scene_%04d'%sceneId, '%03d_labels.npz' % obj_id)):
                    error_flag = True
                    print('No Wrench Label For Object {}'.format(obj_id))
        for sceneId in tqdm(self.sceneIds, 'Checking Collosion Labels'):
            if not os.path.exists(os.path.join(self.root, 'collision_label', '%04d_collision.npz' % sceneId)):
                error_flag = True
                print('No Collision Labels For Scene {}'.format(sceneId))
        for sceneId in tqdm(self.sceneIds, 'Checking Scene Datas'):
            scene_dir = os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId)
            if not os.path.exists(os.path.join(scene_dir,'object_id_list.txt')):
                error_flag = True
                print('No Object Id List For Scene {}'.format(sceneId))
            if not os.path.exists(os.path.join(scene_dir,'rs_wrt_kn.npy')):
                error_flag = True
                print('No rs_wrt_kn.npy For Scene {}'.format(sceneId))
            for camera in [self.camera]:
                camera_dir = os.path.join(scene_dir, camera)
                if not os.path.exists(os.path.join(camera_dir,'cam0_wrt_table.npy')):
                    error_flag = True
                    print('No cam0_wrt_table.npy For Scene {}, Camera:{}'.format(sceneId, camera))
                if not os.path.exists(os.path.join(camera_dir,'camera_poses.npy')):
                    error_flag = True
                    print('No camera_poses.npy For Scene {}, Camera:{}'.format(sceneId, camera)) 
                if not os.path.exists(os.path.join(camera_dir,'camK.npy')):
                    error_flag = True
                    print('No camK.npy For Scene {}, Camera:{}'.format(sceneId, camera))   
                for annId in range(256):
                    if not os.path.exists(os.path.join(camera_dir,'rgb','%04d.png' % annId)):
                        error_flag = True
                        print('No RGB Image For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'depth','%04d.png' % annId)):
                        error_flag = True
                        print('No Depth Image For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'label','%04d.png' % annId)):
                        error_flag = True
                        print('No Mask Label image For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'meta','%04d.mat' % annId)):
                        error_flag = True
                        print('No Meta Data For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'annotations','%04d.xml' % annId)):
                        error_flag = True
                        print('No Annotations For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'rect','%04d.npy' % annId)):
                        error_flag = True
                        print('No Rectangle Labels For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
        return not error_flag

    def getSceneIds(self, objIds=None):
        '''
        **Input:**

        - objIds: int or list of int of the object ids.

        **Output:**

        - a list of int of the scene ids that contains **all** the objects.
        '''
        if objIds is None:
            return self.sceneIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        sceneIds = []
        for i in self.sceneIds:
            f = open(os.path.join(self.root, 'scenes', 'scene_' + str(i).zfill(4), 'object_id_list.txt'))
            idxs = [int(line.strip()) for line in f.readlines()]
            check = all(item in idxs for item in objIds)
            if check:
                sceneIds.append(i)
        return sceneIds

    def getObjIds(self, sceneIds=None):
        '''
        **Input:**

        - sceneIds: int or list of int of the scene ids.

        **Output:**

        - a list of int of the object ids in the given scenes.
        '''
        # get object ids in the given scenes
        if sceneIds is None:
            return self.objIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        objIds = []
        for i in sceneIds:
            f = open(os.path.join(self.root, 'scenes', 'scene_' + str(i).zfill(4), 'object_id_list.txt'))
            idxs = [int(line.strip()) for line in f.readlines()]
            objIds = list(set(objIds+idxs))
        return objIds

    def getDataIds(self, sceneIds=None):
        '''
        **Input:**

        - sceneIds:int or list of int of the scenes ids.

        **Output:**

        - a list of int of the data ids. Data could be accessed by calling self.loadData(ids).
        '''
        # get index for datapath that contains the given scenes
        if sceneIds is None:
            return list(range(len(self.sceneName)))
        ids = []
        indexPosList = []
        for i in sceneIds:
            indexPosList += [ j for j in range(0,len(self.sceneName),256) if self.sceneName[j] == 'scene_'+str(i).zfill(4) ]
        for idx in indexPosList:
            ids += list(range(idx, idx+256))
        return ids

    def loadSuctionLabels(self, objIds=None):
        '''
        **Input:**

        - objIds: int or list of int of the object ids.

        **Output:**

        - a dict of grasplabels of each object. 
        '''
        # load object-level grasp labels of the given obj ids
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        graspLabels = {}
        for i in tqdm(objIds, desc='Loading grasping labels...'):
            file = np.load(os.path.join(self.root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
            graspLabels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32), file['scores'].astype(np.float32))
        return graspLabels

    def loadObjModels(self, objIds=None):
        '''
        **Function:**

        - load object 3D models of the given obj ids

        **Input:**

        - objIDs: int or list of int of the object ids

        **Output:**

        - a list of open3d.geometry.PointCloud of the models
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        models = []
        for i in tqdm(objIds, desc='Loading objects...'):
            plyfile = os.path.join(self.root, 'models','%03d' % i, 'nontextured.ply')
            models.append(o3d.io.read_point_cloud(plyfile))
        return models

    def loadObjTrimesh(self, objIds=None):
        '''
        **Function:**

        - load object 3D trimesh of the given obj ids

        **Input:**

        - objIDs: int or list of int of the object ids

        **Output:**

        - a list of rimesh.Trimesh of the models
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        models = []
        for i in tqdm(objIds, desc='Loading objects...'):
            plyfile = os.path.join(self.root, 'models','%03d' % i, 'nontextured.ply')
            models.append(trimesh.load(plyfile))
        return models

    def loadCollisionLabels(self, sceneIds=None):
        '''
        **Input:**
        
        - sceneIds: int or list of int of the scene ids.

        **Output:**

        - dict of the collision labels.
        '''
        sceneIds = self.sceneIds if sceneIds is None else sceneIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        collisionLabels = {}
        for sid in tqdm(sceneIds, desc='Loading collision labels...'):
            labels = np.load(os.path.join(self.root, 'collision_label','scene_'+str(sid).zfill(4),  'collision_labels.npz'))
            collisionLabel = []
            for j in range(len(labels)):
                collisionLabel.append(labels['arr_{}'.format(j)])
            collisionLabels['scene_'+str(sid).zfill(4)] = collisionLabel
        return collisionLabels

    def loadRGB(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the rgb in RGB order.
        '''
        return cv2.cvtColor(cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'rgb', '%04d.png' % annId)), cv2.COLOR_BGR2RGB)

    def loadBGR(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the rgb in BGR order.
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'rgb', '%04d.png' % annId))

    def loadDepth(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the depth with dtype = np.uint16
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'depth', '%04d.png' % annId), cv2.IMREAD_UNCHANGED)
 
    def loadMask(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the mask with dtype = np.uint16
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'label', '%04d.png' % annId), cv2.IMREAD_UNCHANGED)
   
    def loadWorkSpace(self, sceneId, camera, annId):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        **Output:**

        - numpy array of the workspace with dtype = np.int8
        '''
        mask = self.loadMask(sceneId, camera, annId)
        maskx = np.any(mask, axis=0)
        masky = np.any(mask, axis=1)
        x1 = np.argmax(maskx)
        y1 = np.argmax(masky)
        x2 = len(maskx) - np.argmax(maskx[::-1])
        y2 = len(masky) - np.argmax(masky[::-1]) 
        return (x1, y1, x2, y2)

    def loadScenePointCloud(self, sceneId, camera, annId, align=False, format = 'open3d'):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        - aligh: bool of whether align to the table frame.

        **Output:**

        - open3d.geometry.PointCloud instance of the scene point cloud.

        - or tuple of numpy array of point locations and colors.
        '''
        colors = self.loadRGB(sceneId = sceneId, camera = camera, annId = annId).astype(np.float32) / 255.0
        depths = self.loadDepth(sceneId = sceneId, camera = camera, annId = annId)
        intrinsics = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camK.npy'))
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        s = 1000.0
        
        if align:
            camera_poses = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
            camera_pose = camera_poses[annId]
            align_mat = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
            camera_pose = align_mat.dot(camera_pose)

        xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        points_z = depths / s
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask]
        colors = colors[mask]
        if align:
            points = transform_points(points, camera_pose)
        if format == 'open3d':
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            return cloud
        elif format == 'numpy':
            return points, colors
        else:
            raise ValueError('Format must be either "open3d" or "numpy".')

    def loadSceneModel(self, sceneId, camera = 'kinect', annId = 0, align = False):
        '''
        **Input:**

        - sceneId: int of the scene index.
        
        - camera: string of type of camera, 'realsense' or 'kinect'

        - annId: int of the annotation index.

        - align: bool of whether align to the table frame.

        **Output:**

        - open3d.geometry.PointCloud list of the scene models.
        '''
        if align:
            camera_poses = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
            camera_pose = camera_poses[annId]
            align_mat = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
            camera_pose = np.matmul(align_mat,camera_pose)
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'annotations', '%04d.xml'% annId))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        mat_list = []
        model_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, pose = parse_posevector(posevector)
            obj_list.append(obj_idx)
            mat_list.append(pose)

        for obj_idx, pose in zip(obj_list, mat_list):
            plyfile = os.path.join(self.root, 'models', '%03d'%obj_idx, 'nontextured.ply')
            model = o3d.io.read_point_cloud(plyfile)
            points = np.array(model.points)
            if align:
                pose = np.dot(camera_pose, pose)
            points = transform_points(points, pose)
            model.points = o3d.utility.Vector3dVector(points)
            model_list.append(model)
            pose_list.append(pose)
        return model_list

    def loadData(self, ids=None, *extargs):
        '''
        **Input:**

        - ids: int or list of int of the the data ids.

        - extargs: extra arguments. This function can also be called with loadData(sceneId, camera, annId)

        **Output:**

        - if ids is int, returns a tuple of data path

        - if ids is not specified or is a list, returns a tuple of data path lists
        '''
        if ids is None:
            return (self.rgbPath, self.depthPath, self.segLabelPath, self.metaPath, self.sceneName, self.annId)
        
        if len(extargs) == 0:
            if isinstance(ids, int):
                return (self.rgbPath[ids], self.depthPath[ids], self.segLabelPath[ids], self.metaPath[ids], self.sceneName[ids], self.annId[ids])
            else:
                return ([self.rgbPath[id] for id in ids],
                    [self.depthPath[id] for id in ids],
                    [self.segLabelPath[id] for id in ids],
                    [self.metaPath[id] for id in ids],
                    [self.sceneName[id] for id in ids],
                    [self.annId[id] for id in ids])
        if len(extargs) == 2:
            sceneId = ids
            camera, annId = extargs
            rgbPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'rgb', str(annId).zfill(4)+'.png')
            depthPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'depth', str(annId).zfill(4)+'.png')
            segLabelPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'label', str(annId).zfill(4)+'.png')
            metaPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'meta', str(annId).zfill(4)+'.mat')
            scene_name = 'scene_'+str(sceneId).zfill(4)
            return (rgbPath, depthPath, segLabelPath, metaPath, scene_name,annId)

    def showObjSuction(self, obj_id, visu_num):
        ply_dir = os.path.join(self.root, 'models', '%03d' % obj_id, 'nontextured.ply')
        model = o3d.io.read_point_cloud(ply_dir)
        
        radius = 0.01
        height = 0.1

        suckers = []
        
        seal_dir = os.path.join(self.root, 'seal_label', '%03d_labels.npz' % obj_id)
        sampled_points, normals, scores, _ = get_model_suctions('%s/%03d_labels.npz'%(seal_dir, obj_id))

        point_inds = np.random.choice(sampled_points.shape[0], visu_num)
        np.random.shuffle(point_inds)
        
        sucker_params = []

        for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            normal = normals[point_ind]
            score = scores[point_ind]

            R = viewpoint_to_matrix(normal)
            t = target_point

            sucker = plot_sucker(R, t, score, radius, height)
            suckers.append(sucker)
            sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
                
        o3d.visualization.draw_geometries([model, *suckers], width=1536, height=864)

    def showSceneSuction(self, scene_idx, anno_idx, camera, visu_num_each):
        scene_name = 'scene_%04d' % scene_idx
        model_list, obj_list, pose_list = generate_scene_model(self.root, scene_name, anno_idx, return_poses=True, camera=camera, align=True)
        table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)

        camera_poses = np.load(os.path.join(self.root, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
        camera_pose = camera_poses[anno_idx]
        table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
        
        collision_dir = os.path.join(self.root, 'collision_label')
        collision_dump = np.load(os.path.join(collision_dir, '{}_{}_collision.npz'.format(scene_idx, camera)))

        radius = 0.01
        height = 0.1

        num_obj = len(obj_list)
        
        for obj_i in range(len(obj_list)):
            suckers = []
            print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
            obj_idx = obj_list[obj_i]
            trans = pose_list[obj_i]
            seal_dir = os.path.join(self.root, 'seal_label', '%03d_labels.npz' % obj_i)
            sampled_points, normals, scores, _ = get_model_suctions('%s/%03d_labels.npz'%(seal_dir, obj_idx))
            collisions = collision_dump['arr_{}'.format(obj_i)]

            point_inds = np.random.choice(sampled_points.shape[0], visu_num_each)
            np.random.shuffle(point_inds)
            
            sucker_params = []

            for point_ind in point_inds:
                target_point = sampled_points[point_ind]
                normal = normals[point_ind]
                score = scores[point_ind]
                collision = collisions[point_ind]

                R = viewpoint_to_matrix(normal)
                t = transform_points(target_point[np.newaxis,:], trans).squeeze()
                R = np.dot(trans[:3,:3], R)
                sucker = plot_sucker(R, t, score * float(~bool(collision)), radius, height)
                suckers.append(sucker)
                sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
                
            o3d.visualization.draw_geometries([table, *model_list, *suckers], width=1536, height=864)

    def show6DPose(self, scene_idx, anno_idx, camera, visu_num):
        scene_name = 'scene_%04d' % scene_idx
        model_list, _, _ = generate_scene_model(self.root, scene_name, anno_idx, return_poses=True, camera=camera, align=True)
        table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)

        camera_poses = np.load(os.path.join(self.root, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
        camera_pose = camera_poses[anno_idx]
        table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
        
        o3d.visualization.draw_geometries([table, *model_list], width=1536, height=864)
    