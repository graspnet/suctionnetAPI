import os
import numpy as np
import open3d as o3d
from utils.xmlhandler import xmlReader
from transforms3d.euler import euler2mat, quat2mat


root = r'G:\MyProject\data\Grasping\graspnet'

dense_root = r'G:\MyProject\data\Grasping\graspnet\models_poisson'


def get_scene_name(num):
    return ('scene_%04d' % (num,))

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

def get_model_poses(scene_id, ann_id):
    '''
        pose_list: object pose from model to camera coordinate
        camera_pose: from camera to world coordinate
        align_mat: from world to table-horizontal coordinate
    '''
    scene_dir = os.path.join(root, 'scenes')
    camera_poses_path = os.path.join(root, 'scenes', get_scene_name(scene_id), camera, 'camera_poses.npy')
    camera_poses = np.load(camera_poses_path)
    camera_pose = camera_poses[ann_id]
    align_mat_path = os.path.join(root, 'scenes', get_scene_name(scene_id), camera, 'cam0_wrt_table.npy')
    align_mat = np.load(align_mat_path)
    # print('Scene {}, {}'.format(scene_id, camera))
    scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), camera, 'annotations', '%04d.xml'% (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)
    return obj_list, pose_list, camera_pose, align_mat

def get_scene_models(scene_id, ann_id):
    '''
        return models in model coordinate
    '''
    model_dir = os.path.join(root, 'models')
    # print('Scene {}, {}'.format(scene_id, camera))
    scene_reader = xmlReader(os.path.join(root, 'scenes', get_scene_name(scene_id), camera, 'annotations', '%04d.xml' % (ann_id,)))
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

        # poisson_dir = os.path.join(dense_root, '%03d'%obj_idx + '.npz')     
        # data = np.load(poisson_dir)
        # points = data['points']
        # dense_model_list.append(points)
        
    return model_list, dense_model_list, obj_list


def visu_normals(scene_id, dump_folder, split, camera):
    visu_num = 50

    model_list, dense_model_list, _ = get_scene_models(scene_id, ann_id=0)
    # table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.01)
    
    for ann_id in range(3):
        if os.path.exists(os.path.join(dump_folder, split, 'scene_%04d'%scene_id, camera, 'suction', '%04d.npy' % (ann_id))):
            grasp_group = np.load(os.path.join(dump_folder, split, 'scene_%04d'%scene_id, camera, 'suction', '%04d.npy' % (ann_id)))
        else:
            grasp_group = np.load(os.path.join(dump_folder, split, 'scene_%04d'%scene_id, camera, 'suction', '%04d.npz' % (ann_id)))['arr_0']
        suction_points = grasp_group[:, 4:7]
        
        suction_normals = grasp_group[:, 1:4]

        # suction_index = np.random.choice(suction_points.shape[0], visu_num)
        # suction_points = suction_points[suction_index]
        # suction_normals = suction_normals[suction_index]

        sort_index = np.argsort(-grasp_group[..., 0])
        suction_points = suction_points[sort_index]
        suction_normals = suction_normals[sort_index]
        suction_points = suction_points[:visu_num, :]
        suction_normals = suction_normals[:visu_num, :]
        print('suction_points:', suction_points.shape)
        # suction_points = suction_points[:, [2, 0, 1]]
        # suction_normals = suction_normals[:, [2, 0, 1]]
        _, pose_list, camera_pose, align_mat = get_model_poses(scene_id, ann_id)
        # table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        # table_trans = transform_points(table, np.matmul(align_mat, camera_pose))
        # table_trans = table
        # o3d_table = o3d.geometry.PointCloud()
        # o3d_table.points = o3d.utility.Vector3dVector(table_trans)
        model_trans_list = list()
        o3d_model_list = list()
        for i in range(len(model_list)):
            model = model_list[i]
            model_trans = transform_points(model, pose_list[i])
            model_trans_list.append(model_trans)

            o3d_model = o3d.geometry.PointCloud()
            o3d_model.points = o3d.utility.Vector3dVector(model_trans)
            o3d_model_list.append(o3d_model)

        arrow_list = []

        for idx in range(len(suction_points)):

            suction_point = suction_points[idx]
            print(suction_point)
            suction_normal = suction_normals[idx]
            suction_normal = suction_normal / (np.linalg.norm(suction_normal)+1e-8)
            
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.0075, 
                                                            cylinder_height=0.025, cone_height=0.02)
            arrow_points = np.asarray(arrow.vertices)

            new_z = suction_normal
            new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
            if np.sum(new_y!=0) == 0:
                new_y = np.array((0, 1, 0), dtype=np.float64)
            new_y = new_y / np.linalg.norm(new_y)
            new_x = np.cross(new_y, new_z)

            R = np.c_[new_x, np.c_[new_y, new_z]]
            arrow_points = np.dot(R, arrow_points.T).T + suction_point[np.newaxis,:]
            arrow.vertices = o3d.utility.Vector3dVector(arrow_points)

            arrow_list.append(arrow)
        
        o3d.visualization.draw_geometries([*o3d_model_list, *arrow_list], width=1280, height=720)


if __name__ == "__main__":
    scene_id = 132
    # split = 'test_seen'
    split = 'test_similiar'

    camera = 'kinect'
    # camera = 'realsense'

    # dump_folder = r'G:\MyProject\data\Grasping\network_results\deeplabV3plus'
    dump_folder = r'G:\MyProject\data\Grasping\network_results\dexnet_it_0'
    visu_normals(scene_id, dump_folder, split, camera)
