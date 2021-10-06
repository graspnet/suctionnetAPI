import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import os


def create_dense_point_cloud(model_root, save_root):
    '''
    **Input:**
    
    - save_root: string of the directory to save the created dense point cloud. 

    **Output:**
    
    - No output but the created dense point clouds will be saved in the save_root.
    '''

    models = []
    for i in range(88):
        models.append('%03d' % i)

    for model in models:
        mesh_dir = os.path.join(model_root, model, 'textured.obj')
        save_dir = os.path.join(save_root, model)
        os.makedirs(save_dir, exist_ok=True)
        
        print('Read mesh from:', mesh_dir)
        mesh = o3d.io.read_triangle_mesh(mesh_dir)
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        n = np.asarray(mesh.vertex_normals)
        
        v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(v, f, n, num_samples=-1, radius=0.0002, use_geodesic_distance=True)

        save_file = os.path.join(save_dir, model+'.npz')
        np.savez(save_file, points=v_poisson, normals=n_poisson) 
        
