import numpy as np
import open3d as o3d
import argparse
import point_cloud_utils as pcu
import os


# create dense object point clouds from mesh using Poisson Disk Sampling. https://github.com/fwilliams/point-cloud-utils
parser = argparse.ArgumentParser()
parser.add_argument('--model_root', default='', help='Directory of graspnet models')
parser.add_argument('--save_root', default='', help='Directory to save the created point clouds')
args = parser.parse_args()


if __name__ == "__main__":
    model_root = args.model_root
    save_root = args.save_root

    models = []
    for i in range(88):
        models.append('%03d' % i)

    for model in models:
        mesh_dir = os.path.join(model_root, model, 'textured.obj')
        save_dir = os.path.join(save_root, model)
        os.makedirs(save_dir, exist_ok=True)
        
        print('Reading mesh from:', mesh_dir)
        mesh = o3d.io.read_triangle_mesh(mesh_dir)
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        n = np.asarray(mesh.vertex_normals)
        
        v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(v, f, n, num_samples=-1, radius=0.0002, use_geodesic_distance=True)

        points_file = os.path.join(save_dir, 'points')
        normals_file = os.path.join(save_dir, 'normals')
        print('Saving points to:', points_file)
        np.save(points_file, v_poisson)
        print('Saving normals to:', normals_file)
        np.save(normals_file, n_poisson)
