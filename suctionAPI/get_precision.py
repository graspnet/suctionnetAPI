import os
import argparse
from pickle import dump
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', default='', help='Directory of evaluation results')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument("--top_k", type=int, default=50, help='Num of top suctions to consider (50 or 1) [default: 50]')
args = parser.parse_args()

dump_dir = args.dump_dir
camera = args.camera
k = args.top_k

def get_precision_topk():
    
    dump_file = os.path.join(dump_dir, camera+'.npy')
    res = np.load(dump_file)
    
    ap = [np.mean(res[:, :, :k, :]), np.mean(res[0:30, :, :k, :]), np.mean(res[30:60, :, :k, :]), np.mean(res[60:90, :, :k, :])]
    print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., :k, 0]), np.mean(res[0:30, :, :k, 0]), np.mean(res[30:60, :, :k, 0]), np.mean(res[60:90, :, :k, 0])]
    print('----------\n{}, AP0.2={}, AP0.2 Seen={}, AP0.2 Similar={}, AP0.2 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., :k, 1]), np.mean(res[0:30, :, :k, 1]), np.mean(res[30:60, :, :k, 1]), np.mean(res[60:90, :, :k, 1])]
    print('----------\n{}, AP0.4={}, AP0.4 Seen={}, AP0.4 Similar={}, AP0.4 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., :k, 2]), np.mean(res[0:30, :, :k, 2]), np.mean(res[30:60, :, :k, 2]), np.mean(res[60:90, :, :k, 2])]
    print('----------\n{}, AP0.6={}, AP0.6 Seen={}, AP0.6 Similar={}, AP0.6 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., :k, 3]), np.mean(res[0:30, :, :k, 3]), np.mean(res[30:60, :, :k, 3]), np.mean(res[60:90, :, :k, 3])]
    print('----------\n{}, AP0.8={}, AP0.8 Seen={}, AP0.8 Similar={}, AP0.8 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))


if __name__ == "__main__":

    get_precision_topk()    
