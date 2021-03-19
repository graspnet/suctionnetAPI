import os
import numpy as np


def get_precision():
    camera = 'kinect'
    # camera = 'realsense'

    # dump_dir = '/DATA2/Benchmark/suction/inference_results/normals_std/accuracy'
    # dump_dir = '/DATA2/Benchmark/suction/inference_results/convnet/accuracy'
    dump_dir = '/DATA2/Benchmark/suction/inference_results/deeplabV3plus_v5_scratch_epoch90/accuracy'
    # dump_dir = '/DATA2/Benchmark/suction/inference_results/dexnet3.0/accuracy'
    
    dump_file = os.path.join(dump_dir, camera+'.npy')
    res = np.load(dump_file)
    
    # print('AP:', res[30:60, ...].mean())
    
    # print('res:', res.shape)
    ap = [np.mean(res), np.mean(res[0:30]), np.mean(res[30:60]), np.mean(res[60:90])]
    print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 0]), np.mean(res[0:30, :, :, 0]), np.mean(res[30:60, :, :, 0]), np.mean(res[60:90, :, :, 0])]
    print('----------\n{}, AP0.2={}, AP0.2 Seen={}, AP0.2 Similar={}, AP0.2 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 1]), np.mean(res[0:30, :, :, 1]), np.mean(res[30:60, :, :, 1]), np.mean(res[60:90, :, :, 1])]
    print('----------\n{}, AP0.4={}, AP0.4 Seen={}, AP0.4 Similar={}, AP0.4 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 2]), np.mean(res[0:30, :, :, 2]), np.mean(res[30:60, :, :, 2]), np.mean(res[60:90, :, :, 2])]
    print('----------\n{}, AP0.6={}, AP0.6 Seen={}, AP0.6 Similar={}, AP0.6 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 3]), np.mean(res[0:30, :, :, 3]), np.mean(res[30:60, :, :, 3]), np.mean(res[60:90, :, :, 3])]
    print('----------\n{}, AP0.8={}, AP0.8 Seen={}, AP0.8 Similar={}, AP0.8 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

def get_precision_2third():
    camera = 'kinect'
    # camera = 'realsense'

    # dump_dir = '/DATA2/Benchmark/suction/inference_results/normals_std/accuracy'
    # dump_dir = '/DATA2/Benchmark/suction/inference_results/convnet/accuracy'
    # dump_dir = '/DATA2/Benchmark/suction/inference_results/deeplabV3plus/accuracy'
    dump_dir = '/DATA2/Benchmark/suction/inference_results/dexnet_it_0/accuracy'
    
    dump_file = os.path.join(dump_dir, camera+'_2third.npy')
    res = np.load(dump_file)
    print('res:', res.shape)
    # print('AP:', res[30:60, ...].mean())
    
    ap = [np.mean(res), np.mean(res[0:20]), np.mean(res[20:40]), np.mean(res[40:60])]
    print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 0]), np.mean(res[0:20, :, :, 0]), np.mean(res[20:40, :, :, 0]), np.mean(res[40:60, :, :, 0])]
    print('----------\n{}, AP0.2={}, AP0.2 Seen={}, AP0.2 Similar={}, AP0.2 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 1]), np.mean(res[0:20, :, :, 1]), np.mean(res[20:40, :, :, 1]), np.mean(res[40:60, :, :, 1])]
    print('----------\n{}, AP0.4={}, AP0.4 Seen={}, AP0.4 Similar={}, AP0.4 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 2]), np.mean(res[0:20, :, :, 2]), np.mean(res[20:40, :, :, 2]), np.mean(res[40:60, :, :, 2])]
    print('----------\n{}, AP0.6={}, AP0.6 Seen={}, AP0.6 Similar={}, AP0.6 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 3]), np.mean(res[0:20, :, :, 3]), np.mean(res[20:40, :, :, 3]), np.mean(res[40:60, :, :, 3])]
    print('----------\n{}, AP0.8={}, AP0.8 Seen={}, AP0.8 Similar={}, AP0.8 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

def get_precision_half():
    # camera = 'kinect'
    camera = 'realsense'

    # dump_dir = '/DATA2/Benchmark/suction/inference_results/normals_std/accuracy'
    # dump_dir = '/DATA2/Benchmark/suction/inference_results/convnet/accuracy'
    dump_dir = '/DATA2/Benchmark/suction/inference_results/dexnet_it_0/accuracy'
    # dump_dir = '/DATA2/Benchmark/suction/inference_results/deeplabV3plus_epoch50/accuracy'
    
    dump_file = os.path.join(dump_dir, camera+'_half.npy')
    res = np.load(dump_file)
    
    # print('AP:', res[30:60, ...].mean())
    
    print('res:', res.shape)
    ap = [np.mean(res), np.mean(res[0:15]), np.mean(res[15:30]), np.mean(res[30:45])]
    print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 0]), np.mean(res[0:15, :, :, 0]), np.mean(res[15:30, :, :, 0]), np.mean(res[30:45, :, :, 0])]
    print('----------\n{}, AP0.2={}, AP0.2 Seen={}, AP0.2 Similar={}, AP0.2 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 1]), np.mean(res[0:15, :, :, 1]), np.mean(res[15:30, :, :, 1]), np.mean(res[30:45, :, :, 1])]
    print('----------\n{}, AP0.4={}, AP0.4 Seen={}, AP0.4 Similar={}, AP0.4 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 2]), np.mean(res[0:15, :, :, 2]), np.mean(res[15:30, :, :, 2]), np.mean(res[30:45, :, :, 2])]
    print('----------\n{}, AP0.6={}, AP0.6 Seen={}, AP0.6 Similar={}, AP0.6 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

    ap = [np.mean(res[..., 3]), np.mean(res[0:15, :, :, 3]), np.mean(res[15:30, :, :, 3]), np.mean(res[30:45, :, :, 3])]
    print('----------\n{}, AP0.8={}, AP0.8 Seen={}, AP0.8 Similar={}, AP0.8 Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))

if __name__ == "__main__":
    # get_precision_2third()
    get_precision()
    # get_precision_half()    
