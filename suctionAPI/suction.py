__author__ = 'hwcao'
__version__ = '1.0'

import numpy as np
import open3d as o3d
import copy
import cv2

from utils.utils import plot_sucker, batch_rgbdxyz_2_rgbxy_depth, framexy_depth_2_xyz, batch_framexy_depth_2_xyz, key_point_2_rotation, batch_framexy_depth_2_xyz, batch_key_point_2_rotation

SUCTION_ARRAY_LEN = 8
EPS = 1e-8

class Suction():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the score, direction, translation, object_id

        - the format of numpy array is [score, direction(3), translation(3), object_id]

        - the length of the numpy array is 8.
        '''
        if len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.suction_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == 4:
            score, direction, translation, object_id = args
            self.suction_array = np.concatenate([np.array((score)),direction, translation, np.array((object_id)).reshape(-1)]).astype(np.float32)
        else:
            raise ValueError('only 1 or 4 arguments are accepted')
    
    def __repr__(self):
        return 'Suction: score:{}, translation:{}\ndirection:\n{}\nobject id:{}'.format(self.score(), self.translation(), self.direction(), self.object_id())

    def score(self):
        '''
        **Output:**

        - float of the score.
        '''
        return float(self.suction_array[0])


    def direction(self):
        '''
        **Output:**

        - np.array of shape (3, 3) of the rotation matrix.
        '''
        return self.suction_array[1:4]

    def translation(self):
        '''
        **Output:**

        - np.array of shape (3,) of the translation.
        '''
        return self.suction_array[4:7]

    def object_id(self):
        '''
        **Output:**

        - int of the object id that this suction suctions
        '''
        return int(self.suction_array[7])

    def to_open3d_geometry(self):
        '''
        **Ouput:**

        - list of open3d.geometry.Geometry of the gripper.
        '''
        return plot_sucker(R=self.direction(), t=self.translation(), score=self.score())

class SuctionGroup():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be (1) nothing (2) numpy array of suction group array.
        '''
        if len(args) == 0:
            self.suction_group_array = np.zeros((0, SUCTION_ARRAY_LEN), dtype=np.float32)
        elif len(args) == 1:
            # suction_list = args
            self.suction_group_array = args[0]
            # self.suction_group_array = np.zeros((0, SUCTION_ARRAY_LEN), dtype=np.float32)
            # for suction in suction_list:
            #     self.suction_group_array = np.concatenate((self.suction_group_array, suction.suction_array.reshape((-1, SUCTION_ARRAY_LEN))))
        else:
            raise ValueError('args must be nothing or list of Suction instances.')

    def __len__(self):
        '''
        **Output:**

        - int of the length.
        '''
        return len(self.suction_group_array)

    def __repr__(self):
        repr = '----------\nSuction Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for suction_array in self.suction_group_array:
                repr += Suction(suction_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += Suction(self.suction_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += Suction(self.suction_group_array[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        **Input:**

        - index: int or slice.

        **Output:**

        - if index is int, return Suction instance.

        - if index is slice, return SuctionGroup instance.
        '''
        if type(index) == int:
            return Suction(self.suction_group_array[index])
        elif type(index) == slice:
            suctiongroup = SuctionGroup()
            suctiongroup.suction_group_array = copy.deepcopy(self.suction_group_array[index])
            return suctiongroup
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for SuctionGroup'.format(type(index)))

    def scores(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the scores.
        '''
        return self.suction_group_array[:,0]

    def directions(self):
        '''
        **Output:**

        - np.array of shape (-1, 3, 3) of the rotation matrices.
        '''
        return self.suction_group_array[:, 1:4]

    def translations(self):
        '''
        **Output:**

        - np.array of shape (-1, 3) of the translations.
        '''
        return self.suction_group_array[:, 4:7]

    def object_ids(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the object ids.
        '''
        return self.suction_group_array[:,7].astype(np.int32)

    def add(self, suction):
        '''
        **Input:**

        - suction: Suction instance
        '''
        self.suction_group_array = np.concatenate((self.suction_group_array, suction.suction_array.reshape((-1, SUCTION_ARRAY_LEN))))
        return self

    def remove(self, index):
        '''
        **Input:**

        - index: list of the index of suction
        '''
        self.suction_group_array = np.delete(self.suction_group_array, index, axis = 0)
        return self

    def from_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        if npy_file_path[-3:] == 'npz':
            self.suction_group_array = np.load(npy_file_path)['arr_0']
        else:
            self.suction_group_array = np.load(npy_file_path)
        return self

    def save_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        np.save(npy_file_path, self.suction_group_array)

    def to_open3d_geometry_list(self):
        '''
        **Output:**

        - list of open3d.geometry.Geometry of the grippers.
        '''
        geometry = []
        for i in range(len(self.suction_group_array)):
            g = Suction(self.suction_group_array[i])
            geometry.append(g.to_open3d_geometry())
        return geometry
    
    def sort_by_score(self, reverse = False):
        '''
        **Input:**

        - reverse: bool of order, if True, from low to high, if False, from high to low.

        **Output:**

        - no output but sort the suction group.
        '''
        score = self.suction_group_array[:,0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.suction_group_array = self.suction_group_array[index]
        return self

    def random_sample(self, numSuction = 20):
        '''
        **Input:**

        - numSuction: int of the number of sampled suctions.

        **Output:**

        - SuctionGroup instance of sample suctions.
        '''
        if numSuction > self.__len__():
            raise ValueError('Number of sampled suction should be no more than the total number of suctions in the group')
        shuffled_suction_group_array = copy.deepcopy(self.suction_group_array)
        np.random.shuffle(shuffled_suction_group_array)
        shuffled_suction_group = SuctionGroup()
        shuffled_suction_group.suction_group_array = copy.deepcopy(shuffled_suction_group_array[:numSuction])
        return shuffled_suction_group

    def nms(self, translation_thresh = 0.1, rotation_thresh = 30.0 / 180.0 * np.pi):
        from grasp_nms import nms_grasp
        return SuctionGroup(nms_grasp(self.suction_group_array, translation_thresh, rotation_thresh))

