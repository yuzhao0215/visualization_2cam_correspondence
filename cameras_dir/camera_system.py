'''
@ This file defines the camera basic class and the camera system class
@ Camera basic class contains:
    focal length: fx, fy
    principal point: px, py
    rotation matrix: Ri
    transformation vector: Ti
    intrinsic matrix: Ki
    extrinsic matrix: Ei (modify later)
    distortion matrix: Di

@ Camera system class contains:
    number of cameras: N
    number of camera pairs: N_P
    fundamental matrixes: Fs
    epipolar error: E_epipolar
    reprojection error: E_reprojection
'''

import numpy as np
import cv2 as cv

import warnings
import ruamel.yaml as yaml


def read_pairs(pairs_file):
    files = []
    with open(pairs_file, 'r') as stream:
        pairs_dic = yaml.load_all(stream)
        # pairs_dic = yaml.load_all(stream, Loader=yaml.RoundTripLoader)

        for pair in pairs_dic:
            files.append(pair)
    return files


def read_cameras(cameras_file):
    files = []
    with open(cameras_file, 'r') as stream:
        pairs_dic = yaml.load_all(stream)
        # pairs_dic = yaml.load_all(stream, Loader=yaml.RoundTripLoader)

        for pair in pairs_dic:
            files.append(pair)
    return files

warnings.filterwarnings("ignore")


def correct_hand(r):
    flip_y = np.eye(3)
    flip_y[0, :] = flip_y[0, :] * (-1)
    return np.dot(flip_y, r)

# This is a basic class of camera with a constructor can take 'read_camera' as a input parameter,
# where 'read_camera' is a dictionary from yaml reading
class camera_():
    def __init__(self, *args, **kwargs):
        if 'read_camera' in kwargs:
            self.R = np.array(kwargs['read_camera']['R']).reshape((3, 3))
            self.r_vec_u = np.array(kwargs['read_camera']['r_vec_u']).reshape((-1, 1))
            self.T = np.array(kwargs['read_camera']['T']).reshape((-1, 1))
            self.f = kwargs['read_camera']['f'][0]  # here assume focul length of x and y are the same
            self.c = np.array(kwargs['read_camera']['c']).reshape((-1, 1))
            self.distortion = np.array(kwargs['read_camera']['distortion']).reshape((-1, 1))
            self.sensor_size = np.array(kwargs['read_camera']['sensor_size']).reshape((-1, 1))
            self.pixel_size = np.array(kwargs['read_camera']['pixel_size']).reshape((-1, 1))
        else:
            self.R = np.eye(3)
            self.r_vec_u = np.ones((3, 1))  # maybe not correct, modify later
            self.T = np.zeros((3, 1))  # similar above
            self.f = 1  # normalized focal length
            self.c = np.zeros((2, 1))
            self.distortion = np.zeros((4, 1))
            self.sensor_size = np.array([1280, 1024]).reshape((2, 1)) * 0.001
            self.pixel_size = np.ones(2).reshape((2, 1)) * 0.001
        self.cameraMatrix = np.array([self.f, 0, self.c[0, 0], 0, self.f, self.c[1, 0], 0, 0, 1]).reshape(
            (3, 3))  # ???right
        self.rvec, jac = cv.Rodrigues(self.R)

    def init_camera_matrix(self, c):
        self.cameraMatrix = np.array([self.f, 0, c[0, 0], 0, self.f, c[1, 0], 0, 0, 1]).reshape((3, 3))  # ???right


class camera_system_():
    def __init__(self, *args, **kwargs):
        if 'read_pairs' in kwargs:
            self.pairs = {}
            self.original_pairs = []

            dic = kwargs['read_pairs']

            self.num_pairs = len(dic)

            print("There num of pairs is: ", self.num_pairs)

            for p in dic:
                id = str(p["Camera_A_id"]) + str(p["Camera_B_id"])
                self.pairs[id] = camera_pair_(pair=p)
                self.original_pairs.append(p)

        if 'S' in kwargs:
            self.S = np.loadtxt(kwargs['S'])
        else:
            self.S = np.eye(3)
        if 'P' in kwargs:
            self.P = np.loadtxt(kwargs['P']).reshape((3, 1))
        else:
            self.P = np.array([0, 0, 0]).reshape((3, 1))
        if 'cameras' in kwargs:
            self.cameras = []
            self.original_cameras = []

            self.num_cameras = len(kwargs['cameras'])
            for c in kwargs['cameras']:
                self.cameras.append(camera_(read_camera=c))
                self.original_cameras.append(c)
        if 'correct_hand' in kwargs:
            if kwargs['correct_hand']:
                for camera in self.cameras:
                    camera.R = correct_hand(camera.R)
                    camera.T = correct_hand(camera.T)
                print("Camera rotation matrix and translation vector is corrected to left hand\n")
            else:
                print("Camera rotation matrix and translation vector is in right hand\n")
        if len(kwargs) == 0:  # right??????? check later
            print("There is no dictoriay read into constructor, exiting\n")

        self.correct_hand = False
        self.calibrate_data_matches = object()
        self.calibrate_data_original = object()

        self.calibrate_data_reprojection = object()
        self.matchList = object()
        self.particles3d = object()
        self.Ps = [None for _ in range(len(self.cameras))]

    def reconstruct3D_lsqt(self, matchList):
        ret_3d = []
        for index, match in enumerate(matchList):
            coords = np.array(match.matches).reshape((-1, 3))

            A = []
            B = []

            for coord in coords:
                camera_index = int(coord[-1])
                R = self.cameras[camera_index].R
                T = self.cameras[camera_index].T
                K = self.cameras[camera_index].cameraMatrix

                cx = K[0, 2]
                cy = K[1, 2]
                f = K[0, 0]

                x = coord[0]
                y = coord[1]

                x = (x - cx) / f
                y = (y - cy) / f

                Ai = [[x * R[2, 0] - R[0, 0],
                       x * R[2, 1] - R[0, 1],
                       x * R[2, 2] - R[0, 2]],
                      [y * R[2, 0] - R[1, 0],
                       y * R[2, 1] - R[1, 1],
                       y * R[2, 2] - R[1, 2]]]

                Bi = [T[0] - x * T[2], T[1] - y * T[2]]

                A.append(Ai)
                B.append(Bi)

            As = np.stack(A).reshape((-1, 3))
            Bs = np.stack(B).reshape((-1, 1))

            Xs, r, _, _ = np.linalg.lstsq(As, Bs)
            Xs = Xs.reshape(-1, 3)
            ret_3d.append(Xs)

        return np.stack(ret_3d).reshape((-1, 3))

    def getProjectionMatrixs(self):
        for i, camera in enumerate(self.cameras):
            R = camera.R
            T = camera.T
            RT = np.c_[R, T]
            K = camera.cameraMatrix
            P = np.dot(K, RT)
            self.Ps[i] = P

    def generate_pairs(self):
        for i in range(len(self.cameras[:-1])):
            for j in range(i + 1, len(self.cameras)):
                id = str(i) + str(j)
                pair = camera_pair_()
                pair.cameraAid = i
                pair.cameraBid = j
                pair.fmatrix = self.calculate_fmatrix(base_camera_num=i, target_camera_num=j)
                self.pairs[id] = pair

    def calculate_fmatrix(self, base_camera_num=0, target_camera_num=1):
        base_camera = self.cameras[base_camera_num]
        target_camera = self.cameras[target_camera_num]

        base_r = np.linalg.inv(base_camera.R)  # right? The result is the same with base_camera.R.T
        base_t = base_camera.T

        t = target_camera.T - np.dot(target_camera.R, np.dot(base_r, base_t))
        r = np.dot(target_camera.R, base_r)
        temp = -np.dot(r.T, t)
        # t = target_camera.T - base_t

        tx = np.array([0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0]).astype(float).reshape((3, 3))

        e = np.dot(tx, r)
        base_k = base_camera.cameraMatrix
        target_k = target_camera.cameraMatrix
        f = np.dot(np.linalg.inv(target_k).T, np.dot(e, np.linalg.inv(base_k)))
        f = f / f[2, 2]
        return f

    def reproject_to_cam(self, particle3d, camera_index):
        rvec = self.cameras[camera_index].rvec.copy().ravel()
        T = self.cameras[camera_index].T.copy().ravel()
        K = self.cameras[camera_index].cameraMatrix.copy()

        distortion = self.cameras[camera_index].distortion.copy().ravel()

        original_reprojected_points = cv.projectPoints(particle3d,
                                                       rvec, T, K,
                                                       distortion, None)[0].reshape((-1, 2))
        return original_reprojected_points

    def reprojection(self, particles3d, match_list):
        if particles3d.shape[0] is not len(match_list):
            print("Number of 3d particles not equal to match_list length")
            return

        for i in range(particles3d.shape[0]):
            particle3d = particles3d[i].reshape(1, 3)
            matches = match_list[i].matches

            for match in matches:
                camera_index = int(match[2])
                x = match[0]
                y = match[1]

                # rvec = self.cameras[camera_index].rvec.copy().ravel()
                # T = self.cameras[camera_index].T.copy().ravel()
                # K = self.cameras[camera_index].cameraMatrix.copy()
                #
                # distortion = self.cameras[camera_index].distortion.copy().ravel()
                #
                # original_reprojected_points = cv.projectPoints(particle3d,
                #                                                rvec, T, K,
                #                                                distortion, None)[0].reshape((-1, 2))

                original_reprojected_points = self.reproject_to_cam(particle3d, camera_index)


class camera_pair_():
    def __init__(self, *args, **kwargs):
        if 'pair' in kwargs:
            dic = kwargs['pair']
            self.cameraAid = dic['Camera_A_id']
            self.cameraBid = dic['Camera_B_id']
            self.fmatrix = np.array(dic['FundamentalMatrix']).reshape((3, 3))
            self.epipolar_error = dic['epipolar_error']
        else:
            pass
            # print("no dic is read into pair constructor, existing\n")


if __name__ == '__main__':
    folder = './camera_parameters'

    pairs_file = folder + '/pairs_calibrated.yaml'
    cameras_file = folder + '/cameras_calibrated.yaml'

    s_file = folder + '/S.txt'
    p_file = folder + '/P.txt'

    """
    Parameters for correspondence
    """
    max_particles_per_img = 1500
    maximum_match_candidates = 500
    epipolar_threshold = 1

    distortion = False

    cameras = read_cameras(cameras_file)
    camera_pairs = read_pairs(pairs_file)
    camera_sys = camera_system_(read_pairs=camera_pairs, cameras=cameras, S=s_file, P=p_file)

