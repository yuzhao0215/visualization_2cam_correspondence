"""
This file contains the functions to operate real recognition of particle pairs on actual recorded images.

The images folder is "./data/images/".

This folder contains several sub-folders. Each sub-folder contains images recorded by eight cameras contains
different number of particles. For example, "./data/images/50/" contains eight images, where there are 50 particles in
each image. The file "./data/images/50/xyz_50.txt" contains the 3D coordinates of all particles.
"""

# import cv2 as cv
from PIL import Image
import glob
import numpy as np
from cameras_dir.camera_system import *
from pickle import load
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
import plotly.graph_objects as go

'''
This function display a row of two pictures. On the left side is the image captured by camera A. On the right side is
the image captured by camera B. For a particle p in camera A, the epipolar line will be drawn on the right image. The 
particles in right image that have distances to the epipolar line smaller than a threshold will be considered as particle 
candidates. The features of each pair (particle p and particle candidate in camera B) will be extracted and passed to 
classifier to determine if it is a true correspondence.
'''

'''
Base class containing particle image features:
    xc:     x pixel coordinate of particle image center
    yc:     y pixel coordinate of particle image centerß
    area:   pixel area of particle image
    r:      radius of particle image's enclosing circle
    rou:    roundness of particle image segment
    peri:   perimeter of particle image segment
    I:      intensity of Image[xc, yc]
    MI:     max intensity of pixels around Image[xc-3:xc+3, yc-3:yc+3]
    MEI:    mean intensity of pixels around Image[xc-3:xc+3, yc-3:yc+3]
    MI_SHI_x: pixel shift in x direction from max intensity pixel to [xc, yc]
    MI_SHI_y: pixel shift in y direction from max intensity pixel to [xc, yc]
'''


class ParticleImage:
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.area = 0
        self.r = 0
        self.rou = 0
        self.peri = 0
        self.I = 0
        self.MI = 0
        self.MEI = 0
        self.MI_SHI_x = 0
        self.MI_SHI_y = 0


'''
A class containing particles across multiple cameras that satisfy epipolar constraints.
'''


class Match():
    def __init__(self, residual=0):
        self.residual = residual
        self.matches = []

    def add_particle(self, particle, camera_id):
        p = particle.reshape((2, 1))
        p = np.append(p, camera_id)
        self.matches.append(p)

    def add_particles(self, particles, camera_ids):
        for particle, id in zip(particles, camera_ids):
            self.add_particle(particle, id)

    def isUnique(self):
        unique = True
        for i in range(len(self.matches)):
            unique &= (self.matches[i][0].match_count == 1)
            self.matches[i][0].match_count = 0
        return unique


'''
Extract several features around the centroid of particle image:
max_intensity   The max intensity around centroid
mean_intensity  The mean intensity around centroid
max_location    The pixel coordinates of the max intensity
max_x_shift     The shift from centroid.x to max_location.x
max_y_shift     The shift from centroid.y to max_location.y
'''


def extract_intensities(img, moving_radius, cX, cY):
    height = img.shape[0]
    width = img.shape[1]

    cX = int(cX)
    cY = int(cY)

    if 0 + moving_radius <= cX <= width - 1 - moving_radius and 0 + moving_radius <= cY <= height - 1 - moving_radius:
        # pixel intensity
        intensity = img[int(cY), int(cX)]

        # Max intensity
        max_intensity = np.max(img[int(cY) - moving_radius:int(cY) + moving_radius + 1,
                               int(cX) - moving_radius:int(cX) + moving_radius + 1])

        # Mean intensity
        mean_intensity = np.mean(img[int(cY) - moving_radius:int(cY) + moving_radius + 1,
                                 int(cX) - moving_radius:int(cX) + moving_radius + 1])

        # Max intensity location
        max_location = np.where(img[int(cY) - moving_radius:int(cY) + moving_radius + 1,
                                int(cX) - moving_radius:int(cX) + moving_radius + 1] == max_intensity)

        # Max intensity shift
        max_x_shift = max_location[1][0] - moving_radius
        max_y_shift = max_location[0][0] - moving_radius

        return intensity, max_intensity, mean_intensity, max_x_shift, max_y_shift
    else:
        return 0, 0, 0, moving_radius * 2, moving_radius * 2


'''
For a grascale image, this function first binarize it to black & white. Then pixel segments are extracted as particle
images. For each segment, the centroid coordinates (x, y) are calculated according to their pixel momentum. Except for 
coordinates, for each segment, a ParticleImage object is created containing different features as mentioned above.
'''


def contours_extraction(img, threshold=30, moving_radius=1):
    _, threshold_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(threshold_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    particle_images = []
    coords = np.empty((0, 2))

    for c in contours:
        # compute the center of the contour
        M = cv.moments(c)

        if M["m00"]:
            particle_img = ParticleImage()

            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

            particle_img.xc = cX
            particle_img.yc = cY

            coords = np.append(coords, np.array([cX, cY]).reshape((1, 2)), axis=0)

            area = cv.contourArea(c)
            particle_img.area = area

            perimeter = cv.arcLength(c, True)
            particle_img.peri = perimeter

            _, radius = cv.minEnclosingCircle(c)
            radius = float(radius)
            particle_img.r = radius

            roundness = 4 * np.pi * area / (perimeter ** 2)
            particle_img.rou = roundness

            particle_images.append(particle_img)

            intensity, max_intensity, mean_intensity, max_x_shift, max_y_shift = extract_intensities(img,
                                                                                                     moving_radius,
                                                                                                     cX,
                                                                                                     cY)

            particle_img.I = intensity
            particle_img.MI = max_intensity
            particle_img.MEI = mean_intensity

            particle_img.MI_SHI_x = max_x_shift
            particle_img.MI_SHI_y = max_y_shift

            # # draw the contour and center of the shape on the image
            # cv.drawContours(img, [c], -1, (0, 255, 0), 2)
            # cv.circle(img, (cX, cY), 10, (255, 255, 255), 1)
            # cv.putText(img, "center", (cX - 20, cY - 20),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # # show the image
    # cv.imshow("test", img)
    # cv.waitKey(0)
    return particle_images, coords


'''
This function process all images and extract the particle images.
@images images from eight cameras
 return list of of list of particle images
 [[particle image 0 in camera 0, particle image 1 in camera 0......],
  ...
  [particle image 0 in camera N-1, particle image 1 in camera N-1......]]
'''


def particle_detection(images, num_cameras):
    particles_list = [[] for _ in range(num_cameras)]
    coords_list = [[] for _ in range(num_cameras)]

    for c in range(num_cameras):
        particle_images, coords = contours_extraction(images[c])
        particles_list[c] = particle_images
        coords_list[c] = coords

    return particles_list, coords_list


'''
A utility function that calculates the residual dot product of a epipolar line and a homogeneous 2D particle coordinate
(x, y, 1). If the epipolar line and this particle corresponded perfectly, the residual should be zero (this particle is
on this line).
'''


def calculateEpipolarResidual(line, particle):
    line = line.reshape((1, 3))
    particle = np.append(particle, 1)
    return np.absolute(np.dot(line, particle))


'''
Given two 2D particle coordinates from two cameras (xa, ya), (xb, yb), and the fundamental matrix f of camera a and b.
calculate the epipolar line of particle b in camera a, then calculate the residual using above function.
'''


def calculate_epiline_residual(pa, pb, f):
    epiline = calculateEpiline(pb, f)
    residual = calculateEpipolarResidual(epiline, pa)
    return residual


'''
Calculate the 2D line function ax + by + c = 0, where x and y are in pixels. 
Here, the "particle" is in camera b. And the calculated epipolar line in on camera a's image plane.
'''


def calculateEpiline(particle, F):
    homoParticle = np.append(particle, 1)

    line = np.dot(homoParticle, F)

    line = line / np.linalg.norm(line[:2])  # different from VPTV c++ code; [:, 2] are a and b for y = ax + b
    return line


'''
The reverse of above function, the particle is in camera a, whereas the epipolar line in on camera b's image plane
'''


def calculateEpiline_reverse(particle, F):
    homoParticle = np.append(particle, 1)

    line = np.dot(homoParticle, F.T)

    line = line / np.linalg.norm(line[:2])  # different from VPTV c++ code; [:, 2] are a and b for y = ax + b
    return line


'''
This class contains functions to 
1. for each particle image in camera a, find the particles in camera b that satisfy epipolar constraint
2. for particle pair <pa, pb>, extract the features
3. for feature vector of particle pair <pa, pb>, predict if it is correctly corresponded using trained classifiers
'''


class Correspondor:
    def __init__(self, camera_sys, maxParticlePerImage=10, maxMatch=5, threshold=0.5, distortion=False):
        self.camera_sys = camera_sys
        self.num_pairs = self.camera_sys.num_pairs
        self.num_cameras = self.camera_sys.num_cameras
        self.matchMap = np.ones((maxParticlePerImage * self.num_cameras, self.num_cameras, maxMatch)) * -1
        self.match_residual_map = np.ones((maxParticlePerImage * self.num_cameras, self.num_cameras, maxMatch)) * 10000
        self.global_match_map = np.ones((maxParticlePerImage * self.num_cameras, self.num_cameras, maxMatch)) * -1

        self.match_threshold = threshold
        self.maxMatch = maxMatch
        self.match_overload = 0
        self.matchList = []
        self.maxParticlePerImage = maxParticlePerImage
        self.distortion = distortion
        self.is_4way_matched = object()

        self.reconstructor = object()
        self.reprojector = object()

        self.four_way_match_reprojection_error = 0
        self.scaler = load(open('./model_parameters/scaler.pkl', 'rb'))
        self.pca = load(open('./model_parameters/pca.pkl', 'rb'))
        self.log_clf = load(open('./model_parameters/log_clf.pkl', 'rb'))
        self.tree_clf = load(open('./model_parameters/tree_clf.pkl', 'rb'))
        self.nn_clf = load(open('./model_parameters/nn_clf.pkl', 'rb'))
        self.svm_clf = load(open('./model_parameters/svm_clf.pkl', 'rb'))
        self.frame_length = []

    # set the matchMap to a three dimensional matrix with all -1
    def reset_matchMap(self):
        self.matchMap = np.ones((self.maxParticlePerImage * self.num_cameras, self.num_cameras, self.maxMatch)) * -1

    # given a particle pair <pa, pb>, extract the feature vector for classification
    def extract_particle_pair_features(self, cam_a_index, cam_b_index, particle_a_index, particle_b_index,
                                       frameGroup, particle_images, images, epipolar_error, moving_radius=1):
        coords_a = frameGroup[cam_a_index][particle_a_index]
        coords_b = frameGroup[cam_b_index][particle_b_index]

        xca = coords_a[0]
        yca = coords_a[1]
        xcb = coords_b[0]
        ycb = coords_b[1]

        particle_img_a = particle_images[cam_a_index][particle_a_index]
        particle_img_b = particle_images[cam_b_index][particle_b_index]

        temp_match = Match()
        temp_match.add_particle(coords_a, cam_a_index)
        temp_match.add_particle(coords_b, cam_b_index)

        temp_match_list = [temp_match]

        coords_3d = self.camera_sys.reconstruct3D_lsqt(temp_match_list)

        # reproject to images to extract extra features
        num_cameras = self.num_cameras
        intensity_list = []
        max_intensity_list = []
        mean_intensity_list = []
        max_x_shift_list = []
        max_y_shift_list = []

        xca_repo = -1
        yca_repo = -1
        xcb_repo = -1
        ycb_repo = -1

        for repo_cam_index in range(num_cameras):
            coords_2d = self.camera_sys.reproject_to_cam(coords_3d, repo_cam_index)
            cX_repo = coords_2d[0][0]
            cY_repo = coords_2d[0][1]

            intensity, max_intensity, mean_intensity, max_x_shift, max_y_shift = extract_intensities(
                images[repo_cam_index],
                moving_radius,
                cX_repo,
                cY_repo)

            intensity_list.append(intensity)
            max_intensity_list.append(max_intensity)
            mean_intensity_list.append(mean_intensity)
            max_x_shift_list.append(max_x_shift)
            max_y_shift_list.append(max_y_shift)

            if repo_cam_index == cam_a_index:
                xca_repo = cX_repo
                yca_repo = cY_repo

            if repo_cam_index == cam_b_index:
                xcb_repo = cX_repo
                ycb_repo = cY_repo

        repo_SHI_xa = xca_repo - xca
        repo_SHI_ya = yca_repo - yca
        repo_SHI_xb = xcb_repo - xcb
        repo_SHI_yb = ycb_repo - ycb

        repo_error_a = np.sqrt(repo_SHI_xa ** 2 + repo_SHI_ya ** 2)
        repo_error_b = np.sqrt(repo_SHI_xb ** 2 + repo_SHI_yb ** 2)
        repo_error_total = repo_error_a + repo_error_b

        X = coords_3d[0][0]
        Y = coords_3d[0][1]
        Z = coords_3d[0][2]

        epipolar_error = epipolar_error

        features = [cam_a_index, cam_b_index,
                    xca, yca, xcb, ycb,
                    particle_img_a.area, particle_img_b.area,
                    particle_img_a.r, particle_img_b.r,
                    particle_img_a.rou, particle_img_b.rou,
                    particle_img_a.peri, particle_img_b.peri,
                    particle_img_a.I, particle_img_b.I,
                    particle_img_a.MI, particle_img_b.MI,
                    particle_img_a.MEI, particle_img_b.MEI,
                    particle_img_a.MI_SHI_x, particle_img_b.MI_SHI_x,
                    particle_img_a.MI_SHI_y, particle_img_b.MI_SHI_y,
                    epipolar_error,
                    X, Y, Z,
                    xca_repo, yca_repo, xcb_repo, ycb_repo,
                    repo_error_a, repo_error_b, repo_error_total,
                    repo_SHI_xa, repo_SHI_ya,
                    repo_SHI_xb, repo_SHI_yb]

        for I, MI, MEI in zip(intensity_list, max_intensity_list, mean_intensity_list):
            features.append(I)
            features.append(MI)
            features.append(MEI)

        return features

    # predict the prediction label of feature extracted by above function using three classifiers (logistic regression,
    # decision truee, and neural network)
    def predict_pair(self, cam_a_index, cam_b_index, particle_a_index, particle_b_index,
                     frameGroup, particle_images, images, epipolar_error):
        features = self.extract_particle_pair_features(cam_a_index, cam_b_index,
                                                       particle_a_index, particle_b_index,
                                                       frameGroup, particle_images, images, epipolar_error)

        features = np.array(features).reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        features_scaled[np.isnan(features_scaled)] = 0
        features_pca = self.pca.transform(features_scaled)

        log_label = self.log_clf.predict(features_pca)
        tree_label = self.tree_clf.predict(features_pca)
        nn_label = self.nn_clf.predict(features_pca)

        return log_label, tree_label, nn_label

    def correspond_to_2nd_cam(self, cam_a_index, cam_b_index, particle_a_index, max_particles_per_img,
                              frameGroup, particle_images, images):
        if cam_a_index:
            a_start = self.frame_length[cam_a_index - 1]
        else:
            a_start = 0

        # look up the match table
        matchmap = self.matchMap
        particles_cam_b = matchmap[a_start + particle_a_index][cam_b_index]
        particles_cam_b_non_negative = particles_cam_b[np.where(particles_cam_b != -1)]

        if len(particles_cam_b_non_negative) > 0:
            for i, particle_b_index in enumerate(particles_cam_b_non_negative):
                particle_b_index = int(particle_b_index)
                epipolar_error = self.match_residual_map[a_start + particle_a_index][cam_b_index][i]

                # features = self.extract_particle_pair_features(cam_a_index, cam_b_index,
                #                                                particle_a_index, particle_b_index,
                #                                                frameGroup, particle_images, images, epipolar_error)
                # features = np.array(features).reshape(1, -1)
                #
                # features_scaled = self.scaler.transform(features)
                # features_pca = self.pca.transform(features_scaled)
                # log_label = self.log_clf.predict(features_pca)
                # tree_label = self.tree_clf.predict(features_pca)
                # nn_label = self.nn_clf.predict(features_pca)

                log_label, tree_label, nn_label = self.predict_pair(cam_a_index, cam_b_index,
                                                                    particle_a_index, particle_b_index,
                                                                    frameGroup, particle_images, images, epipolar_error)

    # calculate the accumulated number of particles from camera 0 to num_camera - 1
    def calculate_num_particle_each_camera(self, frameGroup):
        frame_length = []

        for i, frame in enumerate(frameGroup):
            if i == 0:
                frame_length.append(frame.shape[0])
            else:
                frame_length.append(frame.shape[0] + frame_length[i - 1])

    # This function is used to check the two correspond particles in all camera pairs
    def findEpipolarMatches(self, frameGroup):
        match_overload = 0
        all_particles = np.vstack(frameGroup)
        all_length = all_particles.shape[0]

        for i, frame in enumerate(frameGroup):
            if i == 0:
                self.frame_length.append(frame.shape[0])
            else:
                self.frame_length.append(frame.shape[0] + self.frame_length[i - 1])

        frame_length = self.frame_length

        for key in self.camera_sys.pairs:
            pair = self.camera_sys.pairs[key]
            aid = pair.cameraAid
            bid = pair.cameraBid

            a_particles = frameGroup[aid]
            b_particles = frameGroup[bid]

            a_start = 0 if aid == 0 else frame_length[aid - 1]
            b_start = 0 if bid == 0 else frame_length[bid - 1]

            a_end = frame_length[aid]
            b_end = frame_length[bid]

            for p_b in range(b_start, b_end):
                line_a = calculateEpiline(all_particles[p_b], pair.fmatrix)

                for p_a in range(a_start, a_end):
                    residual = calculateEpipolarResidual(line_a, all_particles[p_a])

                    # print('residual is: %f, pa is: %d, pb is %d' % (residual, p_a, p_b))

                    if residual <= self.match_threshold:
                        local_p_a = p_a - a_start
                        local_p_b = p_b - b_start
                        a_ret = np.argwhere(self.matchMap[p_b][aid] == -1)
                        b_ret = np.argwhere(self.matchMap[p_a][bid] == -1)

                        if np.ravel(a_ret).shape[0] != 0:
                            first_empty_b = a_ret[0, 0]
                        else:
                            first_empty_b = self.maxMatch + 1

                        if np.ravel(b_ret).shape[0] != 0:
                            first_empty_a = b_ret[0, 0]
                        else:
                            first_empty_a = self.maxMatch + 1

                        if first_empty_b < self.maxMatch and first_empty_a < self.maxMatch:
                            self.matchMap[p_a, bid, first_empty_a] = local_p_b
                            self.matchMap[p_b, aid, first_empty_b] = local_p_a
                            self.global_match_map[p_a, bid, first_empty_a] = p_b
                            self.global_match_map[p_b, aid, first_empty_b] = p_a
                            self.match_residual_map[p_a, bid, first_empty_a] = residual
                            self.match_residual_map[p_b, aid, first_empty_b] = residual
                        else:
                            self.match_overload += 1


image_folder = "./data/images/"
num_particles_per_img = 50  # 50, 100, 500, 1000, 2000

# this file contains the ground truth 3D coordinates of particles
xyz_coordinate_file = image_folder + '{}/xyz_{}.txt'.format(num_particles_per_img, num_particles_per_img)
xyz_coordinates = np.loadtxt(xyz_coordinate_file)

image_names = sorted(glob.glob(image_folder + str(num_particles_per_img) + '/*.jpg'))

images = []  # the images captured by all cameras at one instant
for name in image_names:
    img = np.array(Image.open(name).convert('L'))  # you can pass multiple arguments in single line
    images.append(img)

num_cameras = len(image_names)

# particle_images_list has a length equals num_cameras, where particle_images_list[i] contains extracted ParticleImages
# captured by camera i. Similarily, particle_coords_list has a length num_cameras, where particle_coords_list contains
# pixel coordinates (x, y) of each particle captured by camera i
particle_images_list, particle_coords_list = particle_detection(images, num_cameras)

# --------------------------
camera_folder = './cameras_dir/camera_parameters'

pairs_file = camera_folder + '/pairs_calibrated.yaml'  # a yaml file with each block contains parameters of camera
# pair <camera_a, camera_b>: cam_a_index, cam_b_index, fundamental matrix
cameras_file = camera_folder + '/cameras_calibrated.yaml'  # a yaml file with each block contains parameters of camera i

s_file = camera_folder + '/S.txt'  # rotation of world coordinate system to new world coordinate system
p_file = camera_folder + '/P.txt'  # translation of world coordinate system to new world coordinate system

"""
Parameters for correspondence
"""
max_particles_per_img = num_particles_per_img * 2  # maximum number of particles per image, used to initiate corresponder.MatchMap
maximum_match_candidates = 5  # maximum number of particles that satisfy the epipolar constraint in camera b to particle p in camera a, used to initiate corresponder.MatchMap

# the threshold to determine if particle pa in camera a and particle pb in camera b satisfy epipolar constraint
# if calculate_epipolar_residual(pa, pb, f) < epipolar_threshold, where f is the fundamental matrix between camera a and b.
# then pb is the particle candidate to particle pa
epipolar_threshold = 4

# if the images are distorted, in this case it is False because all images were already un-distorted
distortion = False

cameras = read_cameras(cameras_file)
camera_pairs = read_pairs(pairs_file)
camera_sys = camera_system_(read_pairs=camera_pairs, cameras=cameras, S=s_file, P=p_file)

corresponder = Correspondor(camera_sys, max_particles_per_img, maximum_match_candidates, epipolar_threshold, distortion)
corresponder.findEpipolarMatches(particle_coords_list)
corresponder.calculate_num_particle_each_camera(particle_coords_list)

fig_0 = go.Figure()
fig_1 = go.Figure()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

first_options = [{'label': 'Camera {}'.format(i), 'value': i} for i in range(7)]

app.layout = html.Div([
    dbc.Row([dbc.Col(html.P('Please select the first camera:', style={'font-size': 20}), width={'size': 3}),
             dbc.Col(dcc.Dropdown(
                 id='dropdown_0',
                 options=first_options,
                 value='0',
             ), width={'size': 3, 'offset': 0})]
            ),
    dbc.Row([
        dbc.Col(html.P('Please select the second camera:'), style={'font-size': 20}, width={'size': 3}),
        dbc.Col(dcc.Dropdown(
            id='dropdown_1',
            value='1',
        ), width={'size': 3, 'offset': 0})
    ]),
    dbc.Row([
        dbc.Col(html.P('Click on image or select the particle in the first camera:'), style={'font-size': 20},
                width={'size': 3}),
        dbc.Col(dcc.Dropdown(
            id='dropdown_2',
            value='0',
        ), width={'size': 3, 'offset': 0})
    ]),
    dbc.Row([
        dbc.Col(html.P('Change to small if images are too large to fit:'), style={'font-size': 20}, width={'size': 3}),
        dbc.Col(dcc.Dropdown(
            id='scale-factors',
            value='1',
            options=[{'label': 'Small', 'value': 0.5},
                     {'label': 'Large', 'value': 1.0}]
        ), width={'size': 3})
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='img0'), width={'size': 6}),
        dbc.Col(dcc.Graph(id='img1'), width={'size': 6}),
    ])
])


@app.callback(
    [Output('dropdown_1', 'options'),
     Output('dropdown_2', 'options')],
    Input('dropdown_0', 'value')
)
def update_2nd_cam_options(value):
    start = int(value) + 1
    cam_index = int(value)
    num_particles = len(particle_images_list[cam_index])
    return [{'label': 'Camera {}'.format(i), 'value': i} for i in range(start, 8)], \
           [{'label': 'Particle {}'.format(i), 'value': i} for i in range(num_particles)]


@app.callback(
    [Output('img0', 'figure'), Output('img1', 'figure')],
    [Input('dropdown_0', 'value'),
     Input('dropdown_1', 'value'),
     Input('dropdown_2', 'value'),
     Input('scale-factors', 'value')]
)
def update_output(value_0, value_1, value_2, value_3):
    fig_0.data = []
    fig_1.data = []

    fig_0.layout = {}
    fig_1.layout = {}

    if type(value_2) == 'NoneType':
        value_2 = 0

    cam_left_index = int(value_0)
    cam_right_index = int(value_1)
    particle_index = int(value_2)

    if cam_left_index == cam_right_index:
        cam_right_index = cam_left_index + 1

    particle_left = particle_coords_list[cam_left_index][particle_index]

    fmatrix = corresponder.camera_sys.pairs[str(cam_left_index) + str(cam_right_index)].fmatrix

    line = calculateEpiline_reverse(particle_left, fmatrix)

    # Constants
    img_width = 1280
    img_height = 1024
    scale_factor = float(value_3)

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig_0.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    temp_x = particle_left[0] * scale_factor
    temp_y = (img_height - particle_left[1] - 1) * scale_factor

    # highlight the selected particle on image 0
    fig_0.add_trace(
        go.Scatter(
            x=[temp_x],
            y=[temp_y],
            mode="markers",
            marker_opacity=1,
        )
    )

    coords_array_left = np.array(particle_coords_list[cam_left_index]).reshape(-1, 2)

    fig_0.add_trace(
        go.Scatter(
            x=coords_array_left[:, 0] * scale_factor,
            y=(img_height - 1 - coords_array_left[:, 1]) * scale_factor,
            mode='markers',
            marker=dict(size=20,
                        line=dict(width=2, color='MediumPurple'),
                        color='rgba(135, 206, 250, 0.1)',
                        )

        )
    )

    fig_0.update_layout(clickmode='event+select')

    # add epipolar line on image 1 corresponding to selected particle in left camera
    xs = np.linspace(0, img_width, 10)
    #  ax + yb + c = 0; y = (-ax - c) / b
    a = line[0]
    b = line[1]
    c = line[2]

    ys = -(a * xs + c) / b

    opacity = 1

    # center line
    fig_1.add_trace(
        go.Scatter(x=xs * scale_factor, y=(img_height - 1 - ys) * scale_factor,
                   mode='lines',
                   line=dict(color='royalblue', width=1),
                   opacity=opacity
                   ),
    )

    epipolar_threshold_factor = 3

    # ax + by + c = 0
    # ax + by + c' = 0

    # distance = |c - c'| / sqrt(1 + (a/b)^2)
    # |c - c'| = distance * sqrt(1 + (a/b)^2)
    # c - c' = distance * sqrt(1 + (a/b)^2)   ->    c' = c - distance * sqrt(1 + (a/b)^2)
    # c - c' = -distance * sqrt(1 + (a/b)^2)  ->    c' = c + distance * sqrt(1 + (a/b)^2)
    distance = epipolar_threshold * epipolar_threshold_factor
    denominator = np.sqrt(1 + (a / b) ** 2)

    c_movement_maximum = distance
    c_movement = distance * denominator

    if c_movement > c_movement_maximum:
        c_movement = c_movement_maximum

    c_1 = c - c_movement
    c_2 = c + c_movement

    if b > 0:
        ys_upper = -(a * xs + c_1) / b
        ys_lower = -(a * xs + c_2) / b
    else:
        ys_upper = -(a * xs + c_2) / b
        ys_lower = -(a * xs + c_1) / b

    ys_upper = img_height - 1 - ys_upper
    ys_lower = img_height - 1 - ys_lower

    fig_1.add_trace(
        go.Scatter(x=xs * scale_factor, y=ys_upper * scale_factor,
                   mode='lines',
                   line=dict(color='orange', width=1, dash='dash'),
                   opacity=opacity
                   ),
    )

    fig_1.add_trace(
        go.Scatter(x=xs * scale_factor, y=ys_lower * scale_factor,
                   mode='lines',
                   line=dict(color='orange', width=1, dash='dash'),
                   opacity=opacity
                   ),
    )

    xs_reverse = xs[::-1]
    ys_lower_reverse = ys_lower[::-1]

    fig_1.add_trace(go.Scatter(
        x=list(xs * scale_factor) + list(xs_reverse * scale_factor),
        y=list(ys_upper * scale_factor) + list(ys_lower_reverse * scale_factor),
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Ideal',
    ))

    # plot the scatter of corresponded particles in right image
    particles_right = particle_coords_list[cam_right_index]

    if cam_left_index:
        left_start_index = corresponder.frame_length[cam_left_index - 1]
    else:
        left_start_index = 0

    particle_left_index = left_start_index + particle_index
    particles_corresponded_index = corresponder.matchMap[particle_left_index][cam_right_index]
    particles_index_close_to_epipolar_line = [int(i) for i in particles_corresponded_index if i != -1]

    particles_epipolars_errors_close_to_epipolar_line = corresponder.match_residual_map[particle_left_index][
        cam_right_index]

    particles_close_to_epipolar_line = particle_coords_list[cam_right_index][particles_index_close_to_epipolar_line]
    particles_close_to_epipolar_line = np.array(particles_close_to_epipolar_line).reshape(-1, 2)

    for particle_right, particle_right_index, epipolar_error in zip(particles_close_to_epipolar_line,
                                                                    particles_index_close_to_epipolar_line,
                                                                    particles_epipolars_errors_close_to_epipolar_line):
        log_label, tree_label, nn_label = corresponder.predict_pair(cam_left_index, cam_right_index,
                                                                    particle_left_index - left_start_index,
                                                                    particle_right_index,
                                                                    particle_coords_list, particle_images_list, images,
                                                                    epipolar_error)

        text = f"log_label: {bool(log_label)} <br> tree_label: {bool(tree_label)} <br> nn_label: {bool(nn_label)}"

        fig_1.add_annotation(
            x=particle_right[0] * scale_factor,
            y=(img_height - 1 - particle_right[1]) * scale_factor,
            xref="x",
            yref="y",
            text=text,
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=5,
            arrowcolor="#636363",
            ax=100,
            ay=-100,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
        )

    fig_1.add_trace(
        go.Scatter(
            x=particles_close_to_epipolar_line[:, 0] * scale_factor,
            y=(img_height - 1 - particles_close_to_epipolar_line[:, 1]) * scale_factor,
            mode='markers',
            marker=dict(size=20,
                        line=dict(width=2, color='Gold'),
                        color='rgba(135, 206, 250, 0.1)',
                        )
        )
    )

    fig_0.update_layout(showlegend=False)
    fig_1.update_layout(showlegend=False)

    # Configure axes
    fig_0.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig_1.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig_0.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    fig_1.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig_0.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            # sizing="stretch",
            source=app.get_asset_url('/images/50/{}.jpg'.format(value_0)))
    )

    # Configure other layout
    fig_0.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    fig_1.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            # sizing="stretch",
            source=app.get_asset_url('/images/50/{}.jpg'.format(value_1)))
    )

    # Configure other layout
    fig_1.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig_0, fig_1


@app.callback(
    Output('dropdown_2', 'value'),
    Input('img0', 'clickData')
)
def update_particle_num_2nd_cam(clickData):
    if clickData is None:
        return 0
    else:
        return clickData['points'][0]['pointIndex']


if __name__ == '__main__':
    app.run_server(debug=True)
