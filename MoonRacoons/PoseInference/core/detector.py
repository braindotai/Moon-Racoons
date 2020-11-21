from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

import cv2
import math, time
import collections
import numpy as np
from functools import reduce
from operator import itemgetter
import os, json, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])

model = None
precision = None
index_label = None
initialized = None
device = None

def initialize():
    global model, device, precision, index_label, initialized
    
    print('Running initialization...')
    device = os.getenv('DEVICE', 'cpu')
    precision = os.getenv('PRECISION', 'float32')
    
    print(f'Device   : {device}')
    print(f'Precision: {precision}')

    if device == 'cuda':
        assert torch.cuda.is_available(), '\n\n"device" is set to "cuda", but no gpu is found :(\n'
    elif device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model().to(device)

    if precision == 'float16':
        assert device == 'cuda', f'\n\nHalf precision is only available when device is set to "cuda", but got "{device}".\n'
        model = model.half()

    initialized = True


@torch.no_grad()
def detect(cv2_image, save_to_file = False, output_dir = None, output_image_dir = None, verbose = False):
    """
    This function is meant to be run on inference.
    Arguments:
    cv2_image              : (str)  Path to the image on which inference will be run.
                             (required)
    
    save_to_file           : (bool) Will save the json and image outputs if `True`.
                                    The output json file is saved with this naming format: <image_name>_pps_{predictions_per_second}_output.json
                                    and the output image file is saved with this naming format: <image_name>_pps_{predictions_per_second}_output.mp4.
                                    And at least one of `output_dir` or `output_image_dir` must be provided is this is set to `True`. 
                             (default: `False`)
    
    output_dir             : (str)  Directory in which json outputs will be saved. This directory will get created if it doesn't exist.
                             (default: None)
    
    output_image_dir       : (str)  Directory in which image outputs will be saved. This directory will get created if it doesn't exist.
                             (default: None)
    
    verbose                : (bool) Will display more information if `True`.
                             (default: None)
    """

    if not initialized:
        initialize()
    
    if save_to_file:
        assert output_dir or output_image_dir, f'\n\nPath for at least one of `output_dir` and `output_image_dir` is required when `save_to_file` is set to `True`, found both `None`.\n'

    if verbose:
        print(f'\nReading image...')
    reader = ImageReader(cv2_image)

    if verbose:
        print('Making predictions...')
    output, output_image = inference(reader, 256, verbose)
    return output_image

def get_model():
    model = PoseEstimationWithMobileNet()

    ckpt_dir = os.path.join(BASE_DIR, 'parameters')
    assert os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)), '\n\nNo pretrained model is found\n'
    
    checkpoint = torch.load(os.path.join(ckpt_dir, 'parameters.pth'), map_location = 'cpu')

    source_state = checkpoint['state_dict']
    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            print('Parameters for "{}" are not found, error in loading the pretrained model....'.format(target_key))
            sys.exit(1)

    model.load_state_dict(new_target_state)
    model.eval()

    return model


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False):
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1                   # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]        # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array(kpts_a[i][0:2])
            for j in range(num_kpts_b):
                kpt_b = np.array(kpts_b[j][0:2])
                mid_point = [(), ()]
                mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                   vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])

                height_n = pafs.shape[0] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        if not demo:
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                        else:
                            px = int(x[point_idx])
                            py = int(y[point_idx])
                        paf = part_pafs[py, px, 0:2]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints

class Pose:
    num_kpts = 18
    color1 = [102, 204, 0]
    color2 = [76, 153, 0]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.id = None

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        pose_points = {}

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]

            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                x_a, y_a = int(x_a), int(y_a)

                if kpt_a_id == 0:
                    # abcd = [0, 255, 255] # nose (yellow)
                    pose_points['nose'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [0, 255, 255], -1)
                elif kpt_a_id == 1:
                    # abcd = [0, 0, 0] # neck (black)
                    pose_points['neck'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [0, 0, 0], -1)
                elif kpt_a_id == 2:
                    # abcd = [204, 0, 102] # right shoulder (purple)
                    pose_points['right_shoulder'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [204, 0, 102], -1)
                elif kpt_a_id == 3:
                    # abcd = [255, 0, 255] # right elbow (pink)
                    pose_points['right_elbow'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [255, 0, 255], -1)
                elif kpt_a_id == 5:
                    # abcd = [0, 0, 255] # left shoulder (red)
                    pose_points['left_shoulder'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [204, 0, 102], -1)
                elif kpt_a_id == 6:
                    # abcd = [0, 255, 0] # left elbow (green)
                    pose_points['left_elbow'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [255, 0, 255], -1)
                elif kpt_a_id == 8:
                    # abcd = [0, 128, 255] # left hip (orange)
                    pose_points['right_hip'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [0, 128, 255], -1)
                elif kpt_a_id == 9:
                    # abcd = [153, 76, 0] # right knee (dark blue)
                    pose_points['right_knee'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [153, 76, 0], -1)
                elif kpt_a_id == 11:
                    # abcd = [153, 255, 204] # right hip (light green)
                    pose_points['left_hip'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [0, 128, 255], -1)
                elif kpt_a_id == 12:
                    # abcd = [255, 255, 0] # left knee (light blue)
                    pose_points['left_knee'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [153, 76, 0], -1)
                elif kpt_a_id == 14:
                    # abcd = [51, 102, 0] # right eye (dark green)
                    pose_points['right_eye'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [51, 102, 0], -1)
                elif kpt_a_id == 15:
                    # abcd = [128, 128, 128] # left eye (gray)
                    pose_points['left_eye'] = (x_a, y_a)
                    cv2.circle(img, (x_a, y_a), 5, [51, 102, 0], -1)


            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                x_b, y_b = int(x_b), int(y_b)

                if kpt_b_id == 4:
                    # abcd_2 = [255, 153, 51] # right hand (sky blue)
                    pose_points['right_hand'] = (x_b, y_b)
                    cv2.circle(img, (x_b, y_b), 5, [255, 153, 51], -1)
                elif kpt_b_id == 7:
                    # abcd_2 = [204, 255, 153] # left hand (medium light blue)
                    pose_points['left_hand'] = (x_b, y_b)
                    cv2.circle(img, (x_b, y_b), 5, [255, 153, 51], -1)
                elif kpt_b_id == 10:
                    # abcd_2 = [153, 255, 204] # right foot (light green)
                    pose_points['right_foot'] = (x_b, y_b)
                    cv2.circle(img, (x_b, y_b), 5, [153, 255, 204], -1)
                elif kpt_b_id == 13:
                    # abcd_2 = [255, 153, 255] # left foot (light pink)
                    pose_points['left_foot'] = (x_b, y_b)
                    cv2.circle(img, (x_b, y_b), 5, [153, 255, 204], -1)
                elif kpt_b_id == 16:
                    # abcd_2 = [204, 204, 255] # right ear (light red)
                    pose_points['right_ear'] = (x_b, y_b)
                    cv2.circle(img, (x_b, y_b), 5, [204, 204, 255], -1)
                elif kpt_b_id == 17:
                    # abcd_2 = [255, 255, 255] # left ear (white)
                    pose_points['left_ear'] = (x_b, y_b)
                    cv2.circle(img, (x_b, y_b), 5, [204, 204, 255], -1)
            
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (x_a, y_a), (x_b, y_b), [255, 183, 138], 2)
        
        return pose_points

def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype = np.float32)
    img = (img - img_mean) * img_scale
    return img


class ImageReader(object):
    def __init__(self, image):
        self.image = image
        shape = self.image.shape
        self.height = shape[0]
        self.width = shape[1]


@torch.no_grad()
def infer_fast(img, net_input_height_size, stride, upsample_ratio, pad_value = (0, 0, 0), img_mean = (128, 128, 128), img_scale = 1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

    tensor_img = tensor_img.to(device)

    if precision == 'float16':
        tensor_img = tensor_img.half()
    
    stages_output = model(tensor_img)

    stage2_heatmaps = stages_output[-2].float()
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1].float()
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def predict(img, reader, height_size):
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    heatmaps, pafs, scale, pad = infer_fast(img, height_size, stride, upsample_ratio)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    img, pose_dict = draw_keypoints_and_normalize(img, reader, current_poses)

    return img, pose_dict

def draw_keypoints_and_normalize(img, reader, current_poses):
    pose_dict = {}

    for i, pose in enumerate(current_poses):
        pose_dict[i + 1] = pose.draw(img)
    
    img = cv2.resize(img, (int(reader.width), int(reader.height)))
    return img, pose_dict


def write_image(image, reader, output_path):
    cv2.imwrite(output_path, image)
    # writer = cv2.imageWriter(output_path, cv2.imageWriter_fourcc(*'mp4v'), reader.fps, (int(reader.width), int(reader.height)))
    # for image in images:
    #     writer.write(image)
    # writer.release()


def get_blocks(reader, predictions_per_second):
    blocks = [[]]
    counter = (reader.fps // predictions_per_second)

    for frame_index, image in enumerate(reader, 1):
        if frame_index < counter:
            blocks[-1].append((frame_index, image))
        elif frame_index == counter:
            counter += (reader.fps // predictions_per_second)
            blocks[-1].append((frame_index, image))
            blocks.append([])
    
    if not len(blocks[-1]):
        blocks.pop()
    
    return blocks


def secondsToStr(seconds):
    return "%02d:%02d:%02d.%03d" % reduce(lambda ll,b : divmod(ll[0],b) + ll[1:], [(round(seconds*1000),),1000,60,60])


def inference(reader, height_size, verbose):
    image, poses = predict(reader.image, reader, height_size)

    return poses, image
    

# for i in range(1, 15):
# 	try:
# 		output = detect(f'test_images/image{i}.jpg', save_to_file = True, output_dir = './json_outputs', output_image_dir = './image_outputs', verbose = True)
# 	except:
# 		output = detect(f'test_images/image{i}.jfif', save_to_file = True, output_dir = './json_outputs', output_image_dir = './image_outputs', verbose = True)
