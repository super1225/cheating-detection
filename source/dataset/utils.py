import numpy as np
import math
import random
import json
import torch
import torch.nn.functional as F

with open("config.json") as f:
    config = json.load(f)

def angle_displacement(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    theta = np.arctan2(det, dot)
    theta = theta/(2*np.pi) if theta>0 else (2*np.pi+theta)/(2*np.pi)
    return config['angle_embed_num']*theta, math.sqrt(v2[0]**2+v2[1]**2)

def append_angle_displacement(line_and_threshold, disturb=False, disturb_angle=5, disturb_disp=500):
  
    max_xy_min_xy = line_and_threshold[0]
    max_disp = math.sqrt((max_xy_min_xy[0]-max_xy_min_xy[2])**2 + (max_xy_min_xy[1]-max_xy_min_xy[3])**2)#
    line = line_and_threshold[1]
    for i in range(1, len(line)-1):
        v1 = [line[i][0]-line[i-1][0], line[i][1]-line[i-1][1]]
        v2 = [line[i+1][0]-line[i][0], line[i+1][1]-line[i][1]]
        angle, disp = angle_displacement(v1, v2)
        disp = config['disp_embed_num']*disp/(max_disp+10e-5)
        if disturb:
            p = random.gauss(0, disturb_angle)
            q = random.gauss(0, disturb_disp)
            
        else:
            p = 0.0
            q = 0.0

        angle = int((angle+p)) % config['angle_embed_num']
        disp = disp + q

        if disp<0:
            disp = 0
        if disp>config['disp_embed_num']:
            disp = config['disp_embed_num']
        line[i].extend([angle, disp])

    line[0].extend([float(config['angle_embed_num']+3), config['disp_embed_num']+3])
    if len(line)-1>0:
        line[len(line)-1].extend([float(config['angle_embed_num']+3), config['disp_embed_num']+3])
    return line

def transformation(line):
    q1 = random.random()
    if q1<0.5:
        mirror = True
    else:
        mirror = False
    q2 = random.random()
    rotate = q2*2*math.pi

    q3 = random.random()
    multi = 0.8 + 0.4*q3

    for i in range(len(line[0])):
        line[0][i] *= multi

    max_x = line[1][0][0]
    max_y = line[1][0][1]
    for i in range(len(line[1])):
        x = line[1][i][0]
        if mirror:
            x = -x
        y = line[1][i][1]

        x1 = (math.cos(rotate)*x - math.sin(rotate)*y)*multi
        y1 = (math.cos(rotate)*y + math.sin(rotate)*x)*multi
        
        if max_x<x:
            max_x = x
        if max_y<y:
            max_y = y

        line[1][i][0] = x1
        line[1][i][1] = y1

    move_x = random.uniform(-max_x/2, max_x/2)
    move_y = random.uniform(-max_y/2, max_y/2)

    for i in range(len(line[1])):
        line[1][i][0] += move_x
        line[1][i][1] += move_y

    return line



def weight_matrix(line, sep, max_disp):
    left = torch.tensor(line)[:sep, :2]
    right = torch.tensor(line)[sep:, :2]
    
    sq_left = left**2
    left_1 = torch.sum(sq_left, dim=1).unsqueeze(1)
    left_2 = torch.sum(sq_left, dim=1).unsqueeze(0)
    left_t = left.t()
    weight_matrix_left = torch.sqrt(left_2+left_1-2*left.mm(left_t))
    weight_matrix_left = weight_matrix_left/max_disp*config['disp_embed_num']
    weight_matrix_left = weight_matrix_left.unsqueeze(0).unsqueeze(0)
    weight_matrix_left = F.pad(weight_matrix_left, [0, config['left_threshold']-sep, 0, config['left_threshold']-sep], 
            'constant', config['disp_embed_num']+3)
    weight_matrix_left = weight_matrix_left.squeeze(0).squeeze(0)

    sq_right = right**2
    right_1 = torch.sum(sq_right, dim=1).unsqueeze(1)
    right_2 = torch.sum(sq_right, dim=1).unsqueeze(0)
    right_t = right.t()
    weight_matrix_right = torch.sqrt(right_2+right_1-2*right.mm(right_t))
    weight_matrix_right = weight_matrix_right/max_disp*config['disp_embed_num']
    weight_matrix_right = weight_matrix_right.unsqueeze(0).unsqueeze(0)
    weight_matrix_right = F.pad(weight_matrix_right, [0, config['right_threshold']-(len(line)-sep),
             0, config['right_threshold']-(len(line)-sep)], 'constant', config['disp_embed_num']+3)
    weight_matrix_right = weight_matrix_right.squeeze(0).squeeze(0)

    left = torch.tensor(line)[:sep, 2]
    right = torch.tensor(line)[sep:, 2]
    left = left.unsqueeze(1)
    right = right.unsqueeze(0)
    time_weight_matrix = left-right
    zeros = torch.zeros_like(time_weight_matrix)
    time_weight_matrix = torch.max(time_weight_matrix, zeros-config['time_threshold'])
    time_weight_matrix = torch.min(time_weight_matrix, zeros+config['time_threshold'])
    time_weight_matrix += config['time_threshold']
    time_weight_matrix = time_weight_matrix/(2*config['time_threshold'])*config['time_segment']
    time_weight_matrix = time_weight_matrix.unsqueeze(0).unsqueeze(0)
    time_weight_matrix = F.pad(time_weight_matrix, [0, config['right_threshold']-(len(line)-sep), 
            0, config['left_threshold']-sep], 'constant', config['time_segment']+3)
    time_weight_matrix = time_weight_matrix.squeeze(0).squeeze(0)

    return weight_matrix_left, weight_matrix_right, time_weight_matrix
