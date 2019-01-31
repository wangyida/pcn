# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
from open3d import *


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.concatenate([np.array(pcd.points), np.array(pcd.colors)], 1)


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    write_point_cloud(filename, pcd)
