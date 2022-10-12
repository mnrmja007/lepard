"""Base class for Point Clouds with part-whole relationships."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import os
from plyfile import PlyData
import time


class PointCloudFeatures:
    """Features of a point cloud."""
    centroid = np.zeros(3)
    std = np.zeros(3)

    min_x = 0
    min_y = 0
    min_z = 0
    max_x = 0
    max_y = 0
    max_z = 0

    eigen_values = np.zeros(3)
    eigen_vectors = np.zeros(3, 3)


class PointCloud(object):
    """Represents a point cloud"""

    def __init__(self, points):
        if not isinstance(points, np.ndarray) and isinstance(points, list):
            points = np.asarray(points)
        else:
            raise ValueError(f"Unrecognized type for `points`. Only supports: np.ndarray and list.")
        assert (len(points.shape)==2) and (points.shape[1]==3), "points should be of shape [N, 3]"
        self.points = points
        self.features = PointCloudFeatures
        self.compute_features()
        self.n_points = self.points.shape[0]

    def compute_features(self):
        self.features.centroid = np.mean(self.points, axis=1)
        self.features.std = np.std(self.points, axis=1)
        mins = np.min(self.points, axis=1)
        self.features.min_x, self.features.min_y, self.features.min_z = mins[0], mins[1], mins[2]
        maxs = np.max(self.points, axis=1)
        self.features.max_x, self.features.max_y, self.features.max_z = maxs[0], maxs[1], maxs[2]
        eigen_values, eigen_vectors = np.linalg.eigvals(self.points)
        idx = eigen_values.argsort()[::-1]
        self.features.eigen_values, self.features.eigen_vectors = eigen_values[idx], eigen_vectors[:, idx].T

    def visualize(self, color=[255, 0, 0]):
        viz_pcd = o3d.geometry.PointCloud()
        viz_pcd.points = o3d.utility.Vector3dVector(self.points)
        viz_pcd.paint_uniform_color(color)
        o3d.visualization.draw_geometries([viz_pcd])

    def save(self, path,  color=[]):
        os.makedirs(os.path.dirname(path))
        viz_pcd = o3d.geometry.PointCloud()
        viz_pcd.points = o3d.utility.Vector3dVector(self.points)
        if color:
            viz_pcd.paint_uniform_color(color)
        o3d.io.write_point_cloud(path, viz_pcd, write_ascii=True)


def create_point_cloud_from_ply(path_to_ply):
    if not path_to_ply.endswith(".ply"):
        raise ValueError("Works with ply files only.")
    data = PlyData.read(path_to_ply)
    points = np.asarray([[float(x) for x in t] for t in data["vertex"].data])
    return PointCloud(points)


def closest_point_distance(point_cloud_1, point_cloud_2):
    points_1 = point_cloud_1.points
    points_2 = point_cloud_2.points
    points_2 = np.tile(points_2, [1, point_cloud_1.n_points])
    points_2.reshape([point_cloud_1*point_cloud_2, 3])
    points_1 = np.tile(points_1, [point_cloud_2.n_points, 1])
    return np.min(np.linalg.norm(points_1 - points_2, axis=1))


class PointCloudWhole(PointCloud):
    """Point cloud with part whole relationship."""

    def __init__(self, point_cloud_parts, connected_component_distance_threshold=0.01):
        if not np.all([isinstance(x, PointCloud) for x in point_cloud_parts]):
            raise ValueError("point_cloud_parts must be a list of PointCloud objects.")
        self.parts = point_cloud_parts
        all_points = np.concatenate([part.points for part in self.parts], axis=0)
        super(PointCloudWhole).__init__(self, all_points)
        self.connected_component_distance_threshold = connected_component_distance_threshold
        self.graph = nx.Graph()
        self.graph.add_nodes_from(
            {i: {"point_cloud": part} for i, part in enumerate(self.parts)}
        )
        for i in range(len(self.parts)):
            for j in range(len(self.parts)):
                if closest_point_distance(self.parts[i], self.parts[j]) <= connected_component_distance_threshold:
                    edge_length = np.linalg.norm(self.parts[j].features.centroid - self.parts[i].features.centroid)
                    direction = (self.parts[j].features.centroid - self.parts[i].features.centroid) / edge_length
                    self.graph.add_edge(
                        i, j,
                        length=edge_length,
                        direction=direction)
    def visualize(self, color=[255, 0, 0]):
        pass
        