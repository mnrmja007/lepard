from cProfile import run
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from plyfile import PlyData
import pdb
import open3d as o3d
import os


def read_ply_file(filename):
    if not filename.endswith(".ply"):
        raise ValueError("Works with ply files only.")
    data = PlyData.read(filename)
    data = np.asarray([[float(x) for x in t] for t in data["vertex"].data])
    return data


def run_clustering(data_path):
    X = read_ply_file(data_path)
    db = DBSCAN(eps=0.1, min_samples=50).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    viz_pcd = o3d.geometry.PointCloud()
    # viz_pcd.points = o3d.utility.Vector3dVector(X)
    rendered_points = []
    pc_colors = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        cluster_points = X[class_member_mask & core_samples_mask]
        # pdb.set_trace()
        rendered_points.extend(cluster_points.tolist())
        pc_colors.extend(np.tile(col, [cluster_points.shape[0], 1]).tolist())

    viz_pcd.points = o3d.utility.Vector3dVector(np.asarray(rendered_points))
    viz_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pc_colors)[:, :3])
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(data_path), "dbscan_clustering.ply"), viz_pcd)
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=14,
    #     )

    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=6,
    #     )

    # plt.title("Estimated number of clusters: %d" % n_clusters_)
    # plt.show()

if __name__=="__main__":
    data_path = "/home/gridraster/Downloads/robo.ply"
    run_clustering(data_path=data_path)