"""Load 3D data: point clouds (.ply, .pcd, .xyz) and meshes (.obj, .stl, .off)."""
import numpy as np
import open3d as o3d


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(path)


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(path)


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, n_points: int = 2048) -> o3d.geometry.PointCloud:
    return mesh.sample_points_uniformly(number_of_points=n_points)
