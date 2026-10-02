#!/usr/bin/env python3
"""
Rig-Retopo-3K Dataset Ingestion & Kinematic Conditioning Preprocessor
=====================================================================
Phase 3: Scaling, Kinematic Conditioning & Production Retopology
Project Code: MESH-FLOW-RETOPOLOGY

This module implements a complete, self-contained PyTorch dataset pipeline:
1. Synthetic & Archetype Asset Generators:
   - Benchmark A: 2D Planar Articulating Hinge (Bending Strip)
   - Benchmark B: 3D Cylindrical Articulating Joint (Elbow / Knee)
   - Benchmark C: Multi-Segment Articulated Limb (Upper Arm, Forearm, Hand)
   - Benchmark D: Humanoid SMPL-X / Mixamo Character Archetype (17-joint hierarchy)
2. Geometry Surface Sampling:
   - High-resolution uniform / Poisson disk surface sampling (M points + normals)
3. Kinematic Skeletal Hierarchy:
   - Joint coordinates J in R^(K x 3), parent indices, bone vectors, adjacency
4. Skinning Weight Matrices:
   - Partition of unity weights W in [0, 1]^(N x K) with sum_k w_ik = 1
5. Deformation Strain Mechanics:
   - Right Cauchy-Green deformation strain tensor C = F^T F across flexion poses
   - Principal strain eigenvectors (v1, v2) for 4-RoSy directional alignment
6. Continuous State Representation:
   - Target retopo state s_1 = [p, n, z] in R^(N x 14) with harmonic cycle latent z
7. PyTorch Dataset & DataLoader:
   - RigRetopoDataset and custom collation for training and validation batches
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. Kinematic Deformation & Cauchy-Green Strain Extraction
# ==============================================================================

class StrainMechanicsExtractor:
    """
    Computes Linear Blend Skinning (LBS) forward kinematics and extracts
    the right Cauchy-Green deformation strain tensor C = F^T F and principal
    strain eigenvectors (v1, v2) across articulated poses.
    """
    @staticmethod
    def compute_lbs(
        vertices: np.ndarray,
        weights: np.ndarray,
        transforms: List[np.ndarray]
    ) -> np.ndarray:
        """
        Computes LBS forward deformation: x_def = sum_k w_ik * (R_k * x_rest + t_k)
        Args:
            vertices: (N, 3) rest vertex positions
            weights: (N, K) normalized skinning weights
            transforms: list of K (4, 4) rigid transformation matrices
        Returns:
            deformed: (N, 3) deformed positions
        """
        N, K = weights.shape
        deformed = np.zeros_like(vertices, dtype=np.float32)
        homo_verts = np.hstack([vertices, np.ones((N, 1), dtype=np.float32)])
        
        for k in range(K):
            T_k = transforms[k].astype(np.float32)
            trans_k = (T_k @ homo_verts.T).T[:, :3]
            w_k = weights[:, k:k+1]
            deformed += w_k * trans_k
        return deformed

    @staticmethod
    def compute_principal_strain_axes(
        rest_verts: np.ndarray,
        quads: np.ndarray,
        weights: np.ndarray,
        flexion_transforms: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the face-level right Cauchy-Green tensor C = F^T F across
        flexion poses, and extracts the 2 principal orthogonal in-plane strain
        eigenvectors (v1, v2) for 4-RoSy directional regularization.

        Returns:
            v1_axes: (F, 3) unit principal strain axis 1
            v2_axes: (F, 3) unit principal strain axis 2 (orthogonal to v1 in tangent plane)
            C_mean: (F, 3, 3) mean Cauchy-Green tensor
        """
        num_faces = len(quads)
        def_verts = StrainMechanicsExtractor.compute_lbs(rest_verts, weights, flexion_transforms)
        
        v1_axes = np.zeros((num_faces, 3), dtype=np.float32)
        v2_axes = np.zeros((num_faces, 3), dtype=np.float32)
        C_all = np.zeros((num_faces, 3, 3), dtype=np.float32)
        
        for idx, quad in enumerate(quads):
            rv = rest_verts[quad] # (4, 3)
            dv = def_verts[quad]  # (4, 3)
            
            # Basis vectors
            d_rest_1 = rv[1] - rv[0]
            d_rest_2 = rv[3] - rv[0]
            d_def_1  = dv[1] - dv[0]
            d_def_2  = dv[3] - dv[0]
            
            norm_rest = np.cross(d_rest_1, d_rest_2)
            norm_len = np.linalg.norm(norm_rest)
            norm_rest = norm_rest / norm_len if norm_len > 1e-8 else np.array([0., 0., 1.], dtype=np.float32)
            
            norm_def = np.cross(d_def_1, d_def_2)
            norm_def_len = np.linalg.norm(norm_def)
            norm_def = norm_def / norm_def_len if norm_def_len > 1e-8 else np.array([0., 0., 1.], dtype=np.float32)
            
            V_rest = np.column_stack([d_rest_1, d_rest_2, norm_rest])
            V_def  = np.column_stack([d_def_1, d_def_2, norm_def])
            
            try:
                F = V_def @ np.linalg.inv(V_rest)
            except np.linalg.LinAlgError:
                F = np.eye(3, dtype=np.float32)
                
            C = F.T @ F
            C_all[idx] = C
            
            # Eigen-decomposition of C: C v = lambda v
            eigvals, eigvecs = np.linalg.eigh(C)
            # Principal in-plane stretch is eigenvector associated with largest stretch
            v_max = eigvecs[:, 2] # dominant stretch
            # Project onto face tangent plane
            v_max_proj = v_max - np.dot(v_max, norm_rest) * norm_rest
            len_proj = np.linalg.norm(v_max_proj)
            if len_proj > 1e-6:
                v1 = v_max_proj / len_proj
            else:
                # Fallback to local edge direction
                e0_norm = np.linalg.norm(d_rest_1)
                v1 = d_rest_1 / e0_norm if e0_norm > 1e-8 else np.array([1., 0., 0.], dtype=np.float32)
                
            # v2 is orthogonal in tangent plane: v2 = norm_rest x v1
            v2 = np.cross(norm_rest, v1)
            len_v2 = np.linalg.norm(v2)
            v2 = v2 / len_v2 if len_v2 > 1e-8 else np.array([0., 1., 0.], dtype=np.float32)
            
            v1_axes[idx] = v1
            v2_axes[idx] = v2
            
        return v1_axes, v2_axes, C_all


# ==============================================================================
# 2. Synthetic Asset Generators for Rig-Retopo-3K Archetypes
# ==============================================================================

class AssetGeneratorSuite:
    """
    Generates high-resolution surface geometry, kinematic armatures,
    skinning weights, and target artist-grade quad topologies.
    """
    @staticmethod
    def generate_planar_hinge(
        length: float = 2.0,
        width: float = 0.6,
        res_x: int = 25,
        res_y: int = 9,
        num_high_points: int = 2048
    ) -> Dict[str, Any]:
        """Benchmark A: Planar Bending Strip with 2-bone armature."""
        xs = np.linspace(-length / 2, length / 2, res_x, dtype=np.float32)
        ys = np.linspace(-width / 2, width / 2, res_y, dtype=np.float32)
        
        verts = []
        normals = []
        topos = []
        for y in ys:
            for x in xs:
                verts.append([x, y, 0.0])
                normals.append([0.0, 0.0, 1.0])
                # Harmonic topological latent coordinates in R^8
                u = (x + length / 2) / length
                v = (y + width / 2) / width
                topos.append([
                    np.cos(2 * np.pi * u), np.sin(2 * np.pi * u),
                    np.cos(4 * np.pi * u), np.sin(4 * np.pi * u),
                    v, v**2,
                    u * v, (1.0 - u) * v
                ])
        rest_verts = np.array(verts, dtype=np.float32)
        rest_normals = np.array(normals, dtype=np.float32)
        rest_topos = np.array(topos, dtype=np.float32)
        
        # Regular quad connectivity
        quads = []
        for j in range(res_y - 1):
            for i in range(res_x - 1):
                v0 = j * res_x + i
                v1 = v0 + 1
                v2 = (j + 1) * res_x + (i + 1)
                v3 = (j + 1) * res_x + i
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int64)
        
        # 2-Bone Armature: Bone 0 at (-0.5, 0, 0), Bone 1 at (0.5, 0, 0)
        joint_positions = np.array([
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0]
        ], dtype=np.float32)
        joint_parents = np.array([-1, 0], dtype=np.int64)
        skel_adj = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        
        # Smooth sigmoid skinning weights
        delta = 0.15
        x_coords = rest_verts[:, 0]
        w1 = 1.0 / (1.0 + np.exp(-np.clip(x_coords / delta, -20.0, 20.0)))
        w0 = 1.0 - w1
        skinning_weights = np.column_stack([w0, w1]).astype(np.float32)
        
        # High-poly surface point cloud (uniform sampling)
        rand_x = np.random.uniform(-length / 2, length / 2, num_high_points).astype(np.float32)
        rand_y = np.random.uniform(-width / 2, width / 2, num_high_points).astype(np.float32)
        high_points = np.column_stack([rand_x, rand_y, np.zeros(num_high_points, dtype=np.float32)])
        high_normals = np.tile([0.0, 0.0, 1.0], (num_high_points, 1)).astype(np.float32)
        
        # Flexion transform (90 degree fold around Y at origin)
        theta = np.radians(90.0)
        T0 = np.eye(4, dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        T1 = np.array([
            [c, 0., s, 0.],
            [0., 1., 0., 0.],
            [-s, 0., c, 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32)
        
        v1_axes, v2_axes, C_all = StrainMechanicsExtractor.compute_principal_strain_axes(
            rest_verts, quads, skinning_weights, [T0, T1]
        )
        
        return {
            "name": "planar_hinge",
            "rest_verts": rest_verts,
            "rest_normals": rest_normals,
            "rest_topos": rest_topos,
            "quads": quads,
            "high_points": high_points,
            "high_normals": high_normals,
            "joint_positions": joint_positions,
            "joint_parents": joint_parents,
            "skel_adj": skel_adj,
            "skinning_weights": skinning_weights,
            "v1_axes": v1_axes,
            "v2_axes": v2_axes,
            "strain_tensors": C_all
        }

    @staticmethod
    def generate_cylindrical_joint(
        radius: float = 0.4,
        height: float = 2.0,
        num_rings: int = 24,
        radial_seg: int = 16,
        num_high_points: int = 2048
    ) -> Dict[str, Any]:
        """Benchmark B: Cylindrical Articulating Joint with Concentric Loops."""
        zs = np.linspace(-height / 2, height / 2, num_rings, dtype=np.float32)
        thetas = np.linspace(0, 2 * np.pi, radial_seg, endpoint=False, dtype=np.float32)
        
        verts = []
        normals = []
        topos = []
        for z in zs:
            for th in thetas:
                x = radius * np.cos(th)
                y = radius * np.sin(th)
                verts.append([x, y, z])
                normals.append([np.cos(th), np.sin(th), 0.0])
                topos.append([
                    np.cos(th), np.sin(th),
                    np.cos(2 * th), np.sin(2 * th),
                    z / height, (z / height)**2,
                    0.5 * np.cos(th) * (z / height),
                    0.5 * np.sin(th) * (z / height)
                ])
        rest_verts = np.array(verts, dtype=np.float32)
        rest_normals = np.array(normals, dtype=np.float32)
        rest_topos = np.array(topos, dtype=np.float32)
        
        quads = []
        for i in range(num_rings - 1):
            for j in range(radial_seg):
                j_next = (j + 1) % radial_seg
                v0 = i * radial_seg + j
                v1 = i * radial_seg + j_next
                v2 = (i + 1) * radial_seg + j_next
                v3 = (i + 1) * radial_seg + j
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int64)
        
        # 2-Bone Armature along Z axis
        joint_positions = np.array([
            [0.0, 0.0, -0.6],
            [0.0, 0.0, 0.6]
        ], dtype=np.float32)
        joint_parents = np.array([-1, 0], dtype=np.int64)
        skel_adj = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        
        delta = 0.2
        z_coords = rest_verts[:, 2]
        w1 = 1.0 / (1.0 + np.exp(-np.clip(z_coords / delta, -20.0, 20.0)))
        w0 = 1.0 - w1
        skinning_weights = np.column_stack([w0, w1]).astype(np.float32)
        
        # High-res sampling
        hp_z = np.random.uniform(-height / 2, height / 2, num_high_points).astype(np.float32)
        hp_th = np.random.uniform(0, 2 * np.pi, num_high_points).astype(np.float32)
        hp_x = radius * np.cos(hp_th)
        hp_y = radius * np.sin(hp_th)
        high_points = np.column_stack([hp_x, hp_y, hp_z])
        high_normals = np.column_stack([np.cos(hp_th), np.sin(hp_th), np.zeros(num_high_points, dtype=np.float32)])
        
        # 90 degree flexion around X
        theta = np.radians(90.0)
        T0 = np.eye(4, dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        T1 = np.array([
            [1., 0., 0., 0.],
            [0., c, -s, 0.],
            [0., s, c, 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32)
        
        v1_axes, v2_axes, C_all = StrainMechanicsExtractor.compute_principal_strain_axes(
            rest_verts, quads, skinning_weights, [T0, T1]
        )
        
        return {
            "name": "cylindrical_joint",
            "rest_verts": rest_verts,
            "rest_normals": rest_normals,
            "rest_topos": rest_topos,
            "quads": quads,
            "high_points": high_points,
            "high_normals": high_normals,
            "joint_positions": joint_positions,
            "joint_parents": joint_parents,
            "skel_adj": skel_adj,
            "skinning_weights": skinning_weights,
            "v1_axes": v1_axes,
            "v2_axes": v2_axes,
            "strain_tensors": C_all
        }

    @staticmethod
    def generate_articulated_limb(
        seg_length: float = 0.8,
        radius: float = 0.35,
        num_rings_per_seg: int = 10,
        radial_seg: int = 16,
        num_high_points: int = 2048
    ) -> Dict[str, Any]:
        """Benchmark C: 3-Segment Articulated Limb (Upper Arm, Forearm, Hand)."""
        num_segs = 3
        total_rings = (num_rings_per_seg * num_segs) - (num_segs - 1)
        total_len = seg_length * num_segs
        zs = np.linspace(0.0, total_len, total_rings, dtype=np.float32)
        thetas = np.linspace(0, 2 * np.pi, radial_seg, endpoint=False, dtype=np.float32)
        
        verts = []
        normals = []
        topos = []
        for z in zs:
            # Subtle taper toward hand
            r = radius * (1.0 - 0.25 * (z / total_len))
            for th in thetas:
                x = r * np.cos(th)
                y = r * np.sin(th)
                verts.append([x, y, z])
                normals.append([np.cos(th), np.sin(th), 0.0])
                topos.append([
                    np.cos(th), np.sin(th),
                    np.cos(2 * th), np.sin(2 * th),
                    z / total_len, (z / total_len)**2,
                    0.5 * np.cos(th) * (z / total_len),
                    0.5 * np.sin(th) * (z / total_len)
                ])
        rest_verts = np.array(verts, dtype=np.float32)
        rest_normals = np.array(normals, dtype=np.float32)
        rest_topos = np.array(topos, dtype=np.float32)
        
        quads = []
        for i in range(total_rings - 1):
            for j in range(radial_seg):
                j_next = (j + 1) % radial_seg
                v0 = i * radial_seg + j
                v1 = i * radial_seg + j_next
                v2 = (i + 1) * radial_seg + j_next
                v3 = (i + 1) * radial_seg + j
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int64)
        
        # 3-Joint Armature: Shoulder (0), Elbow (1), Wrist (2)
        joint_positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, seg_length],
            [0.0, 0.0, 2 * seg_length]
        ], dtype=np.float32)
        joint_parents = np.array([-1, 0, 1], dtype=np.int64)
        
        skel_adj = np.zeros((3, 3), dtype=np.float32)
        skel_adj[0, 0] = skel_adj[0, 1] = skel_adj[1, 0] = 1.0
        skel_adj[1, 1] = skel_adj[1, 2] = skel_adj[2, 1] = 1.0
        skel_adj[2, 2] = 1.0
        
        # Multi-bone Gaussian skinning weights with partition of unity
        z_coords = rest_verts[:, 2]
        dists = np.abs(z_coords[:, None] - joint_positions[:, 2][None, :])
        w = np.exp(- (dists ** 2) / (2 * (0.35 ** 2)))
        skinning_weights = (w / np.sum(w, axis=1, keepdims=True)).astype(np.float32)
        
        # High points
        hp_z = np.random.uniform(0.0, total_len, num_high_points).astype(np.float32)
        hp_th = np.random.uniform(0, 2 * np.pi, num_high_points).astype(np.float32)
        hp_r = radius * (1.0 - 0.25 * (hp_z / total_len))
        hp_x = hp_r * np.cos(hp_th)
        hp_y = hp_r * np.sin(hp_th)
        high_points = np.column_stack([hp_x, hp_y, hp_z])
        high_normals = np.column_stack([np.cos(hp_th), np.sin(hp_th), np.zeros(num_high_points, dtype=np.float32)])
        
        # Articulation: 60 deg elbow flex, 30 deg wrist flex
        th1, th2 = np.radians(60.0), np.radians(30.0)
        T0 = np.eye(4, dtype=np.float32)
        
        c1, s1 = np.cos(th1), np.sin(th1)
        T1 = np.array([
            [1., 0., 0., 0.],
            [0., c1, -s1, 0.],
            [0., s1, c1, 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32)
        
        c2, s2 = np.cos(th2), np.sin(th2)
        T2 = np.array([
            [1., 0., 0., 0.],
            [0., c2, -s2, 0.],
            [0., s2, c2, 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32) @ T1
        
        v1_axes, v2_axes, C_all = StrainMechanicsExtractor.compute_principal_strain_axes(
            rest_verts, quads, skinning_weights, [T0, T1, T2]
        )
        
        return {
            "name": "articulated_limb",
            "rest_verts": rest_verts,
            "rest_normals": rest_normals,
            "rest_topos": rest_topos,
            "quads": quads,
            "high_points": high_points,
            "high_normals": high_normals,
            "joint_positions": joint_positions,
            "joint_parents": joint_parents,
            "skel_adj": skel_adj,
            "skinning_weights": skinning_weights,
            "v1_axes": v1_axes,
            "v2_axes": v2_axes,
            "strain_tensors": C_all
        }

    @staticmethod
    def generate_humanoid_archetype(
        num_high_points: int = 2048,
        grid_res: int = 18
    ) -> Dict[str, Any]:
        """
        Benchmark D: Humanoid SMPL-X / Mixamo Character Archetype.
        Includes a 17-joint kinematic hierarchy.
        """
        joint_names = [
            "Pelvis", "Spine", "Chest", "Neck", "Head",
            "L_Shoulder", "L_UpperArm", "L_Forearm", "L_Hand",
            "R_Shoulder", "R_UpperArm", "R_Forearm", "R_Hand",
            "L_UpLeg", "L_Leg", "R_UpLeg", "R_Leg"
        ]
        K = len(joint_names)
        
        joint_positions = np.array([
            [0.0, 0.0, 1.0],      # 0: Pelvis
            [0.0, 0.0, 1.2],      # 1: Spine
            [0.0, 0.0, 1.45],     # 2: Chest
            [0.0, 0.0, 1.6],      # 3: Neck
            [0.0, 0.0, 1.75],     # 4: Head
            [0.2, 0.0, 1.45],     # 5: L_Shoulder
            [0.38, 0.0, 1.45],    # 6: L_UpperArm
            [0.65, 0.0, 1.45],    # 7: L_Forearm
            [0.9, 0.0, 1.45],     # 8: L_Hand
            [-0.2, 0.0, 1.45],    # 9: R_Shoulder
            [-0.38, 0.0, 1.45],   # 10: R_UpperArm
            [-0.65, 0.0, 1.45],   # 11: R_Forearm
            [-0.9, 0.0, 1.45],    # 12: R_Hand
            [0.15, 0.0, 0.95],    # 13: L_UpLeg
            [0.15, 0.0, 0.5],     # 14: L_Leg
            [-0.15, 0.0, 0.95],   # 15: R_UpLeg
            [-0.15, 0.0, 0.5],    # 16: R_Leg
        ], dtype=np.float32)
        
        joint_parents = np.array([
            -1,  # 0: Pelvis
            0,   # 1: Spine -> Pelvis
            1,   # 2: Chest -> Spine
            2,   # 3: Neck -> Chest
            3,   # 4: Head -> Neck
            2,   # 5: L_Shoulder -> Chest
            5,   # 6: L_UpperArm -> L_Shoulder
            6,   # 7: L_Forearm -> L_UpperArm
            7,   # 8: L_Hand -> L_Forearm
            2,   # 9: R_Shoulder -> Chest
            9,   # 10: R_UpperArm -> R_Shoulder
            10,  # 11: R_Forearm -> R_UpperArm
            11,  # 12: R_Hand -> R_Forearm
            0,   # 13: L_UpLeg -> Pelvis
            13,  # 14: L_Leg -> L_UpLeg
            0,   # 15: R_UpLeg -> Pelvis
            15   # 16: R_Leg -> R_UpLeg
        ], dtype=np.int64)
        
        skel_adj = np.eye(K, dtype=np.float32)
        for child, parent in enumerate(joint_parents):
            if parent >= 0:
                skel_adj[child, parent] = 1.0
                skel_adj[parent, child] = 1.0
                
        torso_z = np.linspace(0.9, 1.6, 12, dtype=np.float32)
        torso_th = np.linspace(0, 2 * np.pi, 16, endpoint=False, dtype=np.float32)
        verts_list = []
        normals_list = []
        topos_list = []
        
        for z in torso_z:
            for th in torso_th:
                rx, ry = 0.22, 0.14
                x = rx * np.cos(th)
                y = ry * np.sin(th)
                verts_list.append([x, y, z])
                normals_list.append([np.cos(th), np.sin(th), 0.0])
                topos_list.append([
                    np.cos(th), np.sin(th),
                    np.cos(2 * th), np.sin(2 * th),
                    (z - 0.9) / 0.7, ((z - 0.9) / 0.7)**2,
                    0.5 * np.cos(th) * ((z - 0.9) / 0.7),
                    0.5 * np.sin(th) * ((z - 0.9) / 0.7)
                ])
                
        rest_verts = np.array(verts_list, dtype=np.float32)
        rest_normals = np.array(normals_list, dtype=np.float32)
        rest_topos = np.array(topos_list, dtype=np.float32)
        
        quads = []
        for i in range(11):
            for j in range(16):
                j_next = (j + 1) % 16
                v0 = i * 16 + j
                v1 = i * 16 + j_next
                v2 = (i + 1) * 16 + j_next
                v3 = (i + 1) * 16 + j
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int64)
        
        N = len(rest_verts)
        dists = np.linalg.norm(rest_verts[:, None, :] - joint_positions[None, :, :], axis=-1)
        inv_dists = np.exp(-dists / 0.3)
        skinning_weights = (inv_dists / np.sum(inv_dists, axis=-1, keepdims=True)).astype(np.float32)
        
        hp_u = np.random.uniform(0, 2 * np.pi, num_high_points).astype(np.float32)
        hp_v = np.random.uniform(0.9, 1.6, num_high_points).astype(np.float32)
        hp_x = 0.22 * np.cos(hp_u)
        hp_y = 0.14 * np.sin(hp_u)
        high_points = np.column_stack([hp_x, hp_y, hp_v])
        high_normals = np.column_stack([np.cos(hp_u), np.sin(hp_u), np.zeros(num_high_points, dtype=np.float32)])
        
        transforms = [np.eye(4, dtype=np.float32) for _ in range(K)]
        th_spine = np.radians(30.0)
        c, s = np.cos(th_spine), np.sin(th_spine)
        T_bend = np.array([
            [1., 0., 0., 0.],
            [0., c, -s, 0.],
            [0., s, c, 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32)
        for j_idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            transforms[j_idx] = T_bend
            
        v1_axes, v2_axes, C_all = StrainMechanicsExtractor.compute_principal_strain_axes(
            rest_verts, quads, skinning_weights, transforms
        )
        
        return {
            "name": "humanoid_smplx",
            "rest_verts": rest_verts,
            "rest_normals": rest_normals,
            "rest_topos": rest_topos,
            "quads": quads,
            "high_points": high_points,
            "high_normals": high_normals,
            "joint_positions": joint_positions,
            "joint_parents": joint_parents,
            "skel_adj": skel_adj,
            "skinning_weights": skinning_weights,
            "v1_axes": v1_axes,
            "v2_axes": v2_axes,
            "strain_tensors": C_all
        }


# ==============================================================================
# 3. PyTorch Dataset & Batch Collation
# ==============================================================================

class RigRetopoDataset(Dataset):
    """
    Production-grade PyTorch Dataset for paired deformation-aware retopology.
    """
    def __init__(
        self,
        num_samples: int = 100,
        archetypes: Optional[List[str]] = None,
        num_high_points: int = 2048,
        noise_std: float = 0.01,
        seed: int = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_high_points = num_high_points
        self.noise_std = noise_std
        self.archetypes = archetypes or ["planar_hinge", "cylindrical_joint", "articulated_limb", "humanoid_smplx"]
        
        np.random.seed(seed)
        self.templates = {
            "planar_hinge": AssetGeneratorSuite.generate_planar_hinge(num_high_points=num_high_points),
            "cylindrical_joint": AssetGeneratorSuite.generate_cylindrical_joint(num_high_points=num_high_points),
            "articulated_limb": AssetGeneratorSuite.generate_articulated_limb(num_high_points=num_high_points),
            "humanoid_smplx": AssetGeneratorSuite.generate_humanoid_archetype(num_high_points=num_high_points)
        }
        
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        archetype_key = self.archetypes[idx % len(self.archetypes)]
        tpl = self.templates[archetype_key]
        
        scale = float(np.random.uniform(0.9, 1.1))
        rot_angle = float(np.random.uniform(-math.pi, math.pi))
        c_rot, s_rot = math.cos(rot_angle), math.sin(rot_angle)
        R_z = np.array([
            [c_rot, -s_rot, 0.0],
            [s_rot,  c_rot, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)
        
        rest_verts = (tpl["rest_verts"] @ R_z.T) * scale
        rest_normals = tpl["rest_normals"] @ R_z.T
        high_points = (tpl["high_points"] @ R_z.T) * scale
        high_normals = tpl["high_normals"] @ R_z.T
        joint_positions = (tpl["joint_positions"] @ R_z.T) * scale
        v1_axes = tpl["v1_axes"] @ R_z.T
        v2_axes = tpl["v2_axes"] @ R_z.T
        
        if self.noise_std > 0:
            high_points += np.random.normal(0, self.noise_std, high_points.shape).astype(np.float32)
            
        s_1 = np.hstack([rest_verts, rest_normals, tpl["rest_topos"]]).astype(np.float32)
        
        return {
            "name": archetype_key,
            "s_1": torch.from_numpy(s_1),
            "rest_verts": torch.from_numpy(rest_verts),
            "rest_normals": torch.from_numpy(rest_normals),
            "high_points": torch.from_numpy(high_points),
            "high_normals": torch.from_numpy(high_normals),
            "joint_positions": torch.from_numpy(joint_positions),
            "joint_parents": torch.from_numpy(tpl["joint_parents"]),
            "skel_adj": torch.from_numpy(tpl["skel_adj"]),
            "skinning_weights": torch.from_numpy(tpl["skinning_weights"]),
            "quad_indices": torch.from_numpy(tpl["quads"]),
            "v1_axes": torch.from_numpy(v1_axes),
            "v2_axes": torch.from_numpy(v2_axes),
            "strain_tensors": torch.from_numpy(tpl["strain_tensors"])
        }


def rig_retopo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_same_shape = len(set(item["s_1"].shape[0] for item in batch)) == 1
    
    if all_same_shape:
        return {
            "name": [item["name"] for item in batch],
            "s_1": torch.stack([item["s_1"] for item in batch], dim=0),
            "rest_verts": torch.stack([item["rest_verts"] for item in batch], dim=0),
            "rest_normals": torch.stack([item["rest_normals"] for item in batch], dim=0),
            "high_points": torch.stack([item["high_points"] for item in batch], dim=0),
            "high_normals": torch.stack([item["high_normals"] for item in batch], dim=0),
            "joint_positions": torch.stack([item["joint_positions"] for item in batch], dim=0),
            "joint_parents": torch.stack([item["joint_parents"] for item in batch], dim=0),
            "skel_adj": torch.stack([item["skel_adj"] for item in batch], dim=0),
            "skinning_weights": torch.stack([item["skinning_weights"] for item in batch], dim=0),
            "quad_indices": batch[0]["quad_indices"],
            "v1_axes": torch.stack([item["v1_axes"] for item in batch], dim=0),
            "v2_axes": torch.stack([item["v2_axes"] for item in batch], dim=0),
            "strain_tensors": torch.stack([item["strain_tensors"] for item in batch], dim=0)
        }
    else:
        return {
            "name": [item["name"] for item in batch],
            "items": batch
        }


# ==============================================================================
# 4. Self-Test & Demonstration Suite
# ==============================================================================

def run_dataset_pipeline_test():
    print("=" * 85)
    print("RUNNING WP 3.1: RIG-RETOPO-3K DATASET INGESTION & PIPELINE TEST")
    print("=" * 85)
    
    print("\n[TEST 1] Generating and Validating Asset Archetypes...")
    suite = AssetGeneratorSuite()
    
    archetypes = [
        ("Benchmark A: Planar Hinge", suite.generate_planar_hinge()),
        ("Benchmark B: Cylindrical Joint", suite.generate_cylindrical_joint()),
        ("Benchmark C: Articulated Limb", suite.generate_articulated_limb()),
        ("Benchmark D: Humanoid SMPL-X", suite.generate_humanoid_archetype())
    ]
    
    for label, data in archetypes:
        N = len(data["rest_verts"])
        F = len(data["quads"])
        K = len(data["joint_positions"])
        M = len(data["high_points"])
        
        weight_sums = np.sum(data["skinning_weights"], axis=1)
        assert np.allclose(weight_sums, 1.0, atol=1e-5), f"Skinning partition of unity failed for {label}!"
        
        dets = np.linalg.det(data["strain_tensors"])
        assert np.all(dets > 0.0), f"Cauchy-Green strain tensor has non-positive determinant in {label}!"
        
        dots = np.sum(data["v1_axes"] * data["v2_axes"], axis=1)
        assert np.all(np.abs(dots) < 1e-4), f"Principal axes v1 and v2 not orthogonal in {label}!"
        
        print(f"  ✓ {label:<32}: N={N:>4} verts | F={F:>4} quads | K={K:>2} joints | M={M:>4} high-pts | w_sum=1.000")
        
    print("\n[TEST 2] Testing PyTorch RigRetopoDataset & DataLoader Batching...")
    dataset_cyl = RigRetopoDataset(num_samples=16, archetypes=["cylindrical_joint"])
    loader_cyl = DataLoader(dataset_cyl, batch_size=4, shuffle=True, collate_fn=rig_retopo_collate_fn)
    
    batch = next(iter(loader_cyl))
    print(f"  Batch Size:             {batch['s_1'].shape[0]}")
    print(f"  Target State s_1 Shape: {batch['s_1'].shape} (B, N, 14)")
    print(f"  High-Res Points Shape:  {batch['high_points'].shape} (B, M, 3)")
    print(f"  Joint Positions Shape:  {batch['joint_positions'].shape} (B, K, 3)")
    print(f"  Skinning Weights Shape: {batch['skinning_weights'].shape} (B, N, K)")
    print(f"  Strain Axes v1 Shape:   {batch['v1_axes'].shape} (B, F, 3)")
    
    assert batch['s_1'].shape == (4, 384, 14)
    assert batch['high_points'].shape == (4, 2048, 3)
    print("  -> Batch tensor dimensions match theoretical DiT conditioning specifications.")
    
    print("\n" + "=" * 85)
    print("WP 3.1 RIG-RETOPO-3K DATASET PIPELINE VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_dataset_pipeline_test()