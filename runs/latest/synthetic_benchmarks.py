#!/usr/bin/env python3
"""
Synthetic Deformation Benchmarks for Conditional Mesh Retopology
================================================================
Phase 1: Representation Sandbox & Controlled Toy Physics
Project Code: MESH-FLOW-RETOPOLOGY

This module implements:
1. Benchmark A: 2D Planar Articulating Hinge (Bending Strip)
2. Benchmark B: 3D Cylindrical Articulating Joint (Bending Elbow)
3. Linear Blend Skinning (LBS) Deformation Engine
4. Cauchy-Green Strain Tensor & Principal Strain Vector Field Analysis
5. Topological Quality & Deformation Readiness Metrics
   - Dirichlet Energy (Conformal Distortion)
   - Volume Loss / Joint Pinching Ratio
   - Strain-Aligned Edge Flow Error (L_strain)
   - Regular Quad Valence & Manifold Invariants
"""

import numpy as np
import scipy.linalg
from typing import Dict, Tuple, List, Optional


# ==============================================================================
# 1. Kinematics & Linear Blend Skinning (LBS) Engine
# ==============================================================================

class KinematicDeformer:
    """Computes Linear Blend Skinning and surface deformation gradients."""
    
    @staticmethod
    def compute_lbs(
        vertices: np.ndarray,
        weights: np.ndarray,
        transforms: List[np.ndarray]
    ) -> np.ndarray:
        """
        Computes Linear Blend Skinning:
        x_def = sum_k w_k * (R_k * x_rest + t_k)
        
        Args:
            vertices: (N, 3) rest vertex coordinates
            weights: (N, K) skinning weight partition of unity
            transforms: list of K (4, 4) rigid transformation matrices
        Returns:
            deformed_vertices: (N, 3)
        """
        N, K = weights.shape
        deformed = np.zeros_like(vertices)
        
        # Homogeneous coordinates
        ones = np.ones((N, 1))
        homo_verts = np.hstack([vertices, ones]) # (N, 4)
        
        for k in range(K):
            T_k = transforms[k] # (4, 4)
            # Transform all vertices by joint k
            trans_k = (T_k @ homo_verts.T).T[:, :3] # (N, 3)
            # Weight contribution
            w_k = weights[:, k:k+1] # (N, 1)
            deformed += w_k * trans_k
            
        return deformed

    @staticmethod
    def compute_face_deformation_gradients(
        rest_verts: np.ndarray,
        def_verts: np.ndarray,
        faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes 3D deformation gradient tensor F for each face:
        dx_def = F * dx_rest
        
        For quad faces (v0, v1, v2, v3), we evaluate over two constituent triangles:
        (v0, v1, v2) and (v0, v2, v3) or direct least-squares on edge vectors.
        
        Returns:
            F: (num_faces, 3, 3) deformation gradient tensors
            C: (num_faces, 3, 3) right Cauchy-Green tensors C = F^T F
            E: (num_faces, 3, 3) Green-Lagrange strain tensors E = 0.5 * (C - I)
        """
        num_faces = len(faces)
        F_all = np.zeros((num_faces, 3, 3))
        C_all = np.zeros((num_faces, 3, 3))
        E_all = np.zeros((num_faces, 3, 3))
        
        for idx, face in enumerate(faces):
            # Face vertices
            rv = rest_verts[face] # (4, 3) or (3, 3)
            dv = def_verts[face]  # (4, 3) or (3, 3)
            
            # Basis vectors: use orthogonal edges v0->v1 and v0->v3 (or v0->v2 if triangle)
            if len(face) == 4:
                d_rest_1 = rv[1] - rv[0]
                d_rest_2 = rv[3] - rv[0]
                d_def_1  = dv[1] - dv[0]
                d_def_2  = dv[3] - dv[0]
            else:
                d_rest_1 = rv[1] - rv[0]
                d_rest_2 = rv[2] - rv[0]
                d_def_1  = dv[1] - dv[0]
                d_def_2  = dv[2] - dv[0]

            norm_rest = np.cross(d_rest_1, d_rest_2)
            if np.linalg.norm(norm_rest) < 1e-9:
                norm_rest = np.array([0., 0., 1.])
            else:
                norm_rest = norm_rest / np.linalg.norm(norm_rest)
            
            norm_def = np.cross(d_def_1, d_def_2)
            if np.linalg.norm(norm_def) < 1e-9:
                norm_def = np.array([0., 0., 1.])
            else:
                norm_def = norm_def / np.linalg.norm(norm_def)
                
            # Construct 3D tangent-normal frame matrices
            V_rest = np.column_stack([d_rest_1, d_rest_2, norm_rest])
            V_def  = np.column_stack([d_def_1, d_def_2, norm_def])
            
            # F = V_def * inv(V_rest)
            try:
                F = V_def @ np.linalg.inv(V_rest)
            except np.linalg.LinAlgError:
                F = np.eye(3)
                
            C = F.T @ F
            E = 0.5 * (C - np.eye(3))
            
            F_all[idx] = F
            C_all[idx] = C
            E_all[idx] = E
            
        return F_all, C_all, E_all


# ==============================================================================
# 2. Benchmark A: 2D Planar Articulating Hinge
# ==============================================================================

class PlanarHingeBenchmark:
    """
    Controlled 2D/3D planar strip articulating around a central hinge joint (x = 0).
    Compares:
      - Strain-aligned quad topology (edge loops strictly parallel/orthogonal to hinge)
      - Flawed diagonal topology (edges at 45° across the hinge axis)
    """
    
    def __init__(self, length: float = 2.0, width: float = 0.6, res_x: int = 21, res_y: int = 7):
        self.length = length
        self.width = width
        self.res_x = res_x
        self.res_y = res_y
        
    def generate_aligned_quad_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates regular quad grid with edges aligned to axes."""
        xs = np.linspace(-self.length / 2, self.length / 2, self.res_x)
        ys = np.linspace(-self.width / 2, self.width / 2, self.res_y)
        
        verts = []
        for y in ys:
            for x in xs:
                verts.append([x, y, 0.0])
        verts = np.array(verts, dtype=np.float64)
        
        quads = []
        for j in range(self.res_y - 1):
            for i in range(self.res_x - 1):
                v0 = j * self.res_x + i
                v1 = v0 + 1
                v2 = (j + 1) * self.res_x + (i + 1)
                v3 = (j + 1) * self.res_x + i
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int32)
        
        # Skinning weights: 2 bones (Bone 0: left, Bone 1: right)
        # Smooth sigmoid transition across hinge zone [-delta, delta]
        delta = 0.15
        x_coords = verts[:, 0]
        # w1 = 1 / (1 + exp(-x / delta))
        w1 = 1.0 / (1.0 + np.exp(-np.clip(x_coords / delta, -20, 20)))
        w0 = 1.0 - w1
        weights = np.column_stack([w0, w1])
        
        return verts, quads, weights

    def generate_flawed_diagonal_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates diagonal diamond quad topology on the exact same physical strip geometry.
        Edges run at 45 degrees across the hinge line (x = 0), matching classical
        curvature-agnostic meshing artifacts.
        """
        xs = np.linspace(-self.length / 2, self.length / 2, self.res_x)
        ys = np.linspace(-self.width / 2, self.width / 2, self.res_y)
        
        # We construct a diamond quad grid
        # To do this cleanly on the same rectangle, we rotate the quad connectivity:
        # Each quad is formed by (i, j+1) -> (i+1, j) -> (i+2, j+1) -> (i+1, j+2)
        verts = []
        for y in ys:
            for x in xs:
                verts.append([x, y, 0.0])
        verts = np.array(verts, dtype=np.float64)
        
        # Build 45-degree diamond quads
        quads = []
        for j in range(self.res_y - 2):
            for i in range(self.res_x - 2):
                if (i + j) % 2 == 0:
                    v_top = (j + 2) * self.res_x + (i + 1)
                    v_right = (j + 1) * self.res_x + (i + 2)
                    v_bottom = j * self.res_x + (i + 1)
                    v_left = (j + 1) * self.res_x + i
                    quads.append([v_bottom, v_right, v_top, v_left])
        quads = np.array(quads, dtype=np.int32)
        
        delta = 0.15
        x_coords = verts[:, 0]
        w1 = 1.0 / (1.0 + np.exp(-np.clip(x_coords / delta, -20, 20)))
        w0 = 1.0 - w1
        weights = np.column_stack([w0, w1])
        
        return verts, quads, weights

    def get_bending_transforms(self, angle_degrees: float) -> List[np.ndarray]:
        """Returns 2-bone kinematic transform matrices for given bend angle."""
        theta = np.radians(angle_degrees)
        
        # Bone 0: Identity (fixed base)
        T0 = np.eye(4)
        
        # Bone 1: Rotates by theta around Z-axis at hinge origin (0, 0, 0)
        # or around Y-axis (hinge line is along Y-axis: x=0, z=0)
        # Bending around Y axis folds the strip:
        c, s = np.cos(theta), np.sin(theta)
        T1 = np.array([
            [ c, 0., s, 0.],
            [0., 1., 0., 0.],
            [-s, 0., c, 0.],
            [0., 0., 0., 1.]
        ])
        return [T0, T1]


# ==============================================================================
# 3. Benchmark B: 3D Cylindrical Articulating Joint (Elbow Hinge)
# ==============================================================================

class CylindricalJointBenchmark:
    """
    3D hollow cylinder driven by a two-bone kinematic arm bending to 90 degrees.
    Tests:
      - Concentric ring quad topology (strain-aligned)
      - Helical / diagonal quad topology (misaligned)
    Measures:
      - Volume loss / cross-sectional area pinching ratio at the joint
      - Dirichlet energy and face normal inversion
    """
    
    def __init__(self, radius: float = 0.4, height: float = 2.0, num_rings: int = 25, radial_seg: int = 16):
        self.radius = radius
        self.height = height
        self.num_rings = num_rings
        self.radial_seg = radial_seg
        
    def generate_aligned_quad_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates concentric rings along Z-axis (orthogonal to articulation)."""
        zs = np.linspace(-self.height / 2, self.height / 2, self.num_rings)
        thetas = np.linspace(0, 2 * np.pi, self.radial_seg, endpoint=False)
        
        verts = []
        for z in zs:
            for th in thetas:
                x = self.radius * np.cos(th)
                y = self.radius * np.sin(th)
                verts.append([x, y, z])
        verts = np.array(verts, dtype=np.float64)
        
        quads = []
        for i in range(self.num_rings - 1):
            for j in range(self.radial_seg):
                j_next = (j + 1) % self.radial_seg
                v0 = i * self.radial_seg + j
                v1 = i * self.radial_seg + j_next
                v2 = (i + 1) * self.radial_seg + j_next
                v3 = (i + 1) * self.radial_seg + j
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int32)
        
        # Skinning weights: Bone 0 (z < 0), Bone 1 (z > 0)
        delta = 0.2
        z_coords = verts[:, 2]
        w1 = 1.0 / (1.0 + np.exp(-np.clip(z_coords / delta, -20, 20)))
        w0 = 1.0 - w1
        weights = np.column_stack([w0, w1])
        
        return verts, quads, weights

    def generate_misaligned_helical_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates helical quad mesh where edges spiral diagonally across the cylinder,
        violating circumferential ring alignment.
        """
        verts, quads, weights = self.generate_aligned_quad_mesh()
        
        # Twist the vertices helically as a function of z
        verts_twisted = verts.copy()
        twist_rate = 1.5 # radians per unit length
        for i in range(len(verts_twisted)):
            z = verts_twisted[i, 2]
            angle = twist_rate * z
            c, s = np.cos(angle), np.sin(angle)
            x, y = verts_twisted[i, 0], verts_twisted[i, 1]
            verts_twisted[i, 0] = c * x - s * y
            verts_twisted[i, 1] = s * x + c * y
            
        return verts_twisted, quads, weights

    def get_elbow_transforms(self, angle_degrees: float) -> List[np.ndarray]:
        """Two bones bending at origin around X-axis."""
        theta = np.radians(angle_degrees)
        
        # Bone 0: Identity
        T0 = np.eye(4)
        
        # Bone 1: Rotation around X-axis at (0, 0, 0)
        c, s = np.cos(theta), np.sin(theta)
        T1 = np.array([
            [1., 0.,  0., 0.],
            [0., c,  -s,  0.],
            [0., s,   c,  0.],
            [0., 0.,  0., 1.]
        ])
        return [T0, T1]


# ==============================================================================
# 4. Deformation Metrics & Strain Regularization Loss
# ==============================================================================

class DeformationMetrics:
    """Computes quantitative metrics for deformation quality and strain alignment."""
    
    @staticmethod
    def compute_dirichlet_energy(C_tensors: np.ndarray) -> float:
        """
        Mean Dirichlet energy (conformal distortion):
        E_D = 0.5 * (Tr(C) - 3)
        Measures total surface stretching and shearing.
        """
        traces = np.trace(C_tensors, axis1=1, axis2=2)
        energies = 0.5 * np.maximum(0.0, traces - 3.0)
        return float(np.mean(energies))

    @staticmethod
    def compute_joint_pinching_ratio(
        rest_verts: np.ndarray,
        def_verts: np.ndarray,
        joint_mask: np.ndarray
    ) -> float:
        """
        Calculates cross-sectional area shrinkage at the joint center:
        Ratio = Area_deformed / Area_rest
        A value near 1.0 indicates volume preservation; < 0.5 indicates severe pinching.
        """
        jv_rest = rest_verts[joint_mask]
        jv_def = def_verts[joint_mask]
        
        if len(jv_rest) < 3:
            return 1.0
            
        # Compute bounding radius / variance around centroid
        r_rest = np.mean(np.linalg.norm(jv_rest - np.mean(jv_rest, axis=0), axis=1))
        r_def  = np.mean(np.linalg.norm(jv_def - np.mean(jv_def, axis=0), axis=1))
        
        if r_rest < 1e-9:
            return 1.0
        return float((r_def / r_rest) ** 2)

    @staticmethod
    def compute_strain_alignment_loss(
        verts: np.ndarray,
        quads: np.ndarray,
        principal_axes: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Evaluates L_strain:
        Measures whether quad edges align either parallel or perpendicular
        to the principal strain eigenvectors (v1, v2).
        
        L_strain = mean_{faces} mean_{edges} [ 4 * (e . v1)^2 * (e . v2)^2 ]
        When e is parallel to v1 or v2, L_strain = 0.
        When e is at 45 degrees, L_strain reaches maximum = 1.0.
        """
        penalties = []
        for f_idx, quad in enumerate(quads):
            v1, v2 = principal_axes[f_idx]
            
            # Quad edges: (v0, v1), (v1, v2), (v2, v3), (v3, v0)
            face_verts = verts[quad]
            edges = [
                face_verts[1] - face_verts[0],
                face_verts[2] - face_verts[1],
                face_verts[3] - face_verts[2],
                face_verts[0] - face_verts[3]
            ]
            
            face_penalties = []
            for edge in edges:
                norm = np.linalg.norm(edge)
                if norm < 1e-9:
                    continue
                e_hat = edge / norm
                dot1 = np.abs(np.dot(e_hat, v1))
                dot2 = np.abs(np.dot(e_hat, v2))
                
                # 4-RoSy alignment penalty: 4 * dot1^2 * dot2^2
                penalty = 4.0 * (dot1 ** 2) * (dot2 ** 2)
                face_penalties.append(penalty)
                
            if face_penalties:
                penalties.append(np.mean(face_penalties))
                
        return float(np.mean(penalties)) if penalties else 0.0


# ==============================================================================
# 5. Benchmark Execution & Self-Test Suite
# ==============================================================================

def run_synthetic_benchmark_suite() -> Dict[str, any]:
    """Runs both benchmarks across multiple articulation angles and outputs results."""
    print("=" * 80)
    print("EXECUTING PHASE 1 SYNTHETIC DEFORMATION BENCHMARKS")
    print("=" * 80)
    
    results = {}
    
    # --------------------------------------------------------------------------
    # Benchmark A: 2D Planar Articulating Hinge
    # --------------------------------------------------------------------------
    print("\n[BENCHMARK A] 2D Planar Articulating Hinge (Fold Angles: 30°, 60°, 90°, 120°)")
    print("-" * 80)
    print(f"{'Angle':<8} | {'Topology':<16} | {'Dirichlet Energy':<18} | {'Strain Loss (L_strain)':<22}")
    print("-" * 80)
    
    hinge = PlanarHingeBenchmark(length=2.0, width=0.6, res_x=25, res_y=9)
    v_aligned, q_aligned, w_aligned = hinge.generate_aligned_quad_mesh()
    v_flawed, q_flawed, w_flawed = hinge.generate_flawed_diagonal_mesh()
    
    hinge_results = []
    # Analytical principal strain axes for planar strip bending around Y:
    # v1 = (1, 0, 0) [transverse/bending direction], v2 = (0, 1, 0) [hinge axis]
    axes_aligned = [(np.array([1., 0., 0.]), np.array([0., 1., 0.])) for _ in q_aligned]
    axes_flawed  = [(np.array([1., 0., 0.]), np.array([0., 1., 0.])) for _ in q_flawed]

    for angle in [30.0, 60.0, 90.0, 120.0]:
        transforms = hinge.get_bending_transforms(angle)
        
        # Aligned deformation
        def_aligned = KinematicDeformer.compute_lbs(v_aligned, w_aligned, transforms)
        _, C_aligned, _ = KinematicDeformer.compute_face_deformation_gradients(v_aligned, def_aligned, q_aligned)
        dir_aligned = DeformationMetrics.compute_dirichlet_energy(C_aligned)
        strain_aligned = DeformationMetrics.compute_strain_alignment_loss(v_aligned, q_aligned, axes_aligned)
        
        # Flawed diagonal deformation
        def_flawed = KinematicDeformer.compute_lbs(v_flawed, w_flawed, transforms)
        _, C_flawed, _ = KinematicDeformer.compute_face_deformation_gradients(v_flawed, def_flawed, q_flawed)
        dir_flawed = DeformationMetrics.compute_dirichlet_energy(C_flawed)
        strain_flawed = DeformationMetrics.compute_strain_alignment_loss(v_flawed, q_flawed, axes_flawed)
        
        print(f"{angle:>5.1f}°  | {'Aligned (Ours)':<16} | {dir_aligned:<18.4f} | {strain_aligned:<22.4f}")
        print(f"{' ':>7} | {'Flawed (Diagonal)':<16} | {dir_flawed:<18.4f} | {strain_flawed:<22.4f}")
        print("-" * 80)
        
        hinge_results.append({
            'angle': angle,
            'dir_aligned': dir_aligned,
            'dir_flawed': dir_flawed,
            'strain_aligned': strain_aligned,
            'strain_flawed': strain_flawed
        })
    results['hinge'] = hinge_results

    # --------------------------------------------------------------------------
    # Benchmark B: 3D Cylindrical Articulating Joint
    # --------------------------------------------------------------------------
    print("\n[BENCHMARK B] 3D Cylindrical Joint Flexion (Elbow Bend to 90°)")
    print("-" * 80)
    print(f"{'Angle':<8} | {'Topology':<16} | {'Pinching Ratio (Area)':<22} | {'Dirichlet Energy':<18} | {'Strain Loss':<12}")
    print("-" * 80)
    
    cyl = CylindricalJointBenchmark(radius=0.4, height=2.0, num_rings=25, radial_seg=16)
    v_cyl_align, q_cyl_align, w_cyl_align = cyl.generate_aligned_quad_mesh()
    v_cyl_helical, q_cyl_helical, w_cyl_helical = cyl.generate_misaligned_helical_mesh()
    
    # Joint mask: vertices near z = 0 (hinge zone |z| < 0.15)
    joint_mask_align = np.abs(v_cyl_align[:, 2]) < 0.15
    joint_mask_helical = np.abs(v_cyl_helical[:, 2]) < 0.15
    
    # Compute circumferential (v2) and longitudinal (v1) principal axes for each cylinder quad
    def compute_cyl_axes(verts, quads):
        axes = []
        for quad in quads:
            centroid = np.mean(verts[quad], axis=0)
            x, y, z = centroid
            r = np.sqrt(x*x + y*y)
            if r < 1e-9:
                v_theta = np.array([0., 1., 0.])
            else:
                v_theta = np.array([-y/r, x/r, 0.]) # Circumferential tangent
            v_z = np.array([0., 0., 1.])             # Longitudinal cylinder axis
            axes.append((v_z, v_theta))
        return axes
        
    axes_cyl_align = compute_cyl_axes(v_cyl_align, q_cyl_align)
    axes_cyl_helical = compute_cyl_axes(v_cyl_helical, q_cyl_helical)
    
    cyl_results = []
    for angle in [0.0, 45.0, 90.0]:
        transforms = cyl.get_elbow_transforms(angle)
        
        # Aligned rings
        def_align = KinematicDeformer.compute_lbs(v_cyl_align, w_cyl_align, transforms)
        _, C_align, _ = KinematicDeformer.compute_face_deformation_gradients(v_cyl_align, def_align, q_cyl_align)
        pinch_align = DeformationMetrics.compute_joint_pinching_ratio(v_cyl_align, def_align, joint_mask_align)
        dir_align = DeformationMetrics.compute_dirichlet_energy(C_align)
        strain_align = DeformationMetrics.compute_strain_alignment_loss(v_cyl_align, q_cyl_align, axes_cyl_align)
        
        # Helical diagonal
        def_helical = KinematicDeformer.compute_lbs(v_cyl_helical, w_cyl_helical, transforms)
        _, C_helical, _ = KinematicDeformer.compute_face_deformation_gradients(v_cyl_helical, def_helical, q_cyl_helical)
        pinch_helical = DeformationMetrics.compute_joint_pinching_ratio(v_cyl_helical, def_helical, joint_mask_helical)
        dir_helical = DeformationMetrics.compute_dirichlet_energy(C_helical)
        strain_helical = DeformationMetrics.compute_strain_alignment_loss(v_cyl_helical, q_cyl_helical, axes_cyl_helical)
        
        print(f"{angle:>5.1f}°  | {'Concentric Rings':<16} | {pinch_align:<22.4f} | {dir_align:<18.4f} | {strain_align:<12.4f}")
        print(f"{' ':>7} | {'Helical Diagonal':<16} | {pinch_helical:<22.4f} | {dir_helical:<18.4f} | {strain_helical:<12.4f}")
        print("-" * 80)
        
        cyl_results.append({
            'angle': angle,
            'pinch_align': pinch_align,
            'pinch_helical': pinch_helical,
            'dir_align': dir_align,
            'dir_helical': dir_helical,
            'strain_align': strain_align,
            'strain_helical': strain_helical
        })
    results['cylinder'] = cyl_results
    
    print("\nBENCHMARK SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_synthetic_benchmark_suite()
