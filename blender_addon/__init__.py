#!/usr/bin/env python3
"""Blender sidebar tool: put quad rings on a long thin mesh.

Not a neural model. Outside Blender, running this file checks that the
same cutter used by the button still produces a quad tube.
"""

bl_info = {
    "name": "环线重拓扑",
    "author": "mesh retopo",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > 环线",
    "description": "顺着细长模型套一圈圈四边面。不是神经网络。",
    "warning": "分叉的身体切不好。",
    "doc_url": "https://github.com/siddhartha-yz/deformation-aware-retopology",
    "category": "Mesh",
}

import sys
import math
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Detect whether running inside Blender runtime or standalone
try:
    import bpy
    from bpy.props import (
        IntProperty,
        FloatProperty,
        EnumProperty,
        BoolProperty,
        PointerProperty,
        StringProperty
    )
    from bpy.types import Panel, Operator, PropertyGroup
    RUNNING_IN_BLENDER = True
except ImportError:
    RUNNING_IN_BLENDER = False


# ==============================================================================
# 1. Core Flow Retopology Inference Engine (Blender-Independent)
# ==============================================================================

class RetopoInferenceEngine:
    """
    Builds a regular quad cylinder inside the target bounding box and
    integrates the closed-form field s_1 - s_0. Not a neural retopology model.
    """
    @staticmethod
    def extract_armature_joints(armature_obj: Any) -> Tuple[np.ndarray, List[str], List[int]]:
        joint_positions = []
        joint_names = []
        parent_map = []
        
        if RUNNING_IN_BLENDER and armature_obj and armature_obj.type == 'ARMATURE':
            bones = armature_obj.data.bones
            bone_name_to_idx = {bone.name: idx for idx, bone in enumerate(bones)}
            
            for bone in bones:
                head_world = armature_obj.matrix_world @ bone.head_local
                joint_positions.append([head_world.x, head_world.y, head_world.z])
                joint_names.append(bone.name)
                
                if bone.parent and bone.parent.name in bone_name_to_idx:
                    parent_map.append(bone_name_to_idx[bone.parent.name])
                else:
                    parent_map.append(-1)
        else:
            joint_positions = [[0.0, 0.0, -0.6], [0.0, 0.0, 0.6]]
            joint_names = ["Bone_Root", "Bone_Limb"]
            parent_map = [-1, 0]
            
        return (
            np.array(joint_positions, dtype=np.float32),
            joint_names,
            parent_map
        )

    @staticmethod
    def sample_surface_points(mesh_obj: Any, num_points: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        if RUNNING_IN_BLENDER and mesh_obj and mesh_obj.type == 'MESH':
            mesh = mesh_obj.data
            matrix_world = mesh_obj.matrix_world
            
            verts = [matrix_world @ v.co for v in mesh.vertices]
            normals = [matrix_world.to_3x3() @ v.normal for v in mesh.vertices]
            
            num_verts = len(verts)
            if num_verts == 0:
                return np.zeros((num_points, 3)), np.zeros((num_points, 3))
                
            indices = np.random.choice(num_verts, size=num_points, replace=(num_verts < num_points))
            sampled_pts = np.array([[verts[i].x, verts[i].y, verts[i].z] for i in indices], dtype=np.float32)
            sampled_nrms = np.array([[normals[i].x, normals[i].y, normals[i].z] for i in indices], dtype=np.float32)
            lens = np.linalg.norm(sampled_nrms, axis=1, keepdims=True)
            sampled_nrms = np.where(lens > 1e-6, sampled_nrms / lens, np.array([[0.0, 0.0, 1.0]]))
            return sampled_pts, sampled_nrms
        else:
            zs = np.random.uniform(-1.0, 1.0, num_points)
            ths = np.random.uniform(0, 2 * np.pi, num_points)
            r = 0.4
            sampled_pts = np.column_stack([r * np.cos(ths), r * np.sin(ths), zs]).astype(np.float32)
            sampled_nrms = np.column_stack([np.cos(ths), np.sin(ths), np.zeros(num_points)]).astype(np.float32)
            return sampled_pts, sampled_nrms

    @staticmethod
    def generate_quad_topology(
        target_pts: np.ndarray,
        joint_positions: np.ndarray,
        num_rings: int = 16,
        radial_seg: int = 16,
        solver_type: str = "Midpoint",
        ode_steps: int = 10,
        lambda_strain: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_bound = np.min(target_pts, axis=0)
        max_bound = np.max(target_pts, axis=0)
        extent = max_bound - min_bound
        center = 0.5 * (min_bound + max_bound)
        
        primary_axis = int(np.argmax(extent))
        ortho_axes = [ax for ax in [0, 1, 2] if ax != primary_axis]
        
        len_axis = max(0.1, extent[primary_axis])
        r_ortho = 0.5 * max(extent[ortho_axes[0]], extent[ortho_axes[1]])
        r_ortho = max(0.05, r_ortho)
        
        N = num_rings * radial_seg
        F = (num_rings - 1) * radial_seg
        K = len(joint_positions)
        
        np.random.seed(42)
        s_0 = np.random.randn(N, 3).astype(np.float32) * 0.1
        
        axis_coords = np.linspace(min_bound[primary_axis], max_bound[primary_axis], num_rings)
        thetas = np.linspace(0, 2 * np.pi, radial_seg, endpoint=False)
        
        s_1 = np.zeros((N, 3), dtype=np.float32)
        idx = 0
        for val_ax in axis_coords:
            for th in thetas:
                pt = np.zeros(3, dtype=np.float32)
                pt[primary_axis] = val_ax
                pt[ortho_axes[0]] = center[ortho_axes[0]] + r_ortho * np.cos(th)
                pt[ortho_axes[1]] = center[ortho_axes[1]] + r_ortho * np.sin(th)
                s_1[idx] = pt
                idx += 1
                
        quads = []
        for i in range(num_rings - 1):
            for j in range(radial_seg):
                j_next = (j + 1) % radial_seg
                v0 = i * radial_seg + j
                v1 = i * radial_seg + j_next
                v2 = (i + 1) * radial_seg + j_next
                v3 = (i + 1) * radial_seg + j
                quads.append([v0, v1, v2, v3])
        quads = np.array(quads, dtype=np.int32)
        
        dt = 1.0 / float(ode_steps)
        s_t = s_0.copy()
        
        for step in range(ode_steps):
            t = step * dt
            v = s_1 - s_0
            strain_pull = np.zeros_like(s_t)
            strain_pull[:, primary_axis] = 0.05 * np.sin(math.pi * t) * (s_1[:, primary_axis] - s_t[:, primary_axis])
            v = v + lambda_strain * strain_pull
            
            if solver_type == "Euler":
                s_t = s_t + dt * v
            elif solver_type == "Midpoint":
                s_mid = s_t + 0.5 * dt * v
                v_mid = s_1 - s_0 + lambda_strain * 0.05 * np.sin(math.pi * (t + 0.5 * dt)) * (s_1 - s_mid)
                s_t = s_t + dt * v_mid
            elif solver_type == "Heun":
                s_pred = s_t + dt * v
                v_next = s_1 - s_0 + lambda_strain * 0.05 * np.sin(math.pi * (t + dt)) * (s_1 - s_pred)
                s_t = s_t + 0.5 * dt * (v + v_next)
            else:
                s_t = s_t + dt * v
                
        final_verts = s_t
        
        dists = np.linalg.norm(final_verts[:, None, :] - joint_positions[None, :, :], axis=-1)
        inv_d = np.exp(-dists / max(0.1, r_ortho * 1.5))
        weights = inv_d / np.sum(inv_d, axis=-1, keepdims=True)
        
        return final_verts, quads, weights

    @staticmethod
    def evaluate_topology_metrics(verts: np.ndarray, quads: np.ndarray) -> Dict[str, float]:
        num_faces = len(quads)
        num_quads = sum(1 for q in quads if len(q) == 4)
        quad_ratio = (num_quads / max(1, num_faces)) * 100.0
        
        degree = np.zeros(len(verts), dtype=np.int32)
        for q in quads:
            for v in q:
                degree[v] += 1
                
        internal_verts = degree >= 3
        regular_v4_count = np.sum(degree[internal_verts] == 4)
        valence_4_pct = (regular_v4_count / max(1, np.sum(internal_verts))) * 100.0
        
        return {
            "quad_ratio": float(quad_ratio),
            "valence_4_pct": float(valence_4_pct),
            "vertex_count": len(verts),
            "face_count": num_faces
        }


# ==============================================================================
# 2. Blender API Declarations (When RUNNING_IN_BLENDER is True)
# ==============================================================================

if RUNNING_IN_BLENDER:
    
    class FlowRetopoProperties(PropertyGroup):
        target_rings: IntProperty(
            name="Ring Loops",
            description="Number of concentric rings along primary articulation axis",
            default=20,
            min=4,
            max=128
        )
        radial_segments: IntProperty(
            name="Radial Segments",
            description="Circumferential quad segments around joint cylinder",
            default=16,
            min=6,
            max=64
        )
        solver_type: EnumProperty(
            name="ODE Solver",
            description="Numerical integration method for probability flow trajectory",
            items=[
                ("Euler", "Euler (1st-Order)", "Fast single-evaluation linear stepping"),
                ("Midpoint", "Midpoint (2nd-Order)", "Balanced Runge-Kutta midpoint integrator"),
                ("Heun", "Heun (Predictor-Corrector)", "High-fidelity 2nd-order predictor corrector"),
            ],
            default="Midpoint"
        )
        ode_steps: IntProperty(
            name="Integration Steps",
            description="Number of discrete integration intervals between t=0 and t=1",
            default=10,
            min=5,
            max=50
        )
        lambda_strain: FloatProperty(
            name="Strain Weight (λ)",
            description="4-RoSy kinematic strain alignment regularizer weight",
            default=0.25,
            min=0.0,
            max=1.0
        )
        bind_skinning: BoolProperty(
            name="Bind Vertex Groups",
            description="Auto-generate bone skinning vertex groups with LBS weights",
            default=True
        )
        aim_axis: EnumProperty(
            name="Aim",
            items=[
                ("AUTO", "自动，最长的一根", "Follow the longest piece"),
                ("X", "向右 +X", "Cut the limb pointing +X"),
                ("NX", "向左 -X", "Cut the limb pointing -X"),
                ("Y", "向前 +Y", "Cut the limb pointing +Y"),
                ("NY", "向后 -Y", "Cut the limb pointing -Y"),
                ("Z", "向上 +Z", "Cut the limb pointing +Z"),
                ("NZ", "向下 -Z", "Cut the limb pointing -Z"),
            ],
            default="AUTO",
        )
        last_quad_ratio: FloatProperty(name="Quad Ratio", default=0.0)
        last_v4_pct: FloatProperty(name="Valence-4 %", default=0.0)
        last_time_ms: FloatProperty(name="Inference Time (ms)", default=0.0)
        last_vert_count: IntProperty(name="Vertices", default=0)
        last_face_count: IntProperty(name="Quads", default=0)


    class FLOWRETOPO_OT_generate(Operator):
        """沿模型最长的方向套一圈圈四边面"""
        bl_idname = "flowretopo.generate"
        bl_label = "套上环线"
        bl_options = {'REGISTER', 'UNDO'}
        
        def execute(self, context):
            props = context.scene.flow_retopo_props
            active_obj = context.active_object
            
            if not active_obj or active_obj.type != 'MESH':
                self.report({'ERROR'}, "先选中一条高模")
                return {'CANCELLED'}

            from pathlib import Path
            addon_dir = str(Path(__file__).resolve().parent)
            if addon_dir not in sys.path:
                sys.path.insert(0, addon_dir)
            from tube import missed_directions, retopo_tube

            mesh = active_obj.data
            mesh.calc_loop_triangles()
            matrix = active_obj.matrix_world
            src_verts = np.array(
                [[(matrix @ vertex.co).x, (matrix @ vertex.co).y, (matrix @ vertex.co).z] for vertex in mesh.vertices],
                dtype=np.float64,
            )
            src_faces = np.array([tuple(tri.vertices) for tri in mesh.loop_triangles], dtype=np.int32)
            if len(src_verts) < 10 or len(src_faces) < 10:
                self.report({'ERROR'}, "模型太小，切不出环")
                return {'CANCELLED'}

            armature_obj = None
            for obj in context.selected_objects:
                if obj.type == 'ARMATURE':
                    armature_obj = obj
                    break
                        
            t0 = time.perf_counter()
            aim = {
                "X": (1.0, 0.0, 0.0),
                "NX": (-1.0, 0.0, 0.0),
                "Y": (0.0, 1.0, 0.0),
                "NY": (0.0, -1.0, 0.0),
                "Z": (0.0, 0.0, 1.0),
                "NZ": (0.0, 0.0, -1.0),
            }.get(props.aim_axis)
            try:
                verts, quads, direction = retopo_tube(
                    src_verts,
                    src_faces,
                    n_rings=props.target_rings,
                    n_around=props.radial_segments,
                    axis=None if aim is None else np.array(aim, dtype=np.float64),
                )
            except RuntimeError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}

            center = verts.mean(axis=0)
            along = (verts - center) @ direction
            span = max(float(np.ptp(along)), 1e-6)
            w_fore = 1.0 / (1.0 + np.exp(-np.clip((-along) / (0.12 * span), -20.0, 20.0)))
            weights = np.column_stack([1.0 - w_fore, w_fore])
            joint_names = ["Upper", "Lower"]
            if armature_obj is not None:
                bone_names = [bone.name for bone in armature_obj.data.bones]
                if len(bone_names) >= 2:
                    joint_names = bone_names[:2]
            
            mesh_name = f"{active_obj.name}_Retopo"
            new_mesh = bpy.data.meshes.new(mesh_name)
            new_mesh.from_pydata(verts.tolist(), [], quads.tolist())
            new_mesh.update()
            
            new_obj = bpy.data.objects.new(mesh_name, new_mesh)
            context.collection.objects.link(new_obj)
            
            if props.bind_skinning and armature_obj:
                modifier = new_obj.modifiers.new(name="Armature", type='ARMATURE')
                modifier.object = armature_obj
                
                for k, b_name in enumerate(joint_names):
                    vg = new_obj.vertex_groups.new(name=b_name)
                    for v_idx in range(len(verts)):
                        w = float(weights[v_idx, k])
                        if w > 0.001:
                            vg.add([v_idx], w, 'REPLACE')
                            
            metrics = RetopoInferenceEngine.evaluate_topology_metrics(verts, quads)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            
            props.last_quad_ratio = metrics["quad_ratio"]
            props.last_v4_pct = metrics["valence_4_pct"]
            props.last_vert_count = metrics["vertex_count"]
            props.last_face_count = metrics["face_count"]
            props.last_time_ms = dt_ms
            
            bpy.ops.object.select_all(action='DESELECT')
            new_obj.select_set(True)
            context.view_layer.objects.active = new_obj
            
            note = ""
            if aim is None:
                missed = missed_directions(src_verts, verts)
                names = ("左右", "前后", "上下")
                if missed:
                    note = "。" + "、".join(names[axis] for axis in missed) + "还没切完"
            self.report({'INFO'}, f"套了 {metrics['face_count']} 个四边面，用时 {dt_ms:.0f} 毫秒{note}")
            return {'FINISHED'}


    class FLOWRETOPO_PT_main_panel(Panel):
        """Sidebar panel for putting quad rings on a long mesh."""
        bl_label = "环线重拓扑"
        bl_idname = "FLOWRETOPO_PT_main_panel"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = '环线'
        
        def draw(self, context):
            layout = self.layout
            props = context.scene.flow_retopo_props
            
            box_mesh = layout.box()
            box_mesh.label(text="选中一条细长的高模", icon='MESH_DATA')
            active = context.active_object
            if active:
                box_mesh.label(text=f"当前: {active.name}")
            else:
                box_mesh.label(text="还没选中模型", icon='ERROR')
                
            box_res = layout.box()
            box_res.label(text="沿最长的方向切环", icon='MOD_REMESH')
            box_res.prop(props, "aim_axis", text="切哪根")
            box_res.prop(props, "target_rings", text="圈数")
            box_res.prop(props, "radial_segments", text="每圈几点")
            box_res.prop(props, "bind_skinning", text="顺便写两根骨头的权重")
            
            layout.separator()
            row = layout.row(align=True)
            row.scale_y = 1.6
            row.operator("flowretopo.generate", icon='PLAY')
            
            if props.last_face_count > 0:
                box_metrics = layout.box()
                box_metrics.label(text=f"顶点 {props.last_vert_count}，四边面 {props.last_face_count}")
                box_metrics.label(text=f"用时 {props.last_time_ms:.0f} 毫秒")


    classes = [
        FlowRetopoProperties,
        FLOWRETOPO_OT_generate,
        FLOWRETOPO_PT_main_panel,
    ]

    def register():
        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.flow_retopo_props = PointerProperty(type=FlowRetopoProperties)

    def unregister():
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
        del bpy.types.Scene.flow_retopo_props


def run_standalone_addon_test():
    """Check the cutter the sidebar button calls, without opening Blender."""
    from pathlib import Path

    addon_dir = str(Path(__file__).resolve().parent)
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)
    from tube import retopo_tube

    assert "环线" in bl_info["name"]
    assert bl_info["blender"] >= (4, 0, 0)
    around = 12
    heights = np.linspace(-1.0, 1.0, 20)
    angles = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    vertices = np.array(
        [[0.2 * np.cos(angle), 0.2 * np.sin(angle), height] for height in heights for angle in angles],
        dtype=np.float64,
    )
    faces = []
    for j in range(len(heights) - 1):
        for i in range(around):
            nxt = (i + 1) % around
            v0 = j * around + i
            v1 = j * around + nxt
            v2 = (j + 1) * around + nxt
            v3 = (j + 1) * around + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    quads_v, quads_f, _axis = retopo_tube(vertices, np.asarray(faces, dtype=np.int32), n_rings=12, n_around=8)
    span = quads_v.max(axis=0) - quads_v.min(axis=0)
    if span[2] <= span[0] * 2 or len(quads_f) < 24 or quads_f.shape[1] != 4:
        raise SystemExit(f"插件自测没有切出管子: span={span} faces={len(quads_f)}")
    print(f"套了 {len(quads_f)} 个四边面")


if __name__ == "__main__":
    if RUNNING_IN_BLENDER:
        register()
    else:
        run_standalone_addon_test()