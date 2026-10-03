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
import time
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
    )
    from bpy.types import Panel, Operator, PropertyGroup
    RUNNING_IN_BLENDER = True
except ImportError:
    RUNNING_IN_BLENDER = False


# ==============================================================================
# 2. Blender API Declarations (When RUNNING_IN_BLENDER is True)
# ==============================================================================

if RUNNING_IN_BLENDER:
    
    class FlowRetopoProperties(PropertyGroup):
        target_rings: IntProperty(
            name="圈数",
            description="沿着这根切多少圈",
            default=20,
            min=4,
            max=128
        )
        radial_segments: IntProperty(
            name="每圈几点",
            description="每一圈上放几个点",
            default=16,
            min=6,
            max=64
        )
        bind_skinning: BoolProperty(
            name="写权重",
            description="同时选中骨架时，给前两根骨头写权重",
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
        last_time_ms: FloatProperty(name="用时", default=0.0)
        last_vert_count: IntProperty(name="顶点", default=0)
        last_face_count: IntProperty(name="四边面", default=0)


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
                            
            dt_ms = (time.perf_counter() - t0) * 1000.0
            props.last_vert_count = len(verts)
            props.last_face_count = len(quads)
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
            self.report({'INFO'}, f"套了 {len(quads)} 个四边面，用时 {dt_ms:.0f} 毫秒{note}")
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
            box_res.prop(props, "bind_skinning", text="选中骨架时写权重")
            
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