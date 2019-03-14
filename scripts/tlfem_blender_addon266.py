# Copyright 2012 Dorival de Moraes Pedroso. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

bl_info = {
    'name'        : 'FEMmesh mesh generator',
    'author'      : 'Dorival Pedroso',
    'version'     : (1, 0),
    'blender'     : (2, 6, 3),
    'location'    : 'View3D > Tools panel > FEMmesh mesh generator',
    'warning'     : '',
    'description' : 'Generate V and C lists for FEMmesh',
    'wiki_url'    : '',
    'tracker_url' : '',
    'category'    : '3D View'}

import bgl
import blf
import bpy
from   bpy_extras import view3d_utils
import subprocess

def draw_callback_px(self, context):
    wm = context.window_manager
    sc = context.scene
    if not wm.do_show_tags: return
    ob = context.object
    if not ob: return
    if not ob.type == 'MESH': return

    # status
    font_id = 0
    blf.position(font_id, 45, 45, 0)
    #blf.size(font_id, 20, 48) # 20, 72)
    blf.size(font_id, 15, 72)
    blf.draw(font_id, "displaying tags")

    # region
    reg = bpy.context.region
    r3d = bpy.context.space_data.region_3d

    # transformation matrix (local co => global co)
    T = ob.matrix_world.copy()

    # vert tags
    if len(ob.vtags)>0 and sc.pyfem_show_vtag:
        blf.size(font_id, sc.pyfem_vert_font, 72)
        r, g, b = sc.pyfem_vert_color
        bgl.glColor4f(r, g, b, 1.0)
        for v in ob.vtags.values():
            if v.tag >= 0: continue
            pm = ob.data.vertices[v.idx].co
            co = view3d_utils.location_3d_to_region_2d(reg, r3d, T*pm)
            blf.position(font_id, co[0], co[1], 0)
            blf.draw(font_id, "%d"%v.tag)

    # edge tags
    if len(ob.etags)>0 and sc.pyfem_show_etag:
        blf.size(font_id, sc.pyfem_edge_font, 72)
        r, g, b = sc.pyfem_edge_color
        bgl.glColor4f(r, g, b, 1.0)
        for v in ob.etags.values():
            if v.tag >= 0: continue
            pa = ob.data.vertices[v.v0].co
            pb = ob.data.vertices[v.v1].co
            pm = (pa+pb)/2.0
            co = view3d_utils.location_3d_to_region_2d(reg, r3d, T*pm)
            blf.position(font_id, co[0], co[1], 0)
            blf.draw(font_id, "%d"%v.tag)

    # cell tags
    if len(ob.ctags)>0 and sc.pyfem_show_ctag:
        blf.size(font_id, sc.pyfem_cell_font, 72)
        r, g, b = sc.pyfem_cell_color
        bgl.glColor4f(r, g, b, 1.0)
        for v in ob.ctags.values():
            if v.tag >= 0: continue
            c  = ob.data.polygons[v.idx]
            pm = ob.data.vertices[c.vertices[0]].co.copy()
            for k in range(1, len(c.vertices)):
                pm += ob.data.vertices[c.vertices[k]].co
            pm /= float(len(c.vertices))
            co = view3d_utils.location_3d_to_region_2d(reg, r3d, T*pm)
            blf.position(font_id, co[0], co[1], 0)
            blf.draw(font_id, "%d"%v.tag)


class FEMmeshDisplayTags(bpy.types.Operator):
    bl_idname      = "view3d.show_tags"
    bl_label       = "Show Tags"
    bl_description = "Display tags on top of mesh"
    last_activity  = 'NONE'
    _handle        = None
    _timer         = None

    @staticmethod
    def handle_add(self, context):
        FEMmeshDisplayTags._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
        FEMmeshDisplayTags._timer  = context.window_manager.event_timer_add(0.075, context.window)

    @staticmethod
    def handle_remove(context):
        if FEMmeshDisplayTags._handle is not None:
            context.window_manager.event_timer_remove(FEMmeshDisplayTags._timer)
            bpy.types.SpaceView3D.draw_handler_remove(FEMmeshDisplayTags._handle, 'WINDOW')
        FEMmeshDisplayTags._handle = None
        FEMmeshDisplayTags._timer  = None

    def modal(self, context, event):
        # redraw
        #print(context)
        if context.area:
            context.area.tag_redraw()
        # stop script
        if not context.window_manager.do_show_tags:
            FEMmeshDisplayTags.handle_remove(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def cancel(self, context):
        if context.window_manager.do_show_tags:
            FEMmeshDisplayTags.handle_remove(context)
            context.window_manager.do_show_tags = False
        return {'CANCELLED'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            # operator is called for the first time, start everything
            if not context.window_manager.do_show_tags:
                context.window_manager.do_show_tags = True
                FEMmeshDisplayTags.handle_add(self, context)
                return {'RUNNING_MODAL'}
            # operator is called again, stop displaying
            else:
                context.window_manager.do_show_tags = False
                return {'CANCELLED'}
        else:
            self.report({'WARNING'}, "View3D not found, can't run operator")
            return {'CANCELLED'}

    def unregister():
        FEMmeshDisplayTags.handle_remove(bpy.context)

class SetVertexTag(bpy.types.Operator):
    bl_idname      = "pyfem.set_vert_tag"
    bl_label       = "Set vertex tag"
    bl_description = "Set vertex tag (for selected vertices)"

    @classmethod
    def poll(cls, context):
        return context.object and (context.object.type == 'MESH') and ('EDIT' in context.object.mode)

    def execute(self, context):
        bpy.ops.object.editmode_toggle()
        sc   = context.scene
        ob   = context.object
        vids = [v.idx for v in ob.vtags.values()]
        for v in ob.data.vertices:
            if v.select == True:    # vertex is selected
                if v.index in vids: # update
                    ob.vtags[vids.index(v.index)].tag = sc.pyfem_default_vert_tag
                else:
                    new     = ob.vtags.add()
                    new.tag = sc.pyfem_default_vert_tag
                    new.idx = v.index
        bpy.ops.object.editmode_toggle()
        return {'FINISHED'}


class SetEdgeTag(bpy.types.Operator):
    bl_idname      = "pyfem.set_edge_tag"
    bl_label       = "Set edge tag"
    bl_description = "Set edge tag (for selected edges)"

    @classmethod
    def poll(cls, context):
        return context.object and (context.object.type == 'MESH') and ('EDIT' in context.object.mode)

    def execute(self, context):
        bpy.ops.object.editmode_toggle()
        sc    = context.scene
        ob    = context.object
        ekeys = [(v.v0,v.v1) for v in ob.etags.values()]
        for e in ob.data.edges:
            if e.select == True:   # edge is selected
                if e.key in ekeys: # update
                    ob.etags[ekeys.index(e.key)].tag = sc.pyfem_default_edge_tag
                else:
                    new     = ob.etags.add()
                    new.tag = sc.pyfem_default_edge_tag
                    new.v0  = e.key[0]
                    new.v1  = e.key[1]
        bpy.ops.object.editmode_toggle()
        return {'FINISHED'}


class SetCellTag(bpy.types.Operator):
    bl_idname      = "pyfem.set_cell_tag"
    bl_label       = "Set cell tag"
    bl_description = "Set cell tag (for selected faces)"

    @classmethod
    def poll(cls, context):
        return context.object and (context.object.type == 'MESH') and ('EDIT' in context.object.mode)

    def execute(self, context):
        bpy.ops.object.editmode_toggle()
        sc   = context.scene
        ob   = context.object
        cids = [v.idx for v in ob.ctags.values()]
        for p in ob.data.polygons:
            if p.select == True:    # polygon is selected
                if p.index in cids: # update
                    ob.ctags[cids.index(p.index)].tag = sc.pyfem_default_cell_tag
                else:
                    new     = ob.ctags.add()
                    new.tag = sc.pyfem_default_cell_tag
                    new.idx = p.index
        bpy.ops.object.editmode_toggle()
        return {'FINISHED'}


def write_mesh_to_file(filepath, context, drawmesh=False, ids=False, tags=True, tol=0.0001, flatten=False):
    sc     = context.scene
    ob     = context.object
    me     = ob.data
    T      = ob.matrix_world.copy()
    vids   = [v.idx       for v in ob.vtags.values()]
    ekeys  = [(v.v0,v.v1) for v in ob.etags.values()]
    cids   = [v.idx       for v in ob.ctags.values()]
    errors = ''
    # header
    l = ''
    if drawmesh:
        l += 'from msys_drawmesh import DrawMesh\n'
    # vertices
    l += 'V=[\n'
    for k, v in enumerate(me.vertices):
        if flatten and abs(v.co[2]) > 0.0:
            v.co[2] = 0.0
        co = T * v.co
        tg = ob.vtags[vids.index(v.index)].tag if (v.index in vids) else 0
        l += '  [%d, %d, %.8f, %.8f]' % (k, tg, co[0], co[1])
        if k<len(me.vertices)-1: l += ','
        l += '\n'
    # cells
    nc = len(ob.data.polygons) # number of cells
    l += ']\nC=[\n'
    for i, p in enumerate(ob.data.polygons):
        tg  = ob.ctags[cids.index(p.index)].tag if (p.index in cids) else 0
        n   = p.normal
        err = ''
        if abs(n[0])>tol or abs(n[1])>tol:
            err += 'Face has normal non-parallel to z'
        if n[2]<tol:
            err += 'Face has wrong normal; vertices must be counter-clockwise'
        l += '  [%d, %d, [' % (i, tg)
        et = {}              # edge tags
        nv = len(p.vertices) # number of vertices
        for k in range(nv):
            v0, v1 = ob.data.vertices[p.vertices[k]].index, ob.data.vertices[p.vertices[(k+1)%nv]].index
            l += '%d' % v0
            if k<nv-1: l += ','
            else:      l += ']'
            ek = (v0,v1) if v0<v1 else (v1,v0) # edge key
            if ek in ekeys:
                if ob.etags[ekeys.index(ek)].tag >=0: continue
                et[k] = ob.etags[ekeys.index(ek)].tag
        if len(et)>0: l += ', {'
        k = 0
        for idx, tag in et.items():
            l += '%d:%d' % (idx, tag)
            if k<len(et)-1: l += ', '
            else:           l += '}'
            k += 1
        if i<nc-1: l += '],'
        else:      l += ']'
        if err!='':
            l += '# ' + err
            errors = err
        l += '\n'
    l += ']\n'
    # footer
    if drawmesh:
        l += 'd = DrawMesh(V, C)\n'
        l += 'd.draw(with_ids=%s, with_tags=%s)\n' % (str(ids), str(tags))
        l += 'd.show()\n'
    # write to file
    f = open(filepath, 'w')
    f.write(l)
    f.close()
    return errors


class FEMmeshExporter(bpy.types.Operator):
    bl_idname      = "pyfem.export_mesh"
    bl_label       = "Export V and C lists"
    bl_description = "Save file with V and C lists"

    filepath = bpy.props.StringProperty(subtype='FILE_PATH',)
    check_existing = bpy.props.BoolProperty(
            name        = "Check Existing",
            description = "Check and warn on overwriting existing files",
            default     = True,
            options     = {'HIDDEN'},)

    @classmethod
    def poll(cls, context):
        return context.object and (context.object.type == 'MESH')

    def execute(self, context):
        bpy.ops.object.editmode_toggle()
        errors = write_mesh_to_file(self.filepath, context,
                       flatten=context.scene.pyfem_flatten)
        if errors!='': self.report({'WARNING'}, errors)
        bpy.ops.object.editmode_toggle()
        return {'FINISHED'}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.ensure_ext(bpy.data.filepath, ".py")
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class FEMmeshMsysDrawmesh(bpy.types.Operator):
    bl_idname      = "pyfem.msys_drawmesh"
    bl_label       = "Draw 2D mesh using Matplotlib"
    bl_description = "View current mesh with Pylab"

    @classmethod
    def poll(cls, context):
        return context.object and (context.object.type == 'MESH')

    def execute(self, context):
        bpy.ops.object.editmode_toggle()
        fn     = bpy.app.tempdir + 'temporary.fem_blender_addon.py'
        errors = write_mesh_to_file(fn, context, True,
                   context.scene.pyfem_msys_with_ids,
                   context.scene.pyfem_msys_with_tags,
                   flatten=context.scene.pyfem_flatten)
        if errors=='':
            try:    subprocess.Popen(['python', fn])
            except: self.report({'WARNING'}, 'calling external Python command failed')
        else: self.report({'WARNING'}, errors)
        bpy.ops.object.editmode_toggle()
        return {'FINISHED'}


class ObjectVertTag(bpy.types.PropertyGroup):
    tag = bpy.props.IntProperty()
    idx = bpy.props.IntProperty()


class ObjectEdgeTag(bpy.types.PropertyGroup):
    tag = bpy.props.IntProperty()
    v0  = bpy.props.IntProperty()
    v1  = bpy.props.IntProperty()


class ObjectCellTag(bpy.types.PropertyGroup):
    tag = bpy.props.IntProperty()
    idx = bpy.props.IntProperty()


def init_properties():
    # object data
    bpy.utils.register_class(ObjectVertTag)
    bpy.utils.register_class(ObjectEdgeTag)
    bpy.utils.register_class(ObjectCellTag)
    bpy.types.Object.vtags = bpy.props.CollectionProperty(type=ObjectVertTag)
    bpy.types.Object.etags = bpy.props.CollectionProperty(type=ObjectEdgeTag)
    bpy.types.Object.ctags = bpy.props.CollectionProperty(type=ObjectCellTag)

    # scene data
    scene = bpy.types.Scene
    scene.pyfem_default_edge_tag = bpy.props.IntProperty(
        name        ="E",
        description = "Default Edge Tag",
        default     = -10,
        min         = -99,
        max         = 0)

    scene.pyfem_default_vert_tag = bpy.props.IntProperty(
        name        ="V",
        description = "Default Vertex Tag",
        default     = -100,
        min         = -1000,
        max         = 0)

    scene.pyfem_default_cell_tag = bpy.props.IntProperty(
        name        ="C",
        description = "Default Cell Tag",
        default     = -1,
        min         = -99,
        max         = 0)

    # show tags
    scene.pyfem_show_etag = bpy.props.BoolProperty(
        name        = "Edge",
        description = "Display Edge Tags",
        default     = True)

    scene.pyfem_show_vtag = bpy.props.BoolProperty(
        name        = "Vertex",
        description = "Display Vertex Tags",
        default     = True)

    scene.pyfem_show_ctag = bpy.props.BoolProperty(
        name        = "Cell",
        description = "Display Cell Tags",
        default     = True)

    # font sizes
    scene.pyfem_vert_font = bpy.props.IntProperty(
        name        = "V",
        description = "Vertex font size",
        default     = 12,
        min         = 6,
        max         = 100)

    scene.pyfem_edge_font = bpy.props.IntProperty(
        name        = "E",
        description = "Edge font size",
        default     = 12,
        min         = 6,
        max         = 100)

    scene.pyfem_cell_font = bpy.props.IntProperty(
        name        = "C",
        description = "Edge font size",
        default     = 20,
        min         = 6,
        max         = 100)

    # font colors
    scene.pyfem_vert_color = bpy.props.FloatVectorProperty(
        name        = "V",
        description = "Vertex color",
        default     = (1.0, 0.805, 0.587),
        min         = 0,
        max         = 1,
        subtype     = 'COLOR')

    scene.pyfem_edge_color = bpy.props.FloatVectorProperty(
        name        = "E",
        description = "Edge color",
        default     = (0.934, 0.764, 1.0),
        min         = 0,
        max         = 1,
        subtype     = 'COLOR')

    scene.pyfem_cell_color = bpy.props.FloatVectorProperty(
        name        = "C",
        description = "Cell color",
        default     = (0.504, 0.786, 1.0),
        min         = 0,
        max         = 1,
        subtype     = 'COLOR')

    # export data
    scene.pyfem_flatten = bpy.props.BoolProperty(
        name        = "Project z back to 0",
        description = "Project z coordinates back to 0 (flatten)",
        default     = False)

    # view in pylab
    scene.pyfem_msys_with_ids = bpy.props.BoolProperty(
        name        = "W. IDs",
        description = "Display IDs in Matplotlib",
        default     = False)

    scene.pyfem_msys_with_tags = bpy.props.BoolProperty(
        name        = "W. Tags",
        description = "Display Tags in Matplotlib",
        default     = True)

    # do_show_tags is initially always False and it is in the window manager, not the scene
    wm = bpy.types.WindowManager
    wm.do_show_tags = bpy.props.BoolProperty(default=False)


class FEMmeshPanel(bpy.types.Panel):
    bl_label       = "CIVL4250 FEMmesh"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "TOOL_PROPS"

    def draw(self, context):
        sc = context.scene
        wm = context.window_manager
        l  = self.layout

        l.label("Set tags:")
        c = l.column(align=True)
        r = c.row(align=True); r.prop(sc, "pyfem_default_vert_tag"); r.operator("pyfem.set_vert_tag")
        r = c.row(align=True); r.prop(sc, "pyfem_default_edge_tag"); r.operator("pyfem.set_edge_tag")
        r = c.row(align=True); r.prop(sc, "pyfem_default_cell_tag"); r.operator("pyfem.set_cell_tag")

        l.label("Show/hide:")
        c = l.column(align=True)
        r = c.row(align=True)
        r.prop(sc, "pyfem_show_vtag")
        r.prop(sc, "pyfem_show_etag")
        r.prop(sc, "pyfem_show_ctag")
        if not wm.do_show_tags:
            l.operator("view3d.show_tags", text="Start display", icon='PLAY')
        else:
            l.operator("view3d.show_tags", text="Stop display", icon='PAUSE')

        l.label("Font size and colors:")
        c = l.column(align=True)
        r = c.row(align=True); r.prop(sc, "pyfem_vert_font"); r.prop(sc, "pyfem_vert_color", text="")
        r = c.row(align=True); r.prop(sc, "pyfem_edge_font"); r.prop(sc, "pyfem_edge_color", text="")
        r = c.row(align=True); r.prop(sc, "pyfem_cell_font"); r.prop(sc, "pyfem_cell_color", text="")

        l.label("Export data:")
        l.prop(sc, "pyfem_flatten")
        l.operator("pyfem.export_mesh",   text="Save .py File")

        l.label("View in Matplotlib:")
        c = l.column(align=True)
        r = c.row(align=True)
        r.prop(sc, "pyfem_msys_with_ids")
        r.prop(sc, "pyfem_msys_with_tags")
        l.operator("pyfem.msys_drawmesh", text="View with Pylab")

def register():
    init_properties()
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
