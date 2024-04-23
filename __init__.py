bl_info = {
    "name": "make_VIsibilityBase_shapekey",
    "author": "Hooha",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Extended Tools > MKVBS",
    "description": "",
    "warning": "",
    "wiki_url": "",
    "category": '3D View'}


import bpy
import bmesh
from bpy.app.handlers import persistent

from mathutils import Vector, Matrix, Euler, Quaternion

import numpy as np
from math import pi
from timeit import default_timer as timer








shapekey_n = 'MKVBS'
threshold = 1e-4
learning_rate = 1e-1







# global data
MKVBS_data = {}
MKVBS_data['inst'] = {}


class loopInfo():
    def __init__(self):
        pass



def create_instance(self, context):
    ob = self.id_data
    
    global glob_dg
    glob_dg = bpy.context.evaluated_depsgraph_get()
    
    if ob.MKVBS.target_obn in [i.name for i in bpy.data.objects if i.type=='MESH' and not i.name == ob.name] and ob.MKVBS.true_target==False:
        print('success_instance')
        ob.MKVBS.true_target = True
        ob.MKVBS.inst_id = timer()
        
        MKVBS_data['inst'][ob.MKVBS.inst_id] = loopInfo()
        loop_inst = MKVBS_data['inst'][ob.MKVBS.inst_id]
        loop_inst.base_ob = ob
        
        if ob.data.shape_keys == None:
            ob.shape_key_add(name='Basis', from_mix=True)
        if ob.MKVBS.shapekey_n in list(ob.data.shape_keys.key_blocks.keys()):
            loop_inst.shapekey = ob.data.shape_keys.key_blocks[ob.MKVBS.shapekey_n]
        else:
            loop_inst.shapekey = ob.shape_key_add(name=ob.MKVBS.shapekey_n, from_mix=True)
        loop_inst.shapekey_n = loop_inst.shapekey.name
        loop_inst.shapekey.value = 1.0
        ob.active_shape_key_index = ob.data.shape_keys.key_blocks.find(loop_inst.shapekey.name)
        
        loop_inst.direct = np.random.randn(len(ob.data.vertices),3)
        loop_inst.direct *= ob.MKVBS.learning_rate
        
        loop_inst.vc = len(ob.data.vertices)
        
        loop_inst.base_co = np.empty(loop_inst.vc*3, dtype=np.float32)
        ob.data.vertices.foreach_get('co', loop_inst.base_co)
        loop_inst.base_co = np.reshape(loop_inst.base_co, (loop_inst.vc,3))
        
        loop_inst.target_co = np.empty(loop_inst.vc*3, dtype=np.float32)
        glob_dg.objects[ob.MKVBS.target_obn].data.vertices.foreach_get('co', loop_inst.target_co)
        loop_inst.target_co = np.reshape(loop_inst.target_co,(loop_inst.vc,3))
        
    elif ob.MKVBS.target_obn in [i.name for i in bpy.data.objects if i.type=='MESH' and not i.name == ob.name] and ob.MKVBS.true_target==True:
        print('already instance')
        pass
    else:
        ob.MKVBS.true_target = False
        ob.MKVBS.inst_id = 0.0
        ob.shape_key_clear()
        try:
            del MKVBS_data['inst'][ob.MKVBS.inst_id]
        except:
            print("you didn't specify target object even once time")
    return





def oneStep_PDiff(self, context):
    if bpy.context.scene.MKVBS.one_step:
        bpy.context.scene.MKVBS.one_step = False
        ob = bpy.context.active_object
        
        if not ob.MKVBS.inst_id in list(MKVBS_data['inst'].keys()):
            bpy.context.active_object.MKVBS.true_target = False
            create_instance(bpy.context.active_object, None)
            inst = MKVBS_data['inst'][ob.MKVBS.inst_id]
        else:
            inst = MKVBS_data['inst'][ob.MKVBS.inst_id]
        
        
        before_vis_co = np.empty(inst.vc*3, dtype=np.float32)
        glob_dg.objects[ob.name].data.vertices.foreach_get('co', before_vis_co)
        before_vis_co = np.reshape(before_vis_co,(inst.vc, 3))
        
        before_sk_co = np.empty(inst.vc*3, dtype=np.float32)
        ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_get('co', before_sk_co.ravel())
        before_sk_co = np.reshape(before_sk_co, (inst.vc, 3))
        
        ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_set('co', (inst.base_co + inst.direct).flatten())
        glob_dg.update()
        ob.data.update()
        
        after_vis_co = np.empty(inst.vc*3, dtype=np.float32)
        glob_dg.objects[ob.name].data.vertices.foreach_get('co', after_vis_co)
        after_vis_co = np.reshape(after_vis_co,(inst.vc, 3))
        
        
        before_loss_v = inst.target_co - before_vis_co
        after_loss_v = inst.target_co - after_vis_co
        
        before_loss = np.sqrt(np.einsum('ij,ij->i', before_loss_v, before_loss_v))
        after_loss = np.sqrt(np.einsum('ij,ij->i', after_loss_v, after_loss_v))
        
        
        after_sk_co = np.empty(inst.vc*3, dtype=np.float32)
        ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_get('co', after_sk_co)
        after_sk_co = np.reshape(after_sk_co,(inst.vc, 3))
        
        diff_loss = (before_loss > after_loss)
        
        before_co = (1-diff_loss)[:,None] * before_sk_co
        current_co = diff_loss[:,None] * after_sk_co.copy()
        update_apply = before_co + current_co
        
        
        ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_set('co', update_apply.flatten())
        glob_dg.update()
        ob.data.update()
        
        stay_lr = (inst.direct * ob.MKVBS.learning_rate) <= 1e-6
        leave_lr = (1-stay_lr) * ob.MKVBS.learning_rate
        
        current_direct_v = diff_loss[:,None] * inst.direct
        new_direct_v = ((1-diff_loss)[:,None] * np.random.randn(inst.vc,3)) * (stay_lr + leave_lr)
        
        inst.direct = current_direct_v + new_direct_v
        
        print('')
    
    
    return


@persistent
def update_realtime(scene=None):
    if bpy.context.scene.MKVBS.loop_switch:
        # print('auto_loop')
        # inst = MKVBS_data['inst'][bpy.context.active_object.MKVBS.inst_id]
        bpy.context.scene.MKVBS.one_step = True
        oneStep_PDiff(bpy.context.active_object, context=None)
    return bpy.context.scene.MKVBS.delay




class MKVBS_PropsObject(bpy.types.PropertyGroup):
    inst_id:\
    bpy.props.FloatProperty(name="Instance ID", description="Unique Instance ID")
    
    target_obn:\
    bpy.props.StringProperty(name="Visibility Target Object Name", description="Set Target object name", update=create_instance)
    
    shapekey_n:\
    bpy.props.StringProperty(name="Receive Shapekey", description="Set Target object", default=shapekey_n)
    
    threshold:\
    bpy.props.FloatProperty(name="threshold", description="Threshold about difference", default=threshold)
    
    learning_rate:\
    bpy.props.FloatProperty(name="learning_rate", description="Update step", default=learning_rate, min=0.0)
    
    true_target:\
    bpy.props.BoolProperty(name="True Target", description="", default=False)




class MKVBS_PropsScene(bpy.types.PropertyGroup):
    loop_switch:\
    bpy.props.BoolProperty(name="loop_switch", description="realtime loop on / off", default=False)
    
    one_step:\
    bpy.props.BoolProperty(name="One Step", description="Just One loop on / off", default=False, update=oneStep_PDiff)
    
    delay:\
    bpy.props.FloatProperty(name="Auto loop delay", description="Auto loop delay", default=0.0)
    




class PANEL_PT_MKVBS(bpy.types.Panel):
    bl_label = "MKVBS"
    bl_idname = "PANEL_PT_MKVBS"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MKVBS'
    
    @classmethod
    def poll(cls, context):
        ob = bpy.context.object
        if ob is None or ob.type != 'MESH':
            return False
        return True

    def __init__(self):
        ob = bpy.context.object
        self.ob = ob

    def draw(self, context):
        sc = bpy.context.scene
        ob = self.ob
        layout = self.layout
        
        if ob.MKVBS.true_target:
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(sc.MKVBS, "loop_switch", text="Auto Loop On / Off", icon='CON_PIVOT')
            
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(sc.MKVBS, "one_step", text="Just One Step", icon='CON_PIVOT')
            
            col = layout.column(align=True)
            col.scale_y = 1.0
            col.prop(sc.MKVBS, "delay", text="Loop delay")
            
            layout.separator()
        
        
        
        layout.label(text='Instance ID')
        col = layout.column(align=True)
        col.scale_y = 1.0
        col.prop(ob.MKVBS, "inst_id", text='')
        
        layout.label(text='Target Object')
        col = layout.column(align=True)
        col.scale_y = 1.0
        col.prop(ob.MKVBS, "target_obn", text='')
        
        layout.label(text='Shapekey Name')
        col = layout.column(align=True)
        col.scale_y = 1.0
        col.prop(ob.MKVBS, "shapekey_n", text='')
        
        if ob.MKVBS.true_target:
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(ob.MKVBS, "threshold", text="Difference Threshold", icon='FULLSCREEN_EXIT')
            
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(ob.MKVBS, "learning_rate", text="Leanring Rate", icon='FULLSCREEN_EXIT')
            
            layout.separator()
        




classes = (
    MKVBS_PropsObject,
    MKVBS_PropsScene,
    PANEL_PT_MKVBS,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)
    
    bpy.types.Object.MKVBS = bpy.props.PointerProperty(type=MKVBS_PropsObject)
    bpy.types.Scene.MKVBS = bpy.props.PointerProperty(type=MKVBS_PropsScene)
        
    bpy.app.timers.register(update_realtime, persistent=True)

def unregister():
    pass

if __name__ == "__main__":
    register()