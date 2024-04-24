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
learning_rate = 5e-1







# global data
MKVBS_data = {}
MKVBS_data['inst'] = {}


class loopInfo():
    def __init__(self):
        pass



def cb_target(self, context):
    ob = self.id_data
    
    global glob_dg
    glob_dg = bpy.context.evaluated_depsgraph_get()
    
    bit_arr = np.array([0,0,0,0], dtype=np.int8)
    
    
    if not ob.MKVBS.target_obn in [i.name for i in bpy.data.objects if i.type == 'MESH']:
        bit_arr[0] = 1
    else:
        print(f'{ob.MKVBS.target_obn} names Mesh Object not exist.')
        try:
            del MKVBS_data['inst'][ob.MKVBS.inst_id]
            ob.MKVBS.inst_id = 0.0
            if len(ob.data.shape_keys.key_blocks) <=2 and ob.data.shape_keys != None:
                ob.shape_key_clear()
            else:
                ob.shape_key_remove(ob.data.shape_keys.key_blocks[ob.MKVBS.shapekey_n])
            print(f'clear instance about {ob.name}. you must check mesh object name.')
        except:
            print('you must check mesh object name')
        return
    if len(glob_dg.objects[ob.MKVBS.target_obn].data.vertices) == len(glob_dg.objects[ob.name].data.vertices):
        bit_arr[1] = 1
    else:
        print(f'base ob vertex count = {len(glob_dg.objects[ob.name].data.vertices)}\n\
            target ob vertex count = {len(glob_dg.objects[ob.MKVBS.target_obn].data.vertices)}')
        print('Visibility mesh vertex counts miss match')
        return
    
    if ob.data.shape_keys != None:
        bit_arr[2] = len(ob.data.shape_keys.key_blocks)
    else:
        bit_arr[2] = 0
    if bit_arr[2]>1:
        for idx, skn in enumerate(ob.data.shape_keys.key_blocks.keys()):
            if skn == ob.MKVBS.shapekey_n:
                bit_arr[3] = idx
                break
            else:
                continue
    else:
        ob.shape_key_add(name='Basis', from_mix=True)
        bit_arr[2] = 0
    
    
    
    ob.MKVBS.inst_id = round(timer(),2)
    MKVBS_data['inst'][ob.MKVBS.inst_id] = loopInfo()
    inst = MKVBS_data['inst'][ob.MKVBS.inst_id]
    inst.base_ob = ob
    inst.base_obn = ob.name
    inst.base_ob = ob
    inst.base_obn = ob.name
    
    
    
    inst.vc = len(ob.data.vertices)
    inst.base_co = np.empty(inst.vc*3, dtype=np.float64)
    ob.data.vertices.foreach_get('co', inst.base_co)
    inst.base_co = np.reshape(inst.base_co, (inst.vc,3))
    
    inst.target_co = np.empty(inst.vc*3, dtype=np.float64)
    glob_dg.objects[ob.MKVBS.target_obn].data.vertices.foreach_get('co', inst.target_co)
    inst.target_co = np.reshape(inst.target_co,(inst.vc,3))
    
    inst.shapekey_n = ob.MKVBS.shapekey_n
    if bit_arr[3] == 0:
        ob.shape_key_add(name=ob.MKVBS.shapekey_n, from_mix=True)
    else:
        pass
    ob.data.shape_keys.key_blocks[inst.shapekey_n].value = 1.0
    ob.active_shape_key_index = ob.data.shape_keys.key_blocks.find(inst.shapekey_n)
    
    inst.direct = np.random.randn(inst.vc,3)
    
    return



def oneStep_PDiff():
    current_mesh_obns = [i.name for i in bpy.data.objects if i.type=='MESH']
    insts = [MKVBS_data['inst'][i] for i in MKVBS_data['inst'].keys() if MKVBS_data['inst'][i].loop_switch and MKVBS_data['inst'][i].base_obn in current_mesh_obns]
    print(insts)
    insts += [MKVBS_data['inst'][i] for i in MKVBS_data['inst'].keys() if MKVBS_data['inst'][i].one_step and MKVBS_data['inst'][i].base_obn in current_mesh_obns]
    
    for inst in insts:
        base_ob = bpy.data.objects[inst.base_obn]
        if base_ob.MKVBS.one_step:
            base_ob.MKVBS.one_step = False
        
        before_vis_co = np.empty(inst.vc*3, dtype=np.float64)
        glob_dg.objects[inst.base_obn].data.vertices.foreach_get('co', before_vis_co)
        before_vis_co = np.reshape(before_vis_co,(inst.vc, 3))
        
        before_sk_co = np.empty(inst.vc*3, dtype=np.float64)
        base_ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_get('co', before_sk_co.ravel())
        before_sk_co = np.reshape(before_sk_co, (inst.vc, 3))
        
        base_ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_set('co', (inst.base_co + inst.direct).flatten())
        glob_dg.update()
        base_ob.data.update()
        
        after_vis_co = np.empty(inst.vc*3, dtype=np.float64)
        glob_dg.objects[base_ob.name].data.vertices.foreach_get('co', after_vis_co)
        after_vis_co = np.reshape(after_vis_co,(inst.vc, 3))
        
        
        before_loss_v = inst.target_co - before_vis_co
        after_loss_v = inst.target_co - after_vis_co
        
        before_loss = np.sqrt(np.einsum('ij,ij->i', before_loss_v, before_loss_v))
        after_loss = np.sqrt(np.einsum('ij,ij->i', after_loss_v, after_loss_v))
        
        after_sk_co = np.empty(inst.vc*3, dtype=np.float64)
        base_ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_get('co', after_sk_co)
        after_sk_co = np.reshape(after_sk_co,(inst.vc, 3))
        
        diff_loss = before_loss > after_loss
        
        before_co = (1-diff_loss)[:,None] * before_sk_co
        current_co = diff_loss[:,None] * after_sk_co
        update_apply = before_co + current_co
        # update_apply = ( (1-diff_loss)[:,None] * before_sk_co ) + ( diff_loss[:,None] * after_sk_co )
        
        
        base_ob.data.shape_keys.key_blocks[inst.shapekey_n].data.foreach_set('co', update_apply.flatten())
        glob_dg.update()
        base_ob.data.update()
        
        inst.use_learning_rate = (np.random.normal(0,1,(inst.vc,1))) * base_ob.MKVBS.learning_rate
        
        current_direct_v = (diff_loss[:,None] * inst.direct) * base_ob.MKVBS.learning_rate
        new_direct_v = ((1-diff_loss)[:,None] * np.random.randn(inst.vc,3)) * inst.use_learning_rate[None]
        
        # least_direct_b = (1e-6 >= (current_direct_v + new_direct_v))
        # least_direct = least_direct_b[:,None] * inst.direct
        # stay_direct = (1-least_direct_b)[:,None] * (current_direct_v + new_direct_v)
        
        inst.direct = current_direct_v + new_direct_v
        # inst.direct = least_direct + stay_direct
        # inst.direct = ( least_direct_b[:,None] * inst.direct ) + ( (1-least_direct_b)[:,None] * (current_direct_v + new_direct_v) )
        
        
        # print('')
    return



@persistent
def update_realtime(scene=None):
    oneStep_PDiff()
    print('realtime')
    return 0.0



class MKVBS_PropsObject(bpy.types.PropertyGroup):
    inst_id:\
    bpy.props.FloatProperty(name="Instance ID", description="Unique Instance ID")
    
    target_obn:\
    bpy.props.StringProperty(name="Visibility Target Object Name", description="Set Target object name", update=cb_target)
    
    shapekey_n:\
    bpy.props.StringProperty(name="Receive Shapekey", description="Set Target object", default=shapekey_n)
    
    threshold:\
    bpy.props.FloatProperty(name="threshold", description="Threshold about difference", default=threshold)
    
    learning_rate:\
    bpy.props.FloatProperty(name="learning_rate", description="Update step", default=learning_rate, min=0.0)
    
    true_target:\
    bpy.props.BoolProperty(name="True Target", description="", default=False)
    
    loop_switch:\
    bpy.props.BoolProperty(name="loop_switch", description="realtime loop on / off", default=False)
    
    one_step:\
    bpy.props.BoolProperty(name="One Step", description="Just One loop on / off", default=False)




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
            col.prop(ob.MKVBS, "loop_switch", text="Auto Loop On / Off", icon='CON_PIVOT')
            
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(ob.MKVBS, "one_step", text="Just One Step", icon='CON_PIVOT')
            
            layout.separator()
            
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(ob.MKVBS, "threshold", text="Difference Threshold", icon='FULLSCREEN_EXIT')
            
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.prop(ob.MKVBS, "learning_rate", text="Leanring Rate", icon='FULLSCREEN_EXIT')
            
            # layout.separator()





classes = (
    MKVBS_PropsObject,
    PANEL_PT_MKVBS,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)
    
    bpy.types.Object.MKVBS = bpy.props.PointerProperty(type=MKVBS_PropsObject)
    
    bpy.app.timers.register(update_realtime, persistent=True)
    print('register_realtime')
        


def unregister():
    # pass
    from bpy.utils import unregister_class

    for cls in classes:
        unregister_class(cls)
    
    del(bpy.types.Object.MKVBS)
    
    bpy.app.timers.unregister(update_realtime)
    print('unregister_realtime')



if __name__ == "__main__":
    register()