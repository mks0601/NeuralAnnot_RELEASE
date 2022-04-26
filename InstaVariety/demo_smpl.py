import json
import numpy as np
import cv2
import torch
import smplx
from pycocotools.coco import COCO
import os.path as osp
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
import joblib

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def render_mesh(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img

def demo():
    target_aid = 0
    
    #smpl_path = '/home/mks0601/workspace/human_model_files'
    smpl_path = '/mnt/backbone-nfs/3d_pose_project/codebase/human_model_files'
    smpl_layer = smplx.create(smpl_path, 'smpl')
   
    db = joblib.load(osp.join('data', 'annotations', 'insta_train_db.pt'))
    # smpl parameter load
    with open(osp.join('data', 'annotations', 'InstaVariety_train_SMPL_NeuralAnnot.json'),'r') as f:
        smpl_params = json.load(f)

    for i, aid in enumerate(db['ann_id']):
        img_path = osp.join('data', 'images', db['file_name'][i])
        height, width = db['img_shape'][i]

        #if aid != target_aid:
        #    continue
        print(aid)
        
        # image path and bbox
        bbox = process_bbox(db['bbox'][i].astype(np.float32), width, height)
        if bbox is None:
            print('invalid bbox')
            break
       
        # smpl parameter
        smpl_param = smpl_params[str(aid)]
        pose, shape, trans = smpl_param['smpl_param']['pose'], smpl_param['smpl_param']['shape'], smpl_param['smpl_param']['trans']
        pose = torch.FloatTensor(pose).view(-1,3) # (24,3)
        root_pose = pose[0,None,:]
        body_pose = pose[1:,:]
        shape = torch.FloatTensor(shape).view(1,-1) # SMPL shape parameter
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
      
        # get mesh and joint coordinates
        with torch.no_grad():
            output = smpl_layer(betas=shape, body_pose=body_pose.view(1,-1), global_orient=root_pose, transl=trans)
        mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()

        # mesh render
        img = cv2.imread(img_path)
        focal = smpl_param['cam_param']['focal']
        princpt = smpl_param['cam_param']['princpt']
        rendered_img = render_mesh(img, mesh_cam, smpl_layer.faces, {'focal': focal, 'princpt': princpt})
        cv2.imwrite('smpl.jpg', rendered_img)
        
        break

if __name__ == "__main__":
    demo()
