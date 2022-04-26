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

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

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
    target_subject = 1
    target_action = 2
    target_subaction = 1
    target_frame = 0
    target_cam = 1

    db = COCO('annotations/Human36M_subject' + str(target_subject) + '_data.json')
    # camera load
    with open('annotations/Human36M_subject' + str(target_subject) + '_camera.json','r') as f:
        cameras = json.load(f)
    # joint coordinate load
    with open('annotations/Human36M_subject' + str(target_subject) + '_joint_3d.json','r') as f:
        joints = json.load(f)
    # smpl parameter load
    with open('annotations/Human36M_subject' + str(target_subject) + '_SMPL_NeuralAnnot.json','r') as f:
        smpl_params = json.load(f)

    smpl_path = '/home/mks0601/workspace/human_model_files'
    smpl_layer = smplx.create(smpl_path, 'smpl')
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
        if str(subject) != str(target_subject) or str(action_idx) != str(target_action) or str(subaction_idx) != str(target_subaction) or str(frame_idx) != str(target_frame) or str(cam_idx) != str(target_cam):
            continue
        
        # image path and bbox
        img_path = osp.join('images', img['file_name'])
        bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
        if bbox is None:
            print('invalid bbox')
            break
       
        # camera parameter
        cam_param = cameras[str(cam_idx)]
        R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
        cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
        
        # project world coordinate to cam, image coordinate space
        joint_world = np.array(joints[str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
        joint_cam = world2cam(joint_world, R, t)
        joint_img = cam2pixel(joint_cam, f, c)[:,:2]

        # smpl parameter
        smpl_param = smpl_params[str(action_idx)][str(subaction_idx)][str(frame_idx)]
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        pose = torch.FloatTensor(pose).view(-1,3) # (24,3)
        root_pose = pose[0,None,:]
        body_pose = pose[1:,:]
        shape = torch.FloatTensor(shape).view(1,-1) # SMPL shape parameter
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
      
        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        root_pose = root_pose.numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        root_pose = torch.from_numpy(root_pose).view(1,3)

        # get mesh and joint coordinates
        with torch.no_grad():
            output = smpl_layer(betas=shape, body_pose=body_pose.view(1,-1), global_orient=root_pose, transl=trans)
        mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()

        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        root_cam = joint_cam[0,None,:]
        joint_cam = joint_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t/1000 # camera-centered coordinate system
        mesh_cam = mesh_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t/1000 # camera-centered coordinate system

        # mesh render
        img = cv2.imread(img_path)
        rendered_img = render_mesh(img, mesh_cam, smpl_layer.faces, cam_param)
        cv2.imwrite('smpl.jpg', rendered_img)
        
        break

if __name__ == "__main__":
    demo()
