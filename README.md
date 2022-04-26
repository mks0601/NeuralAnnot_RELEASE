
# **NeuralAnnot: Neural Annotator for 3D Human Mesh Training Sets**
  
## Introduction  
This repo provides 3D pseudo-GTs (SMPL/MANO/FLAME/SMPL-X parameters) of various datasets, obtained by NeuralAnnot.
* For all 3D pseudo-GTs, rendered meshes will be placed on the original image without cropping humans using bboxs.
* Key of 3D pseudo-GTs of **MSCOCO, MPII 2D Pose Dataset, 3DPW, CrowdPose, FFHQ, and InstaVariety** are annotation id, contained in parsed json data files. Please see [[this example](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/MSCOCO/MSCOCO.py#L114)]
* To obtain 2D/3D joint/mesh vertex coordinates from the SMPL/MANO/FLAME parameters, see [[this function](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/common/utils/preprocessing.py#L202)].
* To obtain 2D/3D joint/mesh vertex coordinates from the SMPL-X parameters, see [[this function](https://github.com/mks0601/Hand4Whole_RELEASE/blob/4586f0cf4274d4529f6b119751d1d98f8fb39e3c/common/utils/preprocessing.py#L199)].

  
## Human3.6M
* [[data](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK?usp=sharing)]
* [[SMPL parameters](https://drive.google.com/drive/folders/1xLkuyrjB832o5aG_M3g3EEf0PXqKkvS8?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/Human36M/Human36M.py#L100)]
* [[SMPL-X parameters](https://drive.google.com/drive/folders/1opns6ta471PPzvVhhm9Anv5HMd5hCdoj?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/4586f0cf4274d4529f6b119751d1d98f8fb39e3c/data/Human36M/Human36M.py#L98)]
* Forwarding SMPL/SMPL-X parameters to SMPL/SMPL-X layer produces 3D meshes in world coordinate system. The camera extrinsics should be applied to transfer them to the camera-centered coordinate system.

## MPI-INF-3DHP
* [[data](https://drive.google.com/drive/folders/1oHzb4oJHPZllLgN_yjyatp1LdqdP0R61?usp=sharing)]
* [[SMPL parameters](https://drive.google.com/file/d/1mxyPTnwM7D5L0NhtSEY1-pl3k5mS2IV6/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/MPI_INF_3DHP/MPI_INF_3DHP.py#L74)]
* [[SMPL-X parameters](https://drive.google.com/file/d/1lBJyu95xN4EhDyDA1GLkLqlh0SfAKU9a/view?usp=sharing)]
* Forwarding SMPL/SMPL-X parameters to SMPL/SMPL-X layer produces 3D meshes in world coordinate system. The camera extrinsics should be applied to transfer them to the camera-centered coordinate system.

## MSCOCO
* [[SMPL parameters](https://drive.google.com/file/d/14XDSCdvpW_fJe_plbQ9wPwLv2VjdNuYZ/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/MSCOCO/MSCOCO.py#L114)]
* [[MANO parameters](https://drive.google.com/file/d/183lpJD88LNxZH9iDJackWCQrg9LBrQ0p/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/MSCOCO/MSCOCO.py#L134)
* [[FLAME parameters](https://drive.google.com/file/d/1VIsrPk9Ub547AN3dZH-XyfZbfNl0OYtX/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/MSCOCO/MSCOCO.py#L166)]
* [[SMPL-X parameters (whole body)](https://drive.google.com/file/d/1Jrx7IWdjg-1HYwv0ztLNv0oy3Y_MOkVy/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/4586f0cf4274d4529f6b119751d1d98f8fb39e3c/data/MSCOCO/MSCOCO.py#L119)]
* Forwarding SMPL/SMPL-X parameters to SMPL/SMPL-X layer produces 3D meshes in camera-centered coordinate system. 

## MPII 2D Pose Dataset
* [[data](https://drive.google.com/drive/folders/1MmQ2FRP0coxHGk0Ntj0JOGv9OxSNuCfK?usp=sharing)]
* [[SMPL parameters](https://drive.google.com/file/d/1dvtXmRWuTw1Rv89I8uGFl-YkZbhg3Lqz/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/MPII/MPII.py#L51)]
* [[SMPL-X parameters](https://drive.google.com/file/d/13YsJra9b_EONRexNxG7k1F9zp10SiWt5/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/4586f0cf4274d4529f6b119751d1d98f8fb39e3c/data/MPII/MPII.py#L52)]
* Forwarding SMPL/SMPL-X parameters to SMPL/SMPL-X layer produces 3D meshes in camera-centered coordinate system. 

## 3DPW
* [[data](https://drive.google.com/drive/folders/1fWrx0jnWzcudU6FN6QCZWefaOFSAadgR?usp=sharing)]
* [[SMPL-X parameters](https://drive.google.com/drive/folders/1iff6d8_BJmbWCcnBGnGgLO5_n4Cvk77e?usp=sharing)]
* Forwarding SMPL-X parameters to SMPL-X layer produces 3D meshes in world coordinate system. The camera extrinsics should be applied to transfer them to the camera-centered coordinate system.

## CrowdPose
* [[SMPL parameters](https://drive.google.com/drive/folders/1cTLcsb54LcUjlaiqdmwfg0UEZJ7IftFi?usp=sharing)]
* Forwarding SMPL parameters to SMPL layer produces 3D meshes in camera-centered coordinate system. 

## FFHQ
* [[FLAME parameters](https://drive.google.com/file/d/1MtEtal-mmE9j36f_Nz160E_N1CLK07yf/view?usp=sharing)][[loading code](https://github.com/mks0601/Hand4Whole_RELEASE/blob/c33591de5a0627ea4211a08abd2f008145ef0575/data/FFHQ/FFHQ.py#L40)]
* Forwarding FLAME parameters to FLAME layer produces 3D meshes in camera-centered coordinate system. 

## InstaVariety
* [[SMPL parameters](https://drive.google.com/drive/folders/1W6LK4h6_dr1gfBMY7gas375M9LxrqDSD?usp=sharing)]
* Forwarding SMPL parameters to SMPL layer produces 3D meshes in camera-centered coordinate system. 

## InterHand2.6M
* [[MANO parameters](https://mks0601.github.io/InterHand2.6M/)]


## 
## Reference  
```  
@InProceedings{Moon_2022_CVPRW_NeuralAnnot,  
author = {Moon, Gyeongsik and Choi, Hongsuk and Lee, Kyoung Mu},  
title = {NeuralAnnot: Neural Annotator for 3D Human Mesh Training Sets},  
booktitle = {Computer Vision and Pattern Recognition Workshop (CVPRW)},  
year = {2022}  
}  
```
