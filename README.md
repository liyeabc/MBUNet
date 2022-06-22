# MBUNet
### Prerequisites
* Pytorch 1.1
* cuda 9.0
* python 3.6
* GPU Memory>20G We Recommend Titan RTX or Tesla V100

### Datasets
We evaluate our method on Celeb-reID, Celeb-reID-light, PRCC, LTCC, and VC-Clothes datasets. You can download datasets from  ([BaiDuDisk](https://pan.baidu.com/s/1fKvMwaOI0T8RJayC3LqbJA) ```pwd:ew5h```)  , and put them into /MBU_reid/data/.

### Pre-trained Model
ResNet-50: ([BaiDuDisk](https://pan.baidu.com/s/19qbp_WYjtXnpfnIyNGDStg) ```pwd: ro4l ```) 
Pose_hrnet:([BaiDuDisk](https://pan.baidu.com/s/19qbp_WYjtXnpfnIyNGDStg) ```pwd: ro4l ```) 
STN model: ([BaiDuDisk](https://pan.baidu.com/s/19qbp_WYjtXnpfnIyNGDStg) ```pwd: ro4l ```) 



Put the pre-trained ResNet-50 model in the folder "MBU_reid". 
```
MBU_reid/resnet50-19c8e357.pth
```
Put the pre-trained Pose_hrnet model in the folder " fastreid ". 
```
fastreid/modeling/models/model_keypoints/pose_hrnet_w48_256x192.pth 
```

Put the pre-trained STN model in the folder " MBU_reid ". 
```
 MBU_reid/stn/model_best_stn.pth 
```




### Train
`cd` to folder:
```
 cd MBU_reid
```
If you want to train the model, run:
```
CUDA_VISIBLE_DEVICES=0 train_net.py --config-file= "configs/MBU_baseline.yml"
```


### Evaluation
To evaluate a model's performance, use:
```
CUDA_VISIBLE_DEVICES=0 train_net.py --config-file= "configs/MBU_baseline.yml" --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```



