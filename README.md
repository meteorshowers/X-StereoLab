### Stereo Matching in O(1) with Slanted Support Windows
### StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth prediction model in pytorch. ECCV2018
### ActiveStereoNet:End-to-End Self-Supervised Learning for Active Stereo Systems ECCV2018 Oral
### HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching


### If you want to communicate with me about the StereoNet, please concact me without hesitating. My email: 
### xuanyili.edu@gmail.com 

### StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth prediction model in pytorch. ECCV2018

### StereoNet Introduction 
I implement the real-time  stereo model according to the   StereoNet   model in pytorch.

| Method |EPE_all on sceneflow dataset |EPE_all on kitti2012 dataset|EPE_all on kitti2015 dataset|
|:---|:---:|:---:|:---:|
|ours(8X single)| stage0:2.26 stage1:1.38|    |   |
| Reference[1]| stage1: 1.525 |    |   |

### my model result

Now, my model's speed can achieve 25 FPS on 540*960 img with the best result of 1.87 EPE_all with 16X multi model， 1.95 EPE_all with 16X single model on sceneflow dataset by end-to-end training. 
the following are the side outputs and the prediction example
#### train example
![train example](https://github.com/meteorshowers/StereoNet/blob/master/doc/iter-21200.jpg)
#### test example
![test example](https://github.com/meteorshowers/StereoNet/blob/master/doc/iter-70.jpg)

![test example](https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/figure2.png)
real time version submission
* KITTI2015 submission: http://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=19f20256af911773b2815a995644f237f229968e
  ranking 175


#### point cloud view example

![test example](https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/3dview.png)

### ActiveStereoNet:End-to-End Self-Supervised Learning for Active Stereo Systems ECCV2018 Oral

#### ActiveStereoNet model disparity vis result
![test example](https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/asn.png)

#### ActiveStereoNet model surface normal vis result
![test example](https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/normal.png)

#### plane fit mertric result


<div align=center>
<img src="https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/plane_fit.png" width="80%" height="80%">
</div>


#### ActiveStereoNet youtube video demo

* youtube video https://www.youtube.com/watch?v=pqKZs1b1b0Y.



<div align=center>
<img src="https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/video.png" width="80%" height="80%">
</div>


### Citation
* refercence[1]

If you find our work useful in your research, please consider citing:

@inproceedings{khamis2018stereonet,
  title={Stereonet: Guided hierarchical refinement for real-time edge-aware depth prediction},
  author={Khamis, Sameh and Fanello, Sean and Rhemann, Christoph and Kowdle, Adarsh and Valentin, Julien and Izadi, Shahram},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany},
  pages={8--14},
  year={2018}
}



### License

* Our code is released under MIT License (see LICENSE file for details).

### Installaton

* python3.6
* pytorch0.4

### Usage

* run main8Xmulti.py

### Updates

* finetune the performance beating the original paper.

### rethink

* Do not design massive deep networks with multiple stages to improve kitti by 1%(no meaning doing this)
* Use metrics that matter for visual navigation (hint: not L1 depth error)
* ...
### pretrain model
#### StereoNet pretrain model(pytorch version)
* Sceneflow pretrain weight https://drive.google.com/open?id=1bSwewxrRfmFCxZDyAtyYyQQiw05nSFI8.
#### ActiveStereoNet pretrain model(pytorch version)
* D435 pretrain weight https://drive.google.com/file/d/1MDbRy4jO3IWM0zqn_D0sbZVjECZIl4g3/view?usp=sharing.
#### ActiveStereoNet pretrain model(tensorflow version)
* D435 pretrain weight https://drive.google.com/open?id=1bSwewxrRfmFCxZDyAtyYyQQiw05nSFI8.

### Citation
* refercence[1]

If you find this work useful in your research, please consider citing:

@article{tankovich2020hitnet,
  title={HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching},
  author={Tankovich, Vladimir and H{\"a}ne, Christian and Fanello, Sean and Zhang, Yinda and Izadi, Shahram and Bouaziz, Sofien},
  journal={arXiv preprint arXiv:2007.12140},
  year={2020}
}

@inproceedings{tankovich2018sos,
  title={Sos: Stereo matching in o (1) with slanted support windows},
  author={Tankovich, Vladimir and Schoenberg, Michael and Fanello, Sean Ryan and Kowdle, Adarsh and Rhemann, Christoph and Dzitsiuk, Maksym and Schmidt, Mirko and Valentin, Julien and Izadi, Shahram},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={6782--6789},
  year={2018},
  organization={IEEE}
}

@inproceedings{fanello2017low,
  title={Low compute and fully parallel computer vision with hashmatch},
  author={Fanello, Sean Ryan and Valentin, Julien and Kowdle, Adarsh and Rhemann, Christoph and Tankovich, Vladimir and Ciliberto, Carlo and Davidson, Philip and Izadi, Shahram},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
  pages={3894--3903},
  year={2017},
  organization={IEEE}
}


### Thanks

* Thanks to  <a href="https://github.com/samehkhamis"> Sameh Khamis' help

