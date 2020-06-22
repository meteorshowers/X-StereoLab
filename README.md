### StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth prediction model in pytorch. ECCV2018
### ActiveStereoNet:End-to-End Self-Supervised Learning for Active Stereo Systems ECCV2018 Oral


### If you want to communicate with me about the StereoNet, please concact me without hesitating. My email: 
### xuanyili.edu@gmail.com  
### my model result

Now, my model's speed can achieve 25 FPS on 540*960 img with the best result of 1.87 EPE_all with 16X multi model， 1.95 EPE_all with 16X single model on sceneflow dataset by end-to-end training. 
the following are the side outputs and the prediction example
#### train example
![train example](https://github.com/meteorshowers/StereoNet/blob/master/doc/iter-21200.jpg)
#### test example
![test example](https://github.com/meteorshowers/StereoNet/blob/master/doc/iter-70.jpg)
[^_^]:

![test example](https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/figure2.png)
over 100FPS on titan xp gpu


#### large example
[^_^]:

![test example](https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/fig/figure3.png)

16FPS on titan xp gpu
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

### Introduction 
I implement the real-time  stereo model according to the   StereoNet   model in pytorch.
The speed can reach 30FPS with top performance.
The speed can reach 60FPS with lower performance.

| Method |EPE_all on sceneflow dataset |EPE_all on kitti2012 dataset|EPE_all on kitti2015 dataset|
|:---|:---:|:---:|:---:|
|ours(8X single)| stage0:2.26 stage1:1.38|    |   |
| Reference[1]| stage1: 1.525 |    |   |

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

* Sceneflow pretrain weight https://drive.google.com/open?id=1bSwewxrRfmFCxZDyAtyYyQQiw05nSFI8.

### Thanks

* Thanks to  <a href="https://github.com/samehkhamis"> Sameh Khamis' help

