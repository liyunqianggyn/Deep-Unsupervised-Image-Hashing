# Deep Unsupervised Image Hashing by Maximizing Bit Entropy

This is the PyTorch implementation of accepted AAAI 2021 paper: [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/abs/2012.12334)

<!-- Our paper presentation is on [YouTube](https://www.youtube.com/watch?v=riZDqdTrNrg) -->

<!-- Meantime, a re-implemented version of our work: [Training code](https://github.com/swuxyj/DeepHash-pytorch)
 -->

## Proposed Bi-half layer
<table border=0 >
	<tbody>
    <tr>
		<tr>
			<td width="19%" align="center"> A simple, parameter-free, bi-half coding layer to maximize hash
channel capacity
  </td>
			<td width="40%" > <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/bi_half_layer.png"> </td>
		</tr>
	</tbody>
</table>


## Datasets and Architectures on different settings
Experiments on **5 image datasets**:
Flickr25k, Nus-wide, Cifar-10, Mscoco, Mnist, and **2 video
datasets**: Ucf-101 and Hmdb-51. 
According to different settings, we divided them into: i) Train an AutoEncoder on Mnist; ii) Image Hashing on Flickr25k, Nus-wide, Cifar-10, Mscoco using Pre-trained Vgg; iii) Video Hashing on Ucf-101 and Hmdb-51 using Pre-trained 3D models.


### Glance

```
3 settings ── AutoEncoder ── ── ── ── ImageHashing ── ── ── ── VideoHashing      
               ├── Sign.py             ├── Cifar10_I.py          └── main.py
               ├── SignReg.py          ├── Cifar10_II.py
               └── BiHalf.py           ├── Flickr25k.py
    	     			       └── Mscoco.py
```


### Datasets download

|#|Datasets|Download|
|---|----|-----|
|1|Flick25k|[Link](https://press.liacs.nl/mirflickr/mirdownload.html)
|2|Mscoco|[Link](https://drive.google.com/file/d/0B7IzDz-4yH_HN0Y0SS00eERSUjQ/view?usp=sharing )|
|3|Nuswide|[Link](https://github.com/TreezzZ/DSDH_PyTorch)  |
|4|Cifar10|[Link](https://www.cs.toronto.edu/~kriz/cifar.html)|
|5|Mnist|[Link](http://yann.lecun.com/exdb/mnist/)|
|6|Ucf101|[Link](https://surfdrive.surf.nl/files/index.php/s/dnYpOzKSmZFxvtX)|
|7|Hmdb51|[Link](https://surfdrive.surf.nl/files/index.php/s/q8Oqu4orntKH79p)|

For video datasets, we converted them from avi to jpg files. The original avi videos can be download: [Ucf101](https://www.crcv.ucf.edu/data/UCF101.php) and [Hmdb51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).


### Implementation Details for Video Setup
For the video datasets ucf101 and hmdb51, to generate a training sample, we first select a video frame by uniform sampling, and then generate a 16-frame clip around the
frame. If the selected position has less than 16 frames before the video ends, then we repeat the procedure until it fits.
We spatially resize the cropped sample to 112 x 112 pixels, resulting in one training sample with size of **3 channels x 16 frames x 112 pixels x 112 pixels**. In the retrieval, we adopt sliding window to generate  clips as input, i.e, each video is split into non-overlapping **16-frame clips**. Each video has an average 92 non-overlapped clips.
Take the ucf101 for example, we obtain a query set of 3,783 videos containing  348,047 non-overlapped clips, and the retrieval set of 9,537 videos containing 891,961 clips.
We then input the non-overlapped clips to extract binary descriptors for hashing. For more details, please see the [paper](https://arxiv.org/abs/1711.09577).


### Pretrained model
You can download kinetics pre-trained 3D models: ResNet-34  and ResNet-101 [here](https://github.com/kenshohara/3D-ResNets-PyTorch).   

------



## 3D Visualization
The continuous feature visualization on an AutoEncoder using Mnist. We compare 3 different models: sign layer, sign+reg and our bi-half layer.

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> <strong>Sign Layer</strong> </td>
			<td width="27%" align="center"> <strong>Sign + Reg</strong> </td>
			<td width="27%" align="center"> <strong>Bi-half Layer</strong> </td>
		</tr>
<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/sign_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/Signreg_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/bihalf_.gif"> </td>
		</tr>
	</tbody>
</table>


## Citation

If you find the code in this repository useful for your research consider citing it.

```
@article{liAAAI2021,
  title={Deep Unsupervised Image Hashing by Maximizing Bit Entropy},
  author={Li, Yunqiang and van Gemert, Jan},
  journal={AAAI},
  year={2021}
}
```
## Contact
If you have any problem about our code, feel free to contact

 - Y.Li-19@tudelft.nl
 - J.C.vanGemert@tudelft.nl
 
 

 
 
 
 


