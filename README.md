# Deep Unsupervised Image Hashing by Maximizing Bit Entropy

This is the PyTorch implementation of accepted AAAI 2021 paper: [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/abs/2012.12334)

Our paper presentation is on [YouTube](https://www.youtube.com/watch?v=riZDqdTrNrg)


## Bi-half layer framework
<table border=0 >
	<tbody>
    <tr>
		<tr>
			<td width="19%" align="center"> A simple, parameter-free, bi-half coding layer to maximize hash
channel capacity
  </td>
			<td width="40%" > <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/bi_half layer.png"> </td>
		</tr>
	</tbody>
</table>


## Datasets and Architectures on different settings
Experimental results on 5 image datasets
Flickr25k, Nus-wide, Cifar-10, Mscoco, Mnist and 2 video
datasets Ucf-101 and Hmdb-51. 
We divided them into three groups according to different settings: (i) Train an AutoEncoder on Mnist; (ii) Image Hashing on Flickr25k, Nus-wide, Cifar-10, Mscoco using Pre-trained Vgg; (iii) Video Hashing on Ucf-101 and Hmdb-51 using Pre-trained 3D ResNet-34 and ResNet-101.

```
Bi-half Net in Pytorch
── settings
    ├── AutoEncoder 
    │   ├── Sign_Layer.py
    │   ├── SignReg.py
    │   └── Bihalf_Layer.py
    ├── ImageHashing
    │   ├── Cifar10_I.py
    │   ├── Cifar10_II.py
    │   ├── Flickr25k.py
    │   └── Mscoco.py
    └── VideoHashing
        └── main.py
```


------
### Datasets download

|#|Datasets|Download|
|---|----|-----|
|1|Mscoco|[Here](https://drive.google.com/file/d/0B7IzDz-4yH_HN0Y0SS00eERSUjQ/view?usp=sharing "悬停显示")|
|2|Cifar10|[Here](https://www.cs.toronto.edu/~kriz/cifar.html)|


<table>
    <tr>
        <td>Datasets</td><td>Mscoco</td><td>Cifar10</td><td>paper</td>
    </tr>
    <tr>
        <td >Link</td><td > [here](https://drive.google.com/file/d/0B7IzDz-4yH_HN0Y0SS00eERSUjQ/view?usp=sharing) </td> <td >[here](https://www.cs.toronto.edu/~kriz/cifar.html)</td> <td >0.6755</td>
    </tr>
</table>

 




Fliker25K: 

## 3D Visualization
This figure visualizes the continuous feature distributions before binarization over different methods by training the network on MNIST with 3 hash bits. We observe that the features learned by sign layer are seriously tangled with each other. By adding an entropy regularization term, the feature tanglement can be mitigated, but it is suboptimal solution which
requires careful hyper-parameter tuning. The proposed bihalf layer can learn evenly distributed features. 

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

 - y.li-19@tudelft.nl
 - J.C.vanGemert@tudelft.nl
 
 

 
 
 
 


