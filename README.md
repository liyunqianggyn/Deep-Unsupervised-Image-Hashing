# Deep Unsupervised Image Hashing by Maximizing Bit Entropy

This is the PyTorch implementation of accepted AAAI 2021 paper: [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/abs/2012.12334)


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



## Smooth Grad
Smooth grad is adding some Gaussian noise to the original image and calculating gradients multiple times and averaging the results [8]. There are two examples at the bottom which use _vanilla_ and _guided_ backpropagation to calculate the gradients. Number of images (_n_) to average over is selected as 50. _σ_ is shown at the bottom of the images.

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> </td>
			<td width="27%" align="center"> <strong>3D Visualization</strong> </td>
			<td width="27%" align="center"> </td>
		</tr>
<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/sign_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/Signreg_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/AutoEncoder/gif/bihalf_.gif"> </td>
		</tr>
	</tbody>
</table>




## Code is coming
Now I am on holiday, the code will come soon :)


