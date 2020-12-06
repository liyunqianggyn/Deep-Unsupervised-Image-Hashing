# Deep Unsupervised Image Hashing

This is the PyTorch implementation of [Deep Unsupervised Image Hashing by Maximizing Bit Entropy]()


## Illustrative 2D example
<table border=0 >
	<tbody>
    <tr>
		<tr>
			<td width="19%" align="center"> The proposed bi-half layer.
    $M$ is the mini-batch size and $K$ is the feature dimensions.
    A bi-half layer (middle part in white and blue) is used to quantize continuous features in $\mathbf{U}$ into binary codes in $\mathbf{B}$ via minimizing $W_1(P_u, P_b)$ in Eq.(\ref{1_wesserstan_distance}). The assignment strategy is the optimal probabilistic coupling $\pi_0$.
     For each bit, \ie per column of $\mathbf{U}$, we first rank its elements and
      then the top half elements is assigned to $+1$ and the remaining half elements to $-1$.   In contrast,  the commonly used sign function directly during training assigns the continuous
       features to their nearest binary codes which minimizes the Euclidean distance. The blue boxes indicate where our method differs from the sign function as the code in that position should flip. </td>
			<td width="40%" > <img src="https://raw.githubusercontent.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing-by-Maximizing-Bit-Entropy/master/bi_half layer.png"> </td>
		</tr>
	</tbody>
</table>


