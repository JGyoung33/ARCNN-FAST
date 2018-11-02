
# ARCNN-FAST tensorflow
[Deep Convolution Networks for Compression Artifacts Reduction] (https://arxiv.org/pdf/1608.02778.pdf)

#### ARCNN model architecture
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/ARCNN-FAST/master/asset/arcnnmodel.png" width="600">
</p>

</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/ARCNN-FAST/master/asset/arcnnmodel2.png" width="600">
</p>


#### ARCNN result (original, jpg20, reconstructed by arcnn on jpg20)
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/ARCNN-FAST/master/asset/arcnnresult.png" width="600">
</p>

#### loss history
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/ARCNN-FAST/master/asset/losshist.png" width="600">
</p>





## Prerequisites
 * python 3.x
 * Tensorflow > 1.5
 * matplotlib
 * argparse
 * opencv2
 
## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3
 * This implements supports color channels based on YCbCr. 
 * This implements reached PSRN score over paper baseline (on Y channel)



## Author
Dohyun Kim



