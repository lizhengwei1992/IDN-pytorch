# IDN-pytorch
  Pytorch implement of the paper, [Fast and Accurate Single Image Super-Resolution via Information Distillation Network](https://arxiv.org/pdf/1803.09454.pdf). You can get the [caffe code and pretrained model](https://github.com/Zheng222/IDN-Caffe) given by author.    
  
  I keep the architecture network in pytorch same as the paper described, ignoring some details.
## Analysis
  The proposed network learns residual to reconstruct HR image.
  ![architecture of network](https://github.com/lizhengwei1992/IDN-pytorch/raw/master/image/architecture%20of%20proposed%20network.png)
  Imformation distillation means that  the proposed enhancement unit mixes together two different types of features and the
compression unit distills more useful information for the sequential blocks.  
  Features sliced from middle layer concatenate with features previous layer, and the result add to subsequent layer, this likes another residual dense connect in my opion.

  ![architecture of enhancement unit](https://github.com/lizhengwei1992/IDN-pytorch/raw/master/image/architecture%20of%20enhancement%20unit.png)![compress](https://github.com/lizhengwei1992/IDN-pytorch/raw/master/image/compress.png)
  
  The compression layer just reduce dimensions of channels using 1x1 convolution, same as dense net.
  

  As I know, many proposed network want to reuse previous layers as more as possiable, inclouding this paper. The RDN([paper](https://arxiv.org/pdf/1802.08797.pdf) and [code](https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch)) network seems more efficent. 
  
# Requirements
- python3.5 / 3.6
- pytorch >= 0.2
- opencv 


## Train
    
    python3 main.py --model_name IDN --load demo_IDN --dataDir ./data/ --need_patch True --blur True --patchSize 288 --nFeat_slice 4 --nFeat 64 --nDiff 16 --scale 3 --epoch 5000 --lrDecay 2000 --lr 1e-4 --batchSize 16 --lossType 'L1' --nThreads 6 
    
    
