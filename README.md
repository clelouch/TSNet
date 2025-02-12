# TSNet
# [A three-stage model for camouflaged object detection](https://doi.org/10.1016/j.neucom.2024.128784)

This repo. is an official implementation of the *TSNet* , which has been accepted in the journal *Neurocomputing, 2024*. 

The main pipeline is shown as the following, 
![TSNet](figures/framework.png)

And some results are presented
![quantitative results](figures/results.png)
![qualitative results](figures/results2.png)

## Datasets
Training and testing datasets are available at 

([Baidu](https://pan.baidu.com/s/1sJ8h8srJg0gmVE-kJLJ4og)) [code:q5e4]

([Google](https://drive.google.com/file/d/1VS8qVUjC__4BZhB-13S3wHDWAs_-YFDI/view?usp=sharing))

## Training
```
python train.py
```

## Test
```
 python test.py
```
We provide the trained model file ResNet:([Baidu](https://pan.baidu.com/s/1Mj8NIRXe8zaJMQDW8pO68g)) [code:dy4u] 

Res2Net:([Baidu](https://pan.baidu.com/s/1QOMps5qYFNOjYeu6LsiLDA)) [code:dhp5] 

PVT:([Baidu](https://pan.baidu.com/s/1gSI1RBPx55DIVlwUpx8LPw)) [code:dmbh] (note: using groupnorm)

The prediction results are also available ResNet:([Baidu](https://pan.baidu.com/s/1B2ayOEToxBbb6a8saJTJ4Q)). [code:rky9]

Res2Net:([Baidu](https://pan.baidu.com/s/1hPpACZu6nOFFaG0Seqk3rQ)). [code:t5ry]

PVT:([Baidu](https://pan.baidu.com/s/1eJReGMGXrPgnquerrB23Cw)). [code:7h4h]
 
## Citation
Please cite the `AFNet` in your publications if it helps your research:
```
@article{CHEN2022,
  title = {Adaptive fusion network for RGB-D salient object detection},
  author = {Tianyou Chen and Jin Xiao and Xiaoguang Hu and Guofeng Zhang and Shaojie Wang},
  journal = {Neurocomputing},
  year = {2023},
}
```
## Reference
[BBSNet](https://github.com/zyjwuyan/BBS-Net)
