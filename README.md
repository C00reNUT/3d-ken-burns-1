# Demo: 3d-ken-burns
[Original Author](https://github.com/sniklaus/3d-ken-burns) |
[Official Demo](https://www.youtube.com/watch?v=DYsfitr-FdY) |
[Arxiv](https://arxiv.org/abs/1909.05483)

# Example
![](./data/output.gif)
# 限制
### Technical Limitation
- 輸入影像：一張任意大小照片(可以是真實彩色、黑白影像或虛擬影像ex.動畫場景)
- 輸出影片解析度： 長寬最大為1024
- 輸出幀率： 25fps(如果要調整成其他幀率，可在`src/autozoom.py`中調整)

### Quality和適用場合
Case    |常見特徵       |損毀機率    |損毀範圍
--------|--------------|-----------|---------
Best    |1. 前後景分隔明顯<br>2. 沒有人<br>3. 沒有柱狀物體(ex. 樹、椅子<br>4. 沒有太多物件在照片中| 小|1. 小、不太影響觀看<br>2. 要仔細觀察比較看得到|
Average |1. 人物在整張照片中比例小| 中|1. 中小<br>2. 損毀部位稍明顯，但不影響整體觀看|
Worse   |1. 有柱狀物體ex.樹、路燈、椅子<br>2. 前後景交疊在一起或沒有特別明顯前後之分<br>3. 有人且人在照片中比例大|大|1. 大、破損嚴重、物件扭曲<br>2. 影像修補區域顏色模糊|

### 測試硬體規格
- OS：Ubuntu 20.04
- GPU：NVIDIA Tesla V100 SXM2 single core
- GPU Memory：30 GB
- Memory：60 GB
- CUDA Driver：460.119.04
- CUDA：11.4
- cuDNN：8.x.x
- Python version：3.8.10

### 實際硬體用量
- 模型大小： 405.9MB
- 測試連續demo 52張影像

| Test | Total Inference Time | Average Inference Time | GPU Memory Usage Peak |
|:----:|---------------------:|-----------------------:|----------------------:|
|  1   |  754s                |  14.5s                 |  5000MB               |
|  2   |  768s                |  14.7s                 |  4759MB               |
|  3   |  837s                |  16.0s                 |  4184MB               |
|  4   |  825s                |  15.8s                 |  4734MB               |

### 安裝(使用Docker)
1. 必須使用GPU，因此請確認host的cuda driver是否支援CUDA 11.1。如果沒有支援，可以使用較低版本的CUDA Image，並修改[Image]安裝的torch版本和cupy版本(需和cuda版本一致)
2. ```docker build -t 3dkbe .```
3. ```docker run -it --gpus all 3dkbe /bin/bash```

## Demo
1. pass the input image to /images
2. run ```python autozoom.py --in ./images/yourImageName.jpg --out ./autozoom.mp4```