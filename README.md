# YOLO-V3-Simple-Train：基于PyTorch框架的目标检测模型

更新时间：2021-11-10

重构了YOLOV3模型，只需要修改少量部分即可完成训练推理，使用迁移学习，更好的让模型收敛，配置化操作，开箱即用。代码中添加了大量注释方便大家进行阅读修改。

欢迎给小星星。

## 测试环境：

- OS：Ubuntu 20.04
- CPU：I7
- GPU：3060
- PyTorch：1.9

## 性能情况

|   训练数据集   |   权值文件名称   |    测试数据集    | 输入图片大小 | mAP 0.5 |
| :------------: | :--------------: | :--------------: | :----------: | :-----: |
| COCO-Train2017 | yolo_weights.pth | 个人安全帽数据集 |   416x416    |  72.05  |

## 权重下载

*下载连接*

## 数据集

项目提供了数据集下载脚本.

### COCO

```python
python scripts/COCO2014.sh
```

### VOC2012

```python
python scripts/VOC2012.sh
```

### VOC2007

```python
python scripts/VOC2007.sh
```

### 个人数据集

本项目采用VOC数据集格式进行数据集操作。

数据集文件夹格式为：

- 根目录：xxx/xxx/VOCdevkit/VOC2007(2012)
- 标注文件：xxx/xxx/VOCdevkit/VOC2007(2012)/Annotations
- 图片文件：xxx/xxx/VOCdevkit/VOC2007(2012)/JPEGImages

## 训练

### 步骤一：检查train.yaml文件

修改**train.yaml**配置文件，需要**重点关注**的参数有：

- classes_path:数据集类别文件，**此文件内容需要手动修改**（具体请看步骤二）
- anchors_path:数据集聚类anchors文件，**此文件内容自动生成**（具体请看步骤三）
- model_path:预加载yolov3权重文件，**此文件根据个人选择**
- dataset_root:数据集根目录，**此参数根据个人数据集进行修改**

```yaml
classes_path: 'model_data/my_classes.yaml' #数据集类别文件
anchors_path: 'model_data/my_anchors.yaml' #数据集聚类anchors

anchors_mask: [ [ 6, 7, 8 ], [ 3, 4, 5 ], [ 0, 1, 2 ] ]
model_path: 'model_data/yolo_weights.pth' #预加载yolov3权重文件
input_shape: [ 416, 416 ] #模型输入图片尺度
cuda: True

Init_Epoch: 0

#冻结训练参数设置
Freeze_Epoch: 50
Freeze_batch_size: 8
Freeze_lr: 1e-3

#全网络训练参数设置
UnFreeze_Epoch: 100
Unfreeze_batch_size: 8
Unfreeze_lr: 1e-4

Freeze_Train: True

num_workers: 8

#训练数据集路径文件
train_annotation_path: '2007_train.txt'
val_annotation_path: '2007_val.txt'

#get_anchors.py config
dataset_root: '/home/cv/AI_Data/hat_worker_voc'
year: '2007'
save_path: 'model_data/my_anchors.yaml'

#数据集分割比例参数
train_percent: 0.9 #训练集划分比例
trainval_percent: 0.9 #训练集+验证集划分比例
```

### 步骤二：检查数据集&配置my_classes.yaml

使用**utils/utils.py**进行数据集分析，将会以饼图的形式展示当前数据集存在的标签个数和分类情况。

修改**model_data/my_classes.yaml**中目标个数（nc），和目标类别名称（names）。

```yaml
nc: 2
names: ['class1','class2']	
```

![](/home/cv/PycharmProjects/YOLOV3-PyTorch/docs/Dataset_Info_Pie_Chart.png)

### 步骤三：配置train.yaml&分割数据集

运行**generate_training_file.py**将数据集进行分割，并在当前文件目录下生成训练数据集文件：2007_train.txt和验证集文件2007_val.txt。

如果需要自己设置数据集分割参数请修改**train.yaml**中的两个参数：

```yaml
#数据集分割比例参数
train_percent: 0.9 #训练集划分比例
trainval_percent: 0.9 #训练集+验证集划分比例
```

```python
python generate_training_file.py
```

### 步骤四：生成anchors

运行**get_anchors.py**生成数据集的anchors文件，文件保存在**model_data/my_anchors.yaml**中。

```python
python get_anchors.py
```

生成内容如下：

```yaml
anchors:
- 13
- 16
- 32
- 12
- 18
- 23
- 24
- 28
- 29
- 36
- 61
- 26
- 40
- 47
- 56
- 64
- 88
- 99

```

### 步骤五：训练

运行**train.py**进行模型训练，自动读取**train.yaml**中的配置信息，进行训练，权重文件在logs文件夹中。

```python
python train.py
```

## 预测

根据个人需求修改**predict.yaml**中的参数。

**需要修改的参数**

- model_path：预测模型权重路径
- classes_path：数据集类别文件
- anchors_path：数据集anchors文件
- dataset_root：数据集根目录

```yaml
model_path: 'logs/backbone_yolov3coco_hat_worker_voc-ep098-loss3.201-val_loss3.313.pth' #模型权重路径
classes_path: 'model_data/my_classes.yaml' #数据类别文件

anchors_path: 'model_data/my_anchors.yaml' #数据集anchors文件
anchors_mask: [ [ 6, 7, 8 ], [ 3, 4, 5 ], [ 0, 1, 2 ] ]

input_shape: [ 416,416 ]

confidence: 0.5 #置信度
nms_iou: 0.3 #nms iou阈值
letterbox_image: False #是否使用letterbox缩放

cuda: True

minoverlap: 0.5 #map计算中的iou阈值
dataset_root: '/home/cv/AI_Data/hat_worker_voc' #数据集根目录
```

使用**predict.py**进行模型测试，测试支持图片，文件夹，视频三种方法。

**需要修改的参数**:

- mode:选择测试类型
- image：图片路径
- video：视频路径
- dir：文件夹路径

```python
parse.add_argument('-m', '--mode', type=str, choices=['image', 'video', 'dir'], default='dir',
                   help='predict image or video or dir')
parse.add_argument('-i', '--image', type=str, default='img.jpg',
                   help='image path')
parse.add_argument('-v', '--video', type=str, default='xxxx/xxxx/xxx.mp4',
                   help='video path')
parse.add_argument('-d', '--dir', type=str, default='/home/cv/PycharmProjects/rabbitmq-proj/download/src/cloud/202193_1',
                   help='dir path')
```

```python
python predict.py
```

## 评估

使用get_mAP.py进行模型评估，评估结果保存在map_out文件夹中

```python
python get_mAP.py
```

## END

调参是个需要耐心的过程。
