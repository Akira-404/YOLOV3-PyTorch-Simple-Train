# YOLO-V3-Sample-Train：基于PyTorch框架的目标检测模型

更新时间：2021-11-10

重构了YOLOV3模型，只需要修改少量部分即可完成训练推理，配置化操作，开箱即用。代码中添加了大量注释方便大家进行阅读修改。

欢迎给小星星。

## 测试环境：

- OS：Ubuntu 20.04
- CPU：I7
- GPU：3060
- PyTorch：1.9



## DataSet

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

## Train

### 步骤一：

获取数据集anchors，修改kmeans_anchors/main.py中的root参数，让其指向个人数据集路径，然后运行main.py文件，程序将自动计算个人数据集anchors并以yaml文件格式保存在model_data/my_anchors.yaml中。

```python
parse.add_argument('-r', '--root', type=str, default='/home/cv/AI_Data/hat_worker_voc',
                   help='voc dataset rot:xxx/xxx')

python kmeans_anchors/main.py
```

### 步骤二：

修改model_data/my_classes.yaml中目标个数（nc），和目标名称（names）。

```yaml
nc: 2
names: ['person','hat']
```

### 步骤三：

修改generate_training_file.py中的参数root，让其指向个人数据集路径。该文件将自动分类数据集，并在当前目录下产生用于train val的txt文件。

```python
parse.add_argument('-r', '--root', type=str,default='/home/cv/AI_Data/hat_worker_voc/VOCdevkit',
                   help='dataset root')
```

### 步骤四：

根据个人需求，修改设置yolo_config.yaml中的参数，也可以不修改。**model_path**参数用于预训练。

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
```

步骤五：

运行train.py进行模型训练，自动读取yolo_cofig.yaml中的配置信息，进行训练，权重文件在logs文件夹中。

```python
python train.py
```

## END

调参是个需要耐心的过程。