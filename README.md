# YOLO-V3-Simple-Train：基于PyTorch框架的目标检测模型

更新时间：2021-11-10

重构了YOLOV3模型，只需要修改少量部分即可完成训练推理，使用迁移学习，更好的让模型收敛，配置化操作，开箱即用。代码中添加了大量注释方便大家进行阅读修改。

欢迎给小星星。

**新增实现（可选）**

- 余弦退火学习率
- Mosaic 图像增强
- SPP Moudule
- Mish激活函数

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

权重包含，COCO，VOC，Head，Helmet，Person

下载连接

[Download](https://pan.baidu.com/s/1CAlqyszbB1sRvOVtCkC5Ag)  密码: rcs4

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

### 数据集格式转换

此项目需要提供VOC数据集格式，项目提供了一个YOLO数据集转VOC数据集的python文件：utils/utils_dataset.py

YOLO数据集格式

- images：图片文件夹
- labels：标签文件夹
- classes.txt：标签分类文件

VOC数据集格式

- VOCdevkit/VOC2007(2012)/Annotations：xml标签文件夹
- VOCdevkit/VOC2007(2012)/ImageSets：数据集划分文件夹
- VOCdevkit/VOC2007(2012)/JPEGImages：图片文件夹

添加YOLO数据集根目录，文件执行后将在YOLO根目录下自动创建 VOCdevkit/VOC2007(2012)/Annotations，VOCdevkit/VOC2007(2012)/JPEGImages两个文件夹，并且自动转换xml标签文件于Annotations中，但是JPEGImages文件夹为空，需要自己手动把图片放进去。

```python
parser = argparse.ArgumentParser('YOLO TO VOC')
parser.add_argument('-r', '--root', type=str, default='D:\\ai_data\\yolo_helmet_train\\anno',
                    help='yolo dataset root')
args = parser.parse_args()
```



## 训练

### 步骤一：检查train.yaml文件

修改**train.yaml**配置文件，**object**为检测类型字典，**obj_type**决定训练时候使用那种物体检测

- classes_path:数据集类别文件，**此文件内容需要手动修改**（具体请看步骤二）
- anchors_path:数据集聚类anchors文件，**此文件内容自动生成**（具体请看步骤三）
- model_path:预加载yolov3权重文件，**此文件根据个人选择，当为""时候不加载**
- dataset_root:数据集根目录，**此参数根据个人数据集进行修改**



**参数解析**

- input_shape：图像缩放尺寸
- spp：当为[]时候不使用spp module
- activation：在leaky_relu和mish之间进行选择，作用在darknet中
- mosaic：数据增强部分是否使用mosaic增强
- cosine_lr：是否使用余弦退火算法
- label_smoothing：平滑标签参数
- train_percent: 训练集分割比例
- trainval_percent: 训练集+验证集分割比例

```yaml
object:
  reflective:
    dataset_root: '/home/cv/AI_Data/reflective'
    model_path: 'weights/reflective-ep124-loss4.193.pth' #预加载yolov3权重文件
    classes_path: 'data/reflective_classes.yaml' #数据集类别文件
    anchors_path: 'data/reflective_anchors.yaml' #数据集聚类anchors
  person:
    dataset_root: '/home/cv/AI_Data/person_yolo'
    model_path: 'weights/person.pth' #预加载yolov3权重文件
    classes_path: 'data/person_classes.yaml' #数据集类别文件
    anchors_path: 'data/person_anchors.yaml' #数据集聚类anchors
  widerperson:
    dataset_root: '/home/cv/AI_Data/widerperson'
    model_path: 'weights/person.pth' #预加载yolov3权重文件
    classes_path: 'data/widerperson_classes.yaml' #数据集类别文件
    anchors_path: 'data/widerperson_anchors.yaml' #数据集聚类anchors
  helmet:
    dataset_root: '/home/cv/AI_Data/yolo_helmet_train\\anno'
    model_path: 'weights/helmet.pth' #预加载yolov3权重文件
    classes_path: 'data/helmet_classes.yaml' #数据类别文件
    anchors_path: 'data/helmet_anchors.yaml' #数据集anchors文件
  head:
    dataset_root: '/home/cv/AI_Data/head_datas_yolo' #数据集根目录
    model_path: 'weights/head_spp.pth'
    classes_path: 'data/head_classes.yaml' #数据类别文件
    anchors_path: 'data/head_anchors.yaml' #数据集anchors文件head

year: '2007'
obj_type: person

cuda: True
num_workers: 8

#spp: [ ] #不使用SPP 则设为[]
spp: [ 5,9,13 ]
activation: mish #leakyrelu or mish
#activation: leakyrelu #leakyrelu or mish
mosaic: True
cosine_lr: True
label_smoothing: 0

Init_Epoch: 0
#input_shape: [ 416, 416 ] #模型输入图片尺度
input_shape: [ 608,608 ] #模型输入图片尺度
anchors_mask: [ [ 6, 7, 8 ], [ 3, 4, 5 ], [ 0, 1, 2 ] ]

#generate config
train_percent: 0.9
trainval_percent: 0.9

Freeze_Train: False
#冻结训练参数设置
Freeze_Epoch: 10
Freeze_batch_size: 16
Freeze_lr: 1e-3
#全网络训练参数设置
UnFreeze_Epoch: 200
Unfreeze_batch_size: 8
Unfreeze_lr: 1e-4

#训练数据集路径文件
train_annotation_path: '2007_train.txt'
val_annotation_path: '2007_val.txt'



```

### 步骤二：检查数据集&配置my_classes.yaml

创建**data/xxx_classes.yaml**中目标个数（nc），和目标类别名称（names）如果对数据集分类不清楚的话可以调用**utils/utils.py**进行数据集分析，将会以饼图的形式展示当前数据集存在的标签个数和分类情况。

```yaml
#voc_classes.yaml
nc: 20
names:
  - aeroplane,
  - bicycle,
  - bird,
  - boat,
  - bottle,
  - bus,
  - car,
  - cat,
  - chair,
  - cow,
  - diningtable,
  - dog,
  - horse,
  - motorbike,
  - person,
  - pottedplant,
  - sheep,
  - sofa,
  - train,
  - tvmonitor
```



### 步骤三：分割数据集

运行**generate_training_file.py**将数据集进行分割，并在当前文件目录下生成训练数据集文件：2007_train.txt和验证集文件2007_val.txt。

```python
python generate_training_file.py
```

### 步骤四：生成anchors

运行`get_anchors.py`生成数据集的anchors文件，文件保存在**data/my_anchors.yaml**中，数据集中需要提供train.txt。

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

根据个人需求修改**predict.yaml**中的参数，其中检测分类按照yaml中的dict类型进行控制，此部分和train.yaml格式相同

**需要修改的参数**

- obj_type:需要使用的目标检测配置

- model_path：预测模型权重路径
- classes_path：数据集类别文件
- anchors_path：数据集anchors文件
- dataset_root：数据集根目录

```yaml
object:
  person:
    dataset_root: '/home/cv/AI_Data/HardHatWorker_voc'
    model_path: 'weights/person.pth' #模型权重路径
    classes_path: 'data/person_classes.yaml' #数据类别文件
    anchors_path: 'data/person_anchors.yaml' #数据集anchors文件
  head:
    dataset_root: '/home/cv/AI_Data/head_datas_yolo' #数据集根目录
    model_path: 'weights/head_spp.pth'
    classes_path: 'data/head_classes.yaml' #数据类别文件
    anchors_path: 'data/head_anchors.yaml' #数据集anchors文件head
  helmet:
    dataset_root: '/home/cv/AI_Data/HardHatWorker_voc'
    model_path: 'weights/helmet.pth' #模型权重路径
    classes_path: 'data/helmet_classes.yaml' #数据类别文件
    anchors_path: 'data/helmet_anchors.yaml' #数据集anchors文件
  reflective:
    dataset_root: '/home/cv/AI_Data/reflective'
    model_path: 'weights/reflective.pth' #预加载yolov3权重文件
    classes_path: 'data/reflective_classes.yaml' #数据集类别文件
    anchors_path: 'data/reflective_anchors.yaml' #数据集聚类anchors


cuda: True
obj_type: person

#spp: [ ] #不使用SPP 则设为[]
spp: [ 5,9,13 ]
activation: mish #leakyrelu or mish
anchors_mask: [ [ 6, 7, 8 ], [ 3, 4, 5 ], [ 0, 1, 2 ] ]

#input_shape: [ 416,416 ]
input_shape: [ 608,608 ]

batch_size: 1

confidence: 0.5 #置信度
nms_iou: 0.3 #nms iou阈值
letterbox_image: False #是否使用letterbox缩放

minoverlap: 0.5 #map计算中的iou阈值

onnx_model: data/yolov3_new_spp.onnx
trt_model: data/head.trt
```

使用**predict.py**进行模型测试，测试支持图片，文件夹，视频三种方法。

**需要修改的参数**:

- mode:选择测试类型（当为test的时候，使用数据为测试集数据）
- image：图片路径
- video：视频路径
- dir：文件夹路径
- Predict中的type类型为目标检测类型

```python
parse.add_argument('-m', '--mode', type=str, choices=['image', 'video', 'dir','test'], default='dir',
                   help='predict image or video or dir,test')
parse.add_argument('-i', '--image', type=str, default='img.jpg',
                   help='image path')
parse.add_argument('-v', '--video', type=str, default='xxxx/xxxx/xxx.mp4',
                   help='video path')
parse.add_argument('-d', '--dir', type=str, default='/home/cv/PycharmProjects/rabbitmq-proj/download/onnxsim/cloud/202193_1',
                   help='dir path')
predict = Predict('predict.yaml', 'person')
```

```python
python predict.py
```

## HTTP服务

项目提供了Head，Helmet，Person三种基于Flask框架的Http服务，

**URL**

- *Helmet：http://192.168.2.165:30002/yolov3_get_helmet*
- *Head：http://192.168.2.165:30002/yolov3_get_head*
- *Person：http://192.168.2.165:30002/yolov3_get_person*

**启动**

```python
python server_head.py
python server_helmet.py
python server_person.py
```

测试数据格式为 json格式，其中base64编码不包含头标签。

```json
{
    "img":["/9j/4AAQSkZ...."]
}
```



其中server_person.py中提供了人员越界检测

测试数据格式：

- image:进过base64编码的图片，不带头标签
- polys:区域坐标列表

```json
{
    "image":["/9j/4AAQSkZ...."],
    "polys":[
        	[[x1,y1],[x2,y2],...],
            [[x1,y1],[x2,y2],...],...
           ]
}
```

在文件夹experime中的area_detection.py中可以进行实时绘制区域检测

```bash
python experiment/area_detection.py
```

## 评估

使用get_mAP.py进行模型评估，评估结果保存在map_out文件夹中( TODO:need to fix bug)

```python
python get_mAP.py
```

## END

调参是个需要耐心的过程。
