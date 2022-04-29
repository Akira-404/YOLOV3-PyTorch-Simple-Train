import torch
import torch.nn as nn

torch.manual_seed(seed=20200910)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 输入torch.Size([64, 1, 28, 28])
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),  # 输出torch.Size([64, 64, 28, 28])
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输出torch.Size([64, 128, 28, 28])
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)  # 输出torch.Size([64, 128, 14, 14])
        )

        self.dense = torch.nn.Sequential(  # 输入torch.Size([64, 14*14*128])
            torch.nn.Linear(14 * 14 * 128, 1024),  # 输出torch.Size([64, 1024])
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)  # 输出torch.Size([64, 10])
        )
        self.layer4cxq1 = torch.nn.Conv2d(2, 33, 4, 4)
        self.layer4cxq2 = torch.nn.ReLU()
        self.layer4cxq3 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.layer4cxq4 = torch.nn.Linear(14 * 14 * 128, 1024)
        self.layer4cxq5 = torch.nn.Dropout(p=0.8)
        self.attribute4cxq = nn.Parameter(torch.tensor(20200910.0))
        self.attribute4lzq = nn.Parameter(torch.tensor([2.0, 3.0, 4.0, 5.0]))
        self.attribute4hh = nn.Parameter(torch.randn(3, 4, 5, 6))
        self.attribute4wyf = nn.Parameter(torch.randn(7, 8, 9, 10))

    def forward(self, x):  # torch.Size([64, 1, 28, 28])
        x = self.conv1(x)  # 输出torch.Size([64, 128, 14, 14])
        x = x.view(-1, 14 * 14 * 128)  # torch.Size([64, 14*14*128])
        x = self.dense(x)  # 输出torch.Size([64, 10])
        return x


print('cuda(GPU)是否可用:', torch.cuda.is_available())
print('torch的版本:', torch.__version__)

model = Model()  # .cuda()

print("测试模型(CPU)".center(100, "-"))
print(type(model))


def init_module4all(module):
    print(f'module class name: {module.__class__.__name__}')
    print('模块', module, '正在被初始化...')



model.apply(init_module4all)
