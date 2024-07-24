import torch
import torch.nn as nn # 专用于搭建申请网络的包，neural network
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(), # 交叉熵
            'MSE': nn.MSELoss() # 均方差
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests), # lr是学习率
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                # 找一个图片可视化一下~
                # plt.imshow(inputs[0].reshape(28, 28), cmap=plt.cm.binary)
                # plt.show()

                self.optimizer.zero_grad() # 梯度清零

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels) # 输出和目标值
                loss.backward() # 反向传播
                self.optimizer.step() # 更新参数

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose( # 将多个数据转换操作组合成一个序列
        [transforms.ToTensor(), # ToTensor() 转换操作，用于将 PIL 图像或 numpy 数组转换为 PyTorch 的张量
         transforms.Normalize([0, ], [1, ])] # 对张量进行标准化,[0, ] 表示每个通道的均值为0，[1, ] 表示每个通道的标准差为1
    )

    # train为True的话，数据集下载的是训练数据集；download是False的话，则会从目标位置读取数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # train为False的话则下载测试数据集
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


# 必须继承，并且实现__init__函数和forward函数
class MnistNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512) # 输入为28*28的图片，转成了列向量的形式
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)