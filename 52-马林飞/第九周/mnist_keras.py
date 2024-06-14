from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2

[1]
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape= ', train_images.shape)
print('train_labels= ', train_labels)
print('test_images.shape= ', test_images.shape)
print('test_labels= ', test_labels)

[2]
# img = train_images[1]
# img = cv2.resize(img, (256, 256))
# cv2.imshow('train_images', img)
# cv2.waitKey(0)

digit = train_images[0]
plt.imshow(digit, cmap=plt.cm.binary)  # cmap 参数指定图像的颜色映射（colormap）。plt.cm.binary 表示使用二值颜色映射（黑白色调）
plt.show()

[3]
network = models.Sequential()
network.add(layers.Dense(units=512, activation='relu', input_shape=(28 * 28,)))  # 隐藏层 Dense用于分类
network.add(layers.Dense(units=10, activation='softmax'))  # 输出层，分10类
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

[4]
train_images = train_images.reshape((60000, 28 * 28))  # 变成一维
train_images = train_images.astype('float32') / 255  # 归一化

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

print('before change:', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:', test_labels[0])

[5]
history = network.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels),
                      verbose=2)

[6]
"""
输入测试集，进行测试
verbose =1 打印训练进度条
test_acc 准确率
"""
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=2)
print(test_loss, test_acc)

[7]
"""
模拟推理
"""

res = network.predict(test_images)

print('res.shape=', res.shape)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is:", i)
        break

[8]
epochs = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(12, 4))

# 绘制训练和验证的损失值
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['loss'], 'r', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的准确率
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['acc'], 'bo', label='Training accuracy')
plt.plot(epochs, history.history['val_acc'], 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

print(history.history.keys())  # dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
