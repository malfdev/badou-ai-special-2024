from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(tf.__version__)

print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

print(train_images[0].shape)

plt.imshow(train_images[0])
plt.show()

# 数据预处理
train_images = train_images.astype('float32') / 255  # 归一化
test_images = test_images.astype('float32') / 255
# one hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 创建网络
network = models.Sequential()  # 模型初始化
network.add(layers.Flatten(input_shape=(28, 28)))
network.add(layers.Dense(units=512, activation='relu', input_shape=(28, 28, 1)))
network.add(layers.Dropout(rate=0.5))
network.add(layers.Dense(units=10, activation="softmax"))
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# 训练
history = network.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=1,
                      validation_data=(test_images, test_labels))

# 测试
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)

epochs = range(1, len(history.history['loss']) + 1)

print(history.history.keys())

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
