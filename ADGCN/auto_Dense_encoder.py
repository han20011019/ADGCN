from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class AutoEncoder(object):
    def __init__(self):
        self.encoding_dim=32
        self.decoding_dim=784
        self.x_test = None
        self.model = self.auto_encoder_model()

    def auto_encoder_model(self):
        input_img = Input(shape=(784,))

        encoded = Dense(128,activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(64,activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

        auto_encoder = Model(inputs=input_img,outputs=decoded)

        auto_encoder.compile(optimizer='adam',loss='binary_crossentropy')

        return auto_encoder

    def train(self):

        data = np.loadtxt('D:/PyCharm 2023.2.1/learn/ADGCN/Few_Shot/WorldOfWarcraft.txt')

        # 分割数据集，90%作为训练集，10%作为测试集
        x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)
        x_train = x_train.astype('float32')/255.
        self.x_test = x_test.astype('float32') / 255.
        print(x_train.shape)
        print(self.x_test.shape)

        self.model.fit(x_train,x_train,
                       epochs=10,
                       batch_size=20,
                       shuffle=True,
                       validation_data=(self.x_test,self.x_test))

        return None
    def display(self):
        # (x_train,_),(x_test,_) = mnist.load_data()
        # x_test  = np.reshape(x_test,(len(x_test),np.prod(x_test.shape[1:])))

        decode_imgs = self.model.predict(self.x_test)
        # 保存重建后的数据到文本文件
        np.savetxt('D:/PyCharm 2023.2.1/learn/ADGCN/process_by_AE/WorldOfWarcraft_Dense_ae_40_784.txt', decode_imgs, fmt='%f', delimiter=' ')
        print(decode_imgs.shape)
        plt.figure(figsize=(20,4))

        n=5
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            plt.imshow(self.x_test[i].reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2,n,i+n+1)
            plt.imshow(decode_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

if __name__=='__main__':
    ae=AutoEncoder()
    ae.train()
    ae.display()