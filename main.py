"""
kerasでCNNやってみる
https://qiita.com/yoyoyo_/items/0034e5e82813b05e41df
"""

"""
GPU使用
"""
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)

"""
ネットワーク定義
まずはSequential()でインスタンスを作り、add()メソッドでlayerを追加する
一番最初のConv2Dにはinput_shapeという引数を入れる
これによって入力サイズを定義
今回は２クラス分類とするので、最終層の出力はnum_classes=2
入力サイズは64x64
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

num_classes = 2

def Mynet():
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

return model

"""
学習の準備
"""
import keras
model = Mynet()

#各層のパラメータを学習させるように設定
for layer in model.layers:
    layer.trainable = True

"""
次にモデルをコンパイルします。
コンパイルはoptimizerのセットのこと 今回はSGDを設定
lossはタスクの種類 今回は多クラス分類なのでcategorical_crossentropy
lrは学習率、decayは重み減衰、momentumはモーメンタム
"""
model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

