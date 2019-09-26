from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

def TslNetV1(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = Conv2D(32, (3,3), padding='same', activation='relu')(visible)
    layer = Conv2D(32, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)

    model = Model(inputs=visible, outputs=layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

