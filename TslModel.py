from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

# Simple CNN
def TslNetV1(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = Conv2D(32, (3,3), padding='same', activation='relu')(visible)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

# Simple CNN on cropped data
def TslNetV2(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = Conv2D(32, (3,3), padding='same', activation='relu')(visible)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

#Add Batch Norm
def TslNetV3(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = Conv2D(32, (3,3), padding='same', activation='relu')(visible)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Flatten()(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

#Add Dropout
def TslNetV4(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = Conv2D(32, (3,3), padding='same', activation='relu')(visible)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.25)(layer)
    layer = Flatten()(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model


# more layers
def TslNetV5(input_shape, no_of_class):
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
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

# Add Image Augmentation
def TslNetV6(input_shape, no_of_class):
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
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

# Inception Module + Reduction
def TslNetV7(input_shape, no_of_class):
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
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

# Add Skip Connections
def TslNetV8(input_shape, no_of_class):
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
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model