from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D

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

# Simple CNN on cropped data, model no change
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
    layer = Dropout(0.1)(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.1)(layer)
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
    layer = ZeroPadding2D(padding=(3, 3))(visible)
    layer = Conv2D(64, (7,7), padding='valid', kernel_initializer='he_normal', strides=(2,2), activation='relu')(layer)
    layer = MaxPooling2D((3,3), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = ZeroPadding2D(padding=(1, 1))(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

# Add Image Augmentation, model no change
def TslNetV6(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = ZeroPadding2D(padding=(3, 3))(visible)
    layer = Conv2D(64, (7,7), padding='valid', kernel_initializer='he_normal', strides=(2,2), activation='relu')(layer)
    layer = MaxPooling2D((3,3), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = ZeroPadding2D(padding=(1, 1))(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(256, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(512, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

def conv_block(layer, filters, kernel_size, strides=1, padding='same', batchNorm=False, activation=True):
    layer = Conv2D(filters, 
            kernel_size=kernel_size, 
            strides=strides,
            padding=padding)(layer)
    if batchNorm:
        layer = BatchNormalization()(layer)
    if activation:
        layer = Activation('relu')(layer)
    return layer

def inception_resnet_block(layer, filters, output_shape, scale, activation='relu'):
    left_branch = conv_block(layer, 32, 1)
    middle_branch = conv_block(layer, 32, 1)
    middle_branch = conv_block(middle_branch, 32, 3)
    right_branch = conv_block(layer, 32, 1)
    right_branch = conv_block(right_branch, 48, 3)
    right_branch = conv_block(right_branch, 64, 3)

    concat = Concatenate()([left_branch, middle_branch, right_branch])
    branch_out = conv_block(concat, filters, 1, batchNorm=True, activation=False)

    #skip connection
    layer = Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape,
        arguments={'scale': scale}
        )([layer, branch_out])

    if activation is not None:
        layer = Activation(activation)(layer)
    return layer

def reduction_block(layer, left_filters, middle_filters):
    left_branch = conv_block(layer, left_filters, 3, strides=2, padding='valid')
    middle_branch = conv_block(layer, int(2*middle_filters/3), 1)
    middle_branch = conv_block(middle_branch, int(2*middle_filters/3), 3)
    middle_branch = conv_block(middle_branch, middle_filters, 3, strides=2, padding='valid')
    right_branch = MaxPooling2D(3, strides=2, padding='valid')(layer)
    layer = Concatenate()([left_branch, middle_branch, right_branch])
    return layer

# Inception Module + Reduction
def TslNetV7(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = ZeroPadding2D(padding=(3, 3))(visible)
    layer = Conv2D(64, (7,7), padding='valid', kernel_initializer='he_normal', strides=(2,2), activation='relu')(layer)
    layer = MaxPooling2D((3,3), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = reduction_block(layer, 64, 64)
    layer = inception_resnet_block(layer, 256, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 256, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 256, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 256, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 256, output_shape=(18,18), scale=0.2, activation='relu')
    layer = reduction_block(layer, 128, 128)
    layer = inception_resnet_block(layer, 512, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 512, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 512, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 512, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 512, output_shape=(8,8), scale=0.2, activation='relu')
    layer = reduction_block(layer, 256, 256)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model

#more filters + more block + global average pooling
def TslNetV8(input_shape, no_of_class):
    visible = Input(shape=input_shape)
    layer = ZeroPadding2D(padding=(3, 3))(visible)
    layer = Conv2D(64, (7,7), padding='valid', kernel_initializer='he_normal', strides=(2,2), activation='relu')(layer)
    layer = MaxPooling2D((3,3), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = Conv2D(128, (3,3), padding='same', activation='relu')(layer)
    layer = MaxPooling2D((2,2), strides=(2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 128, output_shape=(37,37), scale=0.2, activation='relu')
    layer = reduction_block(layer, 128, 128)
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 384, output_shape=(18,18), scale=0.2, activation='relu')
    layer = reduction_block(layer, 384, 384)
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = inception_resnet_block(layer, 1152, output_shape=(8,8), scale=0.2, activation='relu')
    layer = conv_block(layer, 2048, 1)
    layer = GlobalAveragePooling2D(name='avg_pool')(layer)
    layer = Dense(no_of_class, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    return model