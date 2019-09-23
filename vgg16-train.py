import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt



from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
import os


train_folder = '../uncropped/train'
val_folder = '../uncropped/validate'

output_folder = 'output'

batch_size = 32
IMG_SIZE = 224
#seed = 7
#np.random.seed(seed)
modelname = 'vgg-16'

def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

# .............................................................................
datagen = ImageDataGenerator()


train_it = datagen.flow_from_directory(train_folder, shuffle=True, target_size=(IMG_SIZE,IMG_SIZE), class_mode='categorical', batch_size=batch_size)
val_it = datagen.flow_from_directory(val_folder, shuffle=False, target_size=(IMG_SIZE,IMG_SIZE), class_mode='categorical', batch_size=batch_size)


filepath        = os.path.join(output_folder, modelname + ".hdf5")
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(os.path.join(output_folder, modelname +'.csv'))
callbacks_list  = [checkpoint,csv_logger]

def vgg_block(layer_in, n_filters, n_conv):
    for _ in range(n_conv):
          layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
    layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
    return layer_in
    
def createModel():
    visible = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    layer = vgg_block(visible,64,2)
    layer = vgg_block(layer, 128, 2)
    layer = vgg_block(layer, 256, 3)
    layer = vgg_block(layer, 512, 3)
    layer = vgg_block(layer, 512, 3)
    layer = Flatten()(layer)
    layer = Dense(4096, activation='relu')(layer)
    layer = Dense(4096, activation='relu')(layer)
    layer = Dense(4, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
  
  # define model
model = createModel()
model.summary()
from tensorflow.keras.utils import plot_model
model_file = os.path.join(output_folder, modelname + "_model.png")
plot_model(model, 
           to_file=model_file, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')

# fit model
model.fit_generator(train_it, validation_data=val_it,epochs=50,callbacks=callbacks_list)