import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import plot_model
from TslModel import TslNetV7

# ----- declare some constant
train_folder = '../cropped/train'
val_folder = '../cropped/validate'
output_folder = 'output'
classes = ["cendol", "ice kachang", "tauhuay", "tausuan"]
batch_size = 32
train_epoch = 50
IMG_SIZE = 300
seed = 7
np.random.seed(seed)
modelname = 'TslNetV7'
weight_file = os.path.join(output_folder, modelname + ".hdf5")
model_file = os.path.join(output_folder, modelname + "_model.png")

#data preparation
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory(train_folder, shuffle=True, target_size=(IMG_SIZE,IMG_SIZE), class_mode='categorical', batch_size=batch_size)
val_it = datagen.flow_from_directory(val_folder, shuffle=True, target_size=(IMG_SIZE,IMG_SIZE), class_mode='categorical', batch_size=batch_size)

#set up callback list
checkpoint      = ModelCheckpoint(weight_file, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')
                            # Log the epoch detail into csv
csv_logger      = CSVLogger(os.path.join(output_folder, modelname +'.csv'))
callbacks_list  = [checkpoint,csv_logger]
  
#define model
model = TslNetV7(input_shape=(IMG_SIZE,IMG_SIZE,3), no_of_class=4)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, 
           to_file=model_file, 
           show_shapes=True, 
           show_layer_names=False,
           rankdir='TB')

# fit model
model.fit_generator(train_it, validation_data=val_it,epochs=train_epoch,callbacks=callbacks_list)