import csv
import math
import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from TslModel import TslNetV2

test_folder = '../cropped/test'
output_folder = 'output-tsl'
classes = ["cendol", "ice kachang", "tauhuay", "tausuan"]
batch_size = 32
IMG_SIZE = 150
seed = 7
np.random.seed(seed)
modelname = 'TslNetV2'
weight_file = os.path.join(output_folder, modelname + ".hdf5")
learning_rate_file = os.path.join(output_folder, modelname +'.csv')
plt_file = os.path.join(output_folder, modelname + '_plot.png')
prediction_csv = os.path.join(output_folder, modelname + "_prediction.csv")

plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

# .............................................................................
# data preparation
datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory(test_folder, target_size=(IMG_SIZE,IMG_SIZE), class_mode='categorical', batch_size=batch_size, shuffle=False)

# define model
modelGo = TslNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), no_of_class=4)
modelGo.load_weights(weight_file)
modelGo.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

#prediction
predict = modelGo.predict_generator(test_it)


#get actual labels
tsLbl = []
test_it.reset()
count = 0
number_of_examples = len(test_it.filenames)
number_of_generator_calls = math.ceil(number_of_examples / (1.0 * batch_size)) 
files = []
for i in range(0,int(number_of_generator_calls)): 
  fn = test_it.filenames[i*batch_size:(i+1)*batch_size]
  lbls = test_it[i][1]
  files.extend(np.array(fn))
  tsLbl.extend(np.array(lbls))

tsLbl = np.asarray(tsLbl, dtype=np.float32)

#evaluation
predout = np.argmax(predict, axis=1)
testout = np.argmax(tsLbl, axis = 1)
labelname = classes

final = []
for i in range(0, len(files)):
  tup = [files[i], testout[i], predout[i]]
  final.append(tup)

#write prediction csv
with open(prediction_csv, 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["filename", "actual", "predicted"])
    for r in final:
      writer.writerow(r)

testScores = metrics.accuracy_score(testout,predout)
confusion = metrics.confusion_matrix(testout, predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)

records     = pd.read_csv(learning_rate_file)
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0,0.20,0.40,0.60,0.80,1.00])
plt.title('Loss value',fontsize=12)

ax          = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9,1.0])
plt.title('Accuracy',fontsize=12)

plt.show()
plt.savefig(plt_file)
