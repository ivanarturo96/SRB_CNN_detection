#import sys 
#sys.path.append('../../env/lib/python3.10/site-packages/') 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import time
from datetime import timedelta
from tensorflow.keras.models import load_model,model_from_json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import optuna
from optuna.samplers import TPESampler
from pickle import dump,load
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

#for lr in [1e-8,1e-7,1e-6]:
#    for bs in [4,8,16]:
#iris = sklearn.datasets.load_iris()
physical_devices=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
SEED = 7
tf.keras.utils.set_random_seed(SEED) #Establece la semilla de Python, Numpy y TensorFlow. Para que los números aleatorios sean los mismos cada vez que se corre
tf.config.experimental.enable_op_determinism() #Para que con los mismos inputs, se tengan los mismos outputs cada vez que se corre'''
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

initial_learning_rate=1e-6
batch_size = 4
image_size = (224,224)

train_ds = tf.keras.utils.image_dataset_from_directory(
"../datasets/dataset_5/training/",
seed=1337,
#color_mode='grayscale',
image_size=image_size,
batch_size=1)

test_ds = tf.keras.utils.image_dataset_from_directory(
"../datasets/dataset_5/test/",
seed=1337, #1337
image_size=image_size,
#color_mode='grayscale',
#shuffle=False,
batch_size=batch_size)

kf = KFold(n_splits=5,shuffle=True)

def dataset_to_arrays(ds):
    list_ds = list(ds)
    images = []
    labels = []
    
    # Loop through the batches
    for i in range(len(list_ds)):
        img = list_ds[i][0][0].numpy()
        lab = list_ds[i][1][0].numpy()  # Extract the element and convert to a NumPy array
        labels.append(lab)  # Append the element to the list
        images.append(img)  # Append the element to the list
        
    return images, labels

trainX, trainY = dataset_to_arrays(train_ds)

trainX = np.array(trainX)
trainY = np.array(trainY)

for train_index, val_index in kf.split(trainX):
    print("TRAIN: ", len(train_index), "VAL: ", len(val_index))
    trainX_k, valX_k = trainX[train_index], trainX[val_index]
    trainY_k, valY_k= trainY[train_index], trainY[val_index]

    ##################### Pre-trained model #####################
    vgg_pretrained_model = VGG16(weights="imagenet", 
                                 include_top= False,
                                 input_tensor=keras.Input(image_size + (3,)))
    new_model = vgg_pretrained_model.output
    new_model = layers.AveragePooling2D(pool_size=(3, 3))(new_model)
    new_model = layers.Flatten(name="flatten")(new_model)
    new_model = layers.Dense(64, activation="relu")(new_model)
    new_model = layers.Dropout(0.5)(new_model)
    new_model = layers.Dense(1, activation="sigmoid")(new_model)
    
    model = keras.Model(inputs=vgg_pretrained_model.input, outputs=new_model)

    # Create tensor flow datasets to apply prefetch
    '''train_ds = tf.data.Dataset.from_tensor_slices((trainX_k, trainY_k))
    val_ds = tf.data.Dataset.from_tensor_slices((valX_k, valY_k))

    # Shuffle and batch dataset
    #SHUFFLE_BUFFER_SIZE = 1378
    #train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Apply prefetching
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)'''
    
    epochs = 500 #500
    #lr_scheduler = keras.callbacks.LearningRateScheduler(lr_step_decay)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto')
    filepath="training_files/best_model_{}.h5".format(val_index[0])
    
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks = [
        checkpoint,early_stopping
    ]#lr_scheduler early_stopping
    model.compile(
        optimizer=keras.optimizers.Adam(initial_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy","TruePositives","TrueNegatives","FalsePositives", "FalseNegatives","Recall","Precision"]
    )
    
    start_time = time.monotonic()
    
    
    hist=model.fit(
        x= trainX_k,
        y = trainY_k,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(valX_k, valY_k)
    )
    
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    #### Saves training data of each iteration
    filepath="training_files/model_{}.h5".format(val_index[0])
    model.save_weights(filepath) # verificar load_best_model
    model_json = model.to_json()
    with open("training_files/model_{}.json".format(val_index[0]), "w") as json_file:
        json_file.write(model_json)
    hist_df = pd.DataFrame(hist.history)
    # or save to csv: 
    hist_csv_file = 'training_files/history_{}.csv'.format(val_index[0])
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)