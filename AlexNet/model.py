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
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import resize
import random

physical_devices=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
SEED = 7
tf.keras.utils.set_random_seed(SEED) #Establece la semilla de Python, Numpy y TensorFlow. Para que los números aleatorios sean los mismos cada vez que se corre
tf.config.experimental.enable_op_determinism() #Para que con los mismos inputs, se tengan los mismos outputs cada vez que se corre

acc_test = []
list_tp = []
list_tn = []
list_fp = []
list_fn = []
list_recall = []
list_precision = []
list_f1 = []
list_fnr = []
list_fpr = []
acc_train = []
loss_train = []
acc_val = []
loss_val = []
list_ep = []
list_ep_time = []

for img_size in [112,224,512]:
    for batch_size in [4, 8, 16]:
        for initial_learning_rate in [1e-5, 1e-4, 1e-3]:
            image_size = (img_size,img_size)
            
            train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
                "../../datasets/dataset_5/training/",
                validation_split=0.2,
                subset="both",
                seed=1337,
                image_size=image_size,
                batch_size=batch_size)
            
            test_ds = tf.keras.utils.image_dataset_from_directory(
            "../../datasets/dataset_5/test/",
            seed=1337,
            #shuffle=False,
            image_size=image_size,
            batch_size=batch_size)
            
            ##################### Pre-trained model #####################
            
            model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(img_size,img_size,3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
            ])
            
            ##################### Apply data augmentation #####################
            
            data_augmentation = tf.keras.Sequential([layers.RandomTranslation(height_factor=0,width_factor=0.1,fill_mode='wrap'),
                                                     layers.GaussianNoise(3),
                                                     #layers.RandomBrightness((-0.1,0.1))
                                                    ])
            
            train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            
            # If it's used randomcropping, it requires a resize back to the prior image size
            #train_ds = train_ds.map(lambda x,y: (tf.image.resize(x, size=[img_size,img_size]),y), num_parallel_calls=tf.data.AUTOTUNE)
            
            # Prefetching samples in GPU memory helps maximize GPU utilization.
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
            
            epochs = 500 #500
            config = f"{img_size}_{batch_size}_{initial_learning_rate}"
            #lr_scheduler = keras.callbacks.LearningRateScheduler(lr_step_decay)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto')
            filepath="training_files/best_model_{}.h5".format(config)
            
            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            callbacks = [
                checkpoint,early_stopping
            ]#lr_scheduler early_stopping
            model.compile(
                optimizer=keras.optimizers.experimental.SGD(initial_learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy","TruePositives","TrueNegatives","FalsePositives", "FalseNegatives","Recall","Precision"]
            )
            
            start_time = time.monotonic()
            
            
            hist=model.fit(
                train_ds,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=val_ds
            )
            
            end_time = time.monotonic()
            print(timedelta(seconds=end_time - start_time))

            nro_epocas = len(hist.epoch)
            epoch_time = (end_time - start_time)/nro_epocas
            train_loss=hist.history['loss'][-1]
            val_loss=hist.history['val_loss'][-1]
            train_acc=hist.history['accuracy'][-1]
            val_acc=hist.history['val_accuracy'][-1]
            
            #### Saves training data of each iteration
            filepath="training_files/model_{}.h5".format(config)
            model.save_weights(filepath) # verificar load_best_model
            model_json = model.to_json()
            with open("training_files/model_{}.json".format(config), "w") as json_file:
                json_file.write(model_json)
            hist_df = pd.DataFrame(hist.history)
            # or save to csv: 
            hist_csv_file = 'training_files/history_{}.csv'.format(config)
            with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)

            ######################## Important data are saved ########################

            result=model.evaluate(x=test_ds,verbose=0,return_dict=True)
            recall=result["recall"]
            precision=result["precision"]
            tp=result["true_positives"]
            tn=result["true_negatives"]
            fp=result["false_positives"]
            fn=result["false_negatives"]
            f1=2*((recall*precision)/(recall+precision))
            fnr = fn/(tp+fn)
            fpr = fp/(fp+tn)
            acc_test.append(round(result["accuracy"]*100,2))
            list_tp.append(tp)
            list_tn.append(tn)
            list_fp.append(fp)
            list_fn.append(fn)
            list_recall.append(round(recall*100,2))
            list_precision.append(round(precision*100,2))
            list_f1.append(round(f1*100,2))
            list_fnr.append(round(fnr*100,2))
            list_fpr.append(round(fpr*100,2))
            acc_train.append(round(train_acc*100,2))
            loss_train.append(train_loss)
            acc_val.append(round(val_acc*100,2))
            loss_val.append(val_loss)
            list_ep.append(nro_epocas)
            list_ep_time.append(epoch_time)
df = pd.DataFrame({"IS":[112]*9+[224]*9+[512]*9 ,"BS": ([4]*3+[8]*3+[16]*3)*3,"LR": [1e-5, 1e-4, 1e-3]*9,"Test Accuracy": acc_test,"TP": list_tp,"TN": list_tn,"FP":list_fp ,"FN":list_fn ,"Recall":list_recall ,"Precision":list_precision ,"F1 Score":list_f1 , "FNR":list_fnr ,"FPR":list_fpr ,"Train Accuracy":acc_train ,"Train Loss":loss_train ,"Val Accuracy":acc_val ,"Val Loss":loss_val ,"Epochs":list_ep ,"Time per epoch":list_ep_time })

grid_csv_file = 'grid_search.csv'
with open(grid_csv_file, mode='w') as f:
    df.to_csv(f)