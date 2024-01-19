This repository contains srcipts for Solar Radio Bursts (SRB) automatic detection using a Convolutional Neural Network (CNN) algorithm.

To effectively design an AI model for SRB detection, the following workflow specifies each step for a successful development:

![workflow_3](https://github.com/ivanarturo96/SRB_CNN_detection/assets/128187508/5f9e225e-d0ae-4aaf-bd2c-7aad2547abed)

# DATASET AND PRE-PROCESSING:
Curation of dataset, as well as the preprocess can be seen inside the folder "dataset".

Data are collected from the e-CALLISTO server, which serves as a comprehensive hub of daily FITS files encoding spectrograms from global stations. Data acquisition is done with file "download_data.ipynb", final dataset preprocess + allocation is done with file "dataset.ipynb"

# MODEL'S FINE TUNNING:
In folder "model", there's a srcipt called "model.py" that shows a grid search, varying hyperparameters in order to find the best model configuration, using VGG16 CNN architecture.

Finally, results can be checked in file "grid_search.csv"
