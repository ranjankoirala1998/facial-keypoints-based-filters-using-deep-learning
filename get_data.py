''' 
-> this is a simple module and contains different functions that we have discussed in the "data_augmentation.ipynb"
-> this module is build to be used in other program to easily access the training and testing data
-> this module also has a "plot_sample" function which is used to plot sample images from either test set or train set
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('datasets/train.csv')
train['Image'] = train['Image'].apply(lambda x : np.reshape(np.array(x.split(' '), dtype=int), (96, 96, 1)))

train = pd.read_csv('datasets/test.csv')
test['Image'] = test['Image'].apply(lambda x : np.reshape(np.array(x.split(' '), dtype=int), (96, 96, 1)))

def get_train() :
    def load_images(dataframe):
        images = []
        for idx, image in dataframe.iterrows():
            images.append(image['Image'])
        images = np.array(images)/255. # normalized the gray-value in the range 0-1
        return images

    def load_keypoints(dataframe):
        temp = dataframe.drop('Image', axis = 1) # dropping everything from the 'Image' column; doing so, only the keypoints remains
        keypoints = []
        for idx, keypoint in temp.iterrows():
            keypoints.append(keypoint)
        keypoints = np.array(keypoints, dtype = 'float')
        return keypoints

    def reflection(images, keypoints):
        reflected_images = np.flip(images, axis=2) ## this will perform the horizontal flip on all the images in train_images

        reflected_keypoints = []
        for idx, keypoint in enumerate(keypoints):
            reflected_keypoints.append([96. - coordinate if idx%2 == 0 else coordinate for idx, coordinate in enumerate(keypoint)]) ## this will perform the reflection transformation on the keyponts (some mathematics is involved)
        return reflected_images, reflected_keypoints
    
    def brightness_alteration(images):
        alpha = 0.72  # Simple contrast control
        beta = -0.18  # Simple brightness control

        dimmed_images = np.clip(images * 0.72 - 0.18, 0.0, 1.0)

        return dimmed_images
    
    images = load_images(train)
    keypoints = load_keypoints(train)
    
    reflected_images, reflected_keypoints = reflection(images, keypoints)
    more_images = np.concatenate((images, reflected_images))
    more_keypoints = np.concatenate((keypoints, reflected_keypoints))
    
    dimmed_images = brightness_alteration(more_images)
    final_images = np.concatenate((more_images, dimmed_images))
    final_keypoints = np.concatenate((more_keypoints, more_keypoints))

    return final_images, final_keypoints

def get_test(dataframe):
    test_images = []
    for idx, image in dataframe.iterrows():
        images.append(image['Image'])
    test_images = np.array(images)/255. # normalized the gray-value in the range 0-1
    return test_images

def plot_sample(image, keypoint, axis, title):
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='o', s=10)
    axis.set_title(title)
    plt.show()





