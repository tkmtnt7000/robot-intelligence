import os.path

import pickle

import numpy as np
from dataset.mnist import load_mnist

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist_noise_"

width=28
height=28
data_size = width*height

noise_per_list = [i for i in range(1, 10)]
for noise_per in noise_per_list:
    print(noise_per)
    noise_num = int(0.01*noise_per*data_size)

    noise_X = x_train.copy()

    for each_X in noise_X:
        noise_mask = np.random.randint(0, data_size, noise_num)

        for noise_idx in noise_mask:
            each_X[noise_idx] = np.random.random()

def init_noise_mnist():
    raw_data = load_mnist(normalize=True, flatten=True, one_hot_label=True)



def load_mnist_noise(noise_percent):
    
    with open(save_file+str(noise_percent)+"%.pkl", 'rb') as f:
        dataset = pickle.load(f)

    

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':