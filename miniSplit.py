import os
import shutil
from os.path import join
import random
import numpy as np

if __name__ == '__main__':
    root = 'datasets/mini-ImageNet/10'
    # os.mkdir('datasets/mini-ImageNet/10/train')
    # os.mkdir('datasets/mini-ImageNet/10/valid')
    # os.mkdir('datasets/mini-ImageNet/10/test')

    # train:valid:test = 440:80:80

    dirs = os.listdir(os.path.join(root, 'images'))
    print(len(dirs))
    for category in dirs:
        os.mkdir(join(join(root,'test'),category))
        os.mkdir(join(join(root, 'train'), category))
        images = os.listdir(os.path.join(os.path.join(root, 'images'), category))
        print(len(images))
        index = np.arange(0, 600)
        np.random.shuffle(index)
        for i in index[0:100]:
            shutil.copyfile(join(join(join(root,'images'),category),images[i]),join(join(join(root,'test'),category),images[i]))
        for i in index[100:]:
            shutil.copyfile(join(join(join(root,'images'),category),images[i]),join(join(join(root,'train'),category),images[i]))


