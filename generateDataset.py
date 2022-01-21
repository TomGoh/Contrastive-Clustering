import os
import pandas as pd
import shutil

if __name__ == '__main__':
    train_files=pd.read_csv('datasets/mini-ImageNet/train.csv')
    test_files=pd.read_csv('datasets/mini-ImageNet/test.csv')
    valid_files=pd.read_csv('datasets/mini-ImageNet/val.csv')

    train_file_names=train_files['filename']
    train_file_labels=train_files['label']
    test_file_names = test_files['filename']
    test_file_labels = test_files['label']
    valid_file_names = valid_files['filename']
    valid_file_labels = valid_files['label']
    print(len(train_file_labels),len(train_file_names),len(test_file_labels),len(test_file_names),len(valid_file_labels),len(valid_file_names))

    for i in train_file_labels.unique():
        os.mkdir('./datasets/mini-ImageNet/all/'+i)

    for i in test_file_labels.unique():
        os.mkdir('./datasets/mini-ImageNet/all/' + i)

    for i in valid_file_labels.unique():
        os.mkdir('./datasets/mini-ImageNet/all/'+i)

    root_path="C:\\Users\\Tom-G\\Desktop\\images\\"

    for name,label in zip(train_file_names,train_file_labels):
        file=os.path.join(root_path,name)
        if os.path.isfile(file):
            path=os.path.join('datasets\\mini-ImageNet\\all',label)
            shutil.copyfile(file,os.path.join(path,name))

    for name,label in zip(test_file_names,test_file_labels):
        file=os.path.join(root_path,name)
        if os.path.isfile(file):
            path=os.path.join('datasets\\mini-ImageNet\\all',label)
            shutil.copyfile(file,os.path.join(path,name))

    for name,label in zip(valid_file_names,valid_file_labels):
        file=os.path.join(root_path,name)
        if os.path.isfile(file):
            path=os.path.join('datasets\\mini-ImageNet\\all',label)
            shutil.copyfile(file,os.path.join(path,name))


