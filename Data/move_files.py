"""
Modified by: vkyprmr
Last modified on: 2020-10-22, Do., 12:21:39
"""

# Imports
import os
import glob
import shutil
import random

cdir = os.getcwd()
train_dir = cdir + '/train/'
test_dir = cdir + '/test/'
val_dir = cdir + '/validation/'


# Last 'X' files
def move_files(num_files, shuffle=False, multi_dir=False):
    """
    Args:
        multi_dir: does the directory follow a tree structure (default: False)
        num_files: number of files to shift
        shuffle: whether to randomly shift files
                 default: False
    Returns:
        Moves files to the specified directory
    """
    if shuffle:
        print('Shuffling...')
        if multi_dir:
            print('Looking in sub-dirs...')
            dirs = os.listdir(train_dir)
            total_files = []
            for d in dirs:
                files = glob.glob(f'{train_dir}{d}/*.jpg')
                i = [i for i in range(len(files))]
                ids = random.sample(i, num_files)
                ids.sort()
                f2m = [files[i] for i in ids]
                mv_path = f'{val_dir}{d}/'
                for i in f2m:
                    shutil.move(i, val_dir + d + '/')
                    print(f'Moved {i} to {mv_path}')
                total_files.append(f2m)
                print(f'Moved {len(f2m)} files')
            print(f'Moved {len(total_files)}')
        else:
            print('Looking in files...')
            total_files = []
            files = glob.glob(f'{train_dir}/*.jpg')
            i = [i for i in range(len(files))]
            ids = random.sample(i, num_files)
            ids.sort()
            f2m = [files[i] for i in ids]
            mv_path = f'{val_dir}/'
            for i in f2m:
                shutil.move(i, val_dir + '/')
                print(f'Moved {i} to {mv_path}')
            total_files.append(f2m)
            print(f'Moved {len(f2m)} files')
        
    else:
        print(f'Selecting the last {num_files}...')
        if multi_dir:
            print('Looking through directories')
            dirs = os.listdir(train_dir)
            total_files = []
            for d in dirs:
                files = glob.glob(f'{train_dir}{d}/*.jpg')
                f2m = files[-num_files:]
                mv_path = f'{val_dir}{d}/'
                for i in f2m:
                    shutil.move(i, val_dir + d + '/')
                    print(f'Moved {i} to {mv_path}')
                total_files.append(f2m)
                print(f'Moved {len(f2m)} files')
            print(f'Moved {len(total_files)}')
        else:
            print('Looking through files')
            total_files = []
            files = glob.glob(f'{train_dir}/*.jpg')
            f2m = files[-num_files:]
            mv_path = f'{val_dir}/'
            for i in f2m:
                shutil.move(i, val_dir + '/')
                print(f'Moved {i} to {mv_path}')
            total_files.append(f2m)
            print(f'Moved {len(f2m)} files')

# move_files()
