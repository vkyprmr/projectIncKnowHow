import os
import glob
import shutil

cdir = os.getcwd()
train_dir = cdir + '/train/'
val_dir = cdir + '/validation/'


def move_files():
    dirs = os.listdir(train_dir)
    total_files = []
    for d in dirs:
        files = glob.glob(f'{train_dir}{d}/*.jpg')
        f2m = files[-450:]
        mv_path = f'{val_dir}{d}/'
        for i in f2m:
            shutil.move(i, val_dir + d + '/')
            print(f'Moved {i} to {mv_path}')
        total_files.append(f2m)
        print(f'Moved {len(f2m)} files')
    print(f'Moved {len(total_files)}')


move_files()
