## Setting for feature
SR = 48000  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
MFCC_dim = 13  # the MFCC dimension

## Setting for training
fold_num = 5
seed = 2022

## Setting for directory
DIR_DATA = "./sounddr_data/"
OUTPUT_DIR = DIR_DATA + 'output/'

import os
os.makedirs(OUTPUT_DIR, exist_ok = True)