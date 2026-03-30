# config.py
import os

# Paths
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
ANNOTATION_PATH = os.path.join(DATA_DIR, 'annotations')

# Image processing parameters
MASK_LEFT = 30  # 30% width
MASK_RIGHT = 70  # 70% width
NUM_SEGMENTS = 16
THRESHOLD_PERCENT = 10
GAMMA = 2
KERNEL_SIZE = 3
SHARPEN_KERNEL = (-1, -1, -1, -1, 8, -1, -1, -1, -1)  # 3x3 sharpening kernel

LABEL =['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6','T5','T4','T3','T2','T1','C7','C6','C5','C4','C3','C2','C1']
