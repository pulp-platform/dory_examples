import numpy as np
import re

CONV_OVERLAP = 2
SUBTILE_SHAPE = (5, 5, 16)
OUT_BUFFER_SHAPE = (32, 3, 3)

def div_and_ceil(a, b):
    return ((a-1) // b) + 1

def loadtxt(filename, in_shape):
    with open(filename, 'r') as infile:
        infile.readline() # skip first line
        return np.array([line[0:-2] for line in infile], dtype=np.uint8).reshape(in_shape)

def loadlog(filename, buffer_shape):
    with open(filename, 'r') as infile:
        shape = (-1,) + buffer_shape
        return np.array([(line[0:-2] if line[-2] == ',' else line[0:-1]).split(',') for line in infile], dtype=np.uint8).reshape(shape)

def tile_dim(in_shape, tile_shape, conv_overlap = 2):
    h_in, w_in, k_in = in_shape
    h_tile, w_tile, k_tile = tile_shape
    n_tiles_h = div_and_ceil(h_in - conv_overlap, (h_tile - conv_overlap))
    n_tiles_w = div_and_ceil(w_in - conv_overlap, (w_tile - conv_overlap))
    n_tiles_k = div_and_ceil(k_in, k_tile)
    return (n_tiles_h, n_tiles_w, n_tiles_k)

def loadtiling(tiling_log, layer=0):
    matched_lines = []
    with open(tiling_log, 'r') as infile:
        for line in infile:
            if re.search(r'L2 size', line):
                matched_lines.append(line)
    
    shapes = re.findall(r'\[(.*?)\]', matched_lines[layer])
    chw_x, chw_y, kokihw_w = [tuple([int(i) for i in s.split('x')]) for s in shapes]

    matched_lines = []
    with open(tiling_log, 'r') as infile:
        for line in infile:
            if re.search(r'tiles L2-L1', line):
                matched_lines.append(line)
    
    shapes = re.findall(r'\[(.*?)\]', matched_lines[layer])
    chw_x_tile, chw_y_tile, kokihw_w_tile = [tuple([int(i) for i in s.split('x')]) for s in shapes]
    def chw2hwc(s):
        return (s[1], s[2], s[0])
    return (chw2hwc(chw_x), chw2hwc(chw_y), kokihw_w, chw2hwc(chw_x_tile), chw2hwc(chw_y_tile), kokihw_w_tile)


X_SHAPE, Y_SHAPE, W_SHAPE, X_TILE_SHAPE, Y_TILE_SHAPE, W_TILE_SHAPE = loadtiling("logs/Tiling_profiling.log")
input = loadtxt("MiniNet/input.txt", X_SHAPE)
acc_input = loadlog("logs/inputs_cleaned.log", SUBTILE_SHAPE)

w_ko, _, _, _ = W_SHAPE
w_tile_ko, _, _, _ = W_TILE_SHAPE

n_tiles_ko = div_and_ceil(w_ko, w_tile_ko)
n_tiles_h, n_tiles_w, n_tiles_ki = tile_dim(input.shape, X_TILE_SHAPE)
n_tiles = n_tiles_ko * n_tiles_h * n_tiles_w * n_tiles_ki

n_body_subtile_h, n_body_subtile_w, n_body_subtile_ki = tile_dim(X_TILE_SHAPE, SUBTILE_SHAPE)

in_h, in_w, in_k = input.shape
tile_h, tile_w, tile_k = X_TILE_SHAPE

h_rem_tile = in_h - (n_tiles_h - 1) * (tile_h - CONV_OVERLAP)
w_rem_tile = in_w - (n_tiles_w - 1) * (tile_w - CONV_OVERLAP)
k_rem_tile = in_k - (n_tiles_ki - 1) *  tile_k

REM_TILE_SHAPE = (h_rem_tile, w_rem_tile, k_rem_tile)

n_rem_subtile_h, n_rem_subtile_w, n_rem_subtile_k = tile_dim(REM_TILE_SHAPE, SUBTILE_SHAPE)

n_subtile_h = n_body_subtile_h * (n_tiles_h - 1) + n_rem_subtile_h
n_subtile_w = n_body_subtile_w * (n_tiles_w - 1) + n_rem_subtile_w
n_subtile_ki = n_body_subtile_ki * (n_tiles_ki - 1) + n_rem_subtile_k
n_subtile_ko = n_tiles_ko * div_and_ceil(w_tile_ko, OUT_BUFFER_SHAPE[0])

n_subtiles = n_subtile_h * n_subtile_w * n_subtile_ki * n_subtile_ko

if (n_subtiles != acc_input.shape[0]):
    print(f'n_subtiles({n_subtiles}) != acc_input.shape[0]({acc_input.shape[0]})')

acc_golden = np.zeros_like(acc_input)

def process_tile(tile, n_subtile_ko):
    h, w, ki = tile.shape
    n_subtile_h, n_subtile_w, n_subtile_ki = tile_dim(tile.shape, SUBTILE_SHAPE)
    acc_data = np.zeros((n_subtile_ko, n_subtile_h, n_subtile_w, n_subtile_ki) + SUBTILE_SHAPE)
    for i in range(n_subtile_ko):
        for j in range(n_subtile_h):
            for k in range(n_subtile_w):
                for l in range(n_subtile_ki):
                    size_h = min(h - j*3, 5)
                    size_w = min(w - k*3, 5)
                    size_k = min(ki - l*16, 16)
                    acc_data[i, j, k, l, :size_h, :size_w, :size_k] = tile[j*3:j*3+5, k*3:k*3+5, l*16:l*16+16]
    return acc_data
    
acc_golden = process_tile(input, n_subtile_ko)

acc_golden = np.empty((0,) + SUBTILE_SHAPE, dtype=np.uint8)

in_k, in_h, in_w = X_SHAPE
tile_h, tile_w, tile_k = X_TILE_SHAPE
for i in range(n_tiles_ko):
    for j in range(0, in_h-CONV_OVERLAP, tile_h-CONV_OVERLAP):
        for k in range(0, in_w-CONV_OVERLAP, tile_w-CONV_OVERLAP):
            for l in range(0, in_k-CONV_OVERLAP, tile_k-CONV_OVERLAP):
                acc_golden = np.append(acc_golden, process_tile(input[j:j+tile_h, k:k+tile_w, l:l+tile_k], n_subtile_ko).reshape((-1, ) + SUBTILE_SHAPE), axis=0)

if np.array_equal(acc_input, acc_golden):
    print("OK! Inputs are the same.")
else:
    print("ERROR! Inputs are not the same.")
    print("Golden:")
    print(acc_golden[np.equal(acc_golden, acc_input)])
    print("From log:")
    print(acc_input[np.equal(acc_golden, acc_input)])
