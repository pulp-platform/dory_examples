import numpy as np
import util
from util import div_and_ceil, number_of_subtiles
from util import number_of_tiles

CONV_OVERLAP = 2
SUBTILE_SHAPE = (5, 5, 16)
OUT_BUFFER_SHAPE = (32, 3, 3)

X_SHAPE, Y_SHAPE, W_SHAPE, X_TILE_SHAPE, Y_TILE_SHAPE, W_TILE_SHAPE = util.loadtiling("logs/Tiling_profiling.log")
input = util.loadtxt("MiniNet/input.txt", X_SHAPE)
acc_input = util.loadlog("logs/inputs_cleaned.log", SUBTILE_SHAPE)

w_ko, _, _, _ = W_SHAPE
w_tile_ko, _, _, _ = W_TILE_SHAPE

n_tile_ko = div_and_ceil(w_ko, w_tile_ko)
n_subtile_ko = div_and_ceil(w_tile_ko, OUT_BUFFER_SHAPE[0])

n_subtiles = n_tile_ko * n_subtile_ko * number_of_subtiles(X_SHAPE, X_TILE_SHAPE, SUBTILE_SHAPE, conv_overlap=2)

print(f'Number of subtiles: {n_subtiles}')

if (n_subtiles != acc_input.shape[0]):
    print(f'n_subtiles({n_subtiles}) != acc_input.shape[0]({acc_input.shape[0]})')

def process_tile(tile, subtile_shape, conv_overlap, n_subtile_ko):
    subtile_h, subtile_w, subtile_k = subtile_shape
    stride_h = subtile_h - conv_overlap
    stride_w = subtile_w - conv_overlap
    stride_k = subtile_k
    n_subtile_h, n_subtile_w, n_subtile_ki = number_of_tiles(tile.shape, subtile_shape)
    acc_data = np.zeros((n_subtile_ko, n_subtile_h, n_subtile_w, n_subtile_ki) + subtile_shape)
    for i in range(n_subtile_ko):
        for h in range(n_subtile_h):
            for w in range(n_subtile_w):
                for k in range(n_subtile_ki):
                    subtile = tile[h*stride_h:h*stride_h + subtile_h, w*stride_w:w*stride_w + subtile_w, k*stride_k:k*stride_k + subtile_k]
                    size_h, size_w, size_k = subtile.shape
                    acc_data[i, h, w, k, :size_h, :size_w, :size_k] = subtile
    return acc_data
    
acc_golden = np.empty((0,) + SUBTILE_SHAPE, dtype=np.uint8)

in_k, in_h, in_w = X_SHAPE
tile_h, tile_w, tile_k = X_TILE_SHAPE
for i in range(n_tile_ko):
    for j in range(0, in_h-CONV_OVERLAP, tile_h-CONV_OVERLAP):
        for k in range(0, in_w-CONV_OVERLAP, tile_w-CONV_OVERLAP):
            for l in range(0, in_k, tile_k):
                input_tile = input[j:j+tile_h, k:k+tile_w, l:l+tile_k]
                processed_tile = process_tile(input_tile, SUBTILE_SHAPE, CONV_OVERLAP, n_subtile_ko).reshape((-1, ) + SUBTILE_SHAPE)
                acc_golden = np.append(acc_golden, processed_tile, axis=0)

if np.array_equal(acc_input, acc_golden):
    print("OK! Inputs are the same.")
else:
    print("ERROR! Inputs are not the same.")
    print("Golden:")
    print(acc_golden[np.equal(acc_golden, acc_input)])
    print("From log:")
    print(acc_input[np.equal(acc_golden, acc_input)])
