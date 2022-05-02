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

def chw2hwc(s):
    return (s[1], s[2], s[0])

def hwc2chw(s):
    return (s[2], s[0], s[1])

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
    return (chw2hwc(chw_x), chw2hwc(chw_y), kokihw_w, chw2hwc(chw_x_tile), chw2hwc(chw_y_tile), kokihw_w_tile)


root = "/home/lmacan/pulp/dory/dory_examples/"
X_SHAPE, Y_SHAPE, W_SHAPE, X_TILE_SHAPE, Y_TILE_SHAPE, W_TILE_SHAPE = loadtiling(root + "logs/Tiling_profiling.log")
interm = loadtxt(root + "MiniNet/interm_layer0.txt", Y_SHAPE)
acc_interm = loadlog(root + "logs/interm_cleaned.log", OUT_BUFFER_SHAPE)

print(interm.shape)
print(acc_interm.shape)

def subtile_check(subtile, subtile_acc):
    h, w, k = subtile.shape
    return np.array_equal(subtile, subtile_acc[:h, :w, :k])

def tile_check(tile, tile_acc):
    is_equal = True
    i_subtile = 0

    tile_h, tile_w, tile_k = tile.shape
    _, sub_h, sub_w, sub_k = tile_acc.shape

    n_sub_h = div_and_ceil(tile_h, sub_h)
    n_sub_w = div_and_ceil(tile_w, sub_w)
    n_sub_k = div_and_ceil(tile_k, sub_k)

    tile_acc = tile_acc.reshape(n_sub_k, n_sub_h, n_sub_w, sub_h, sub_w, sub_k)

    for i_k, start_k in zip(range(n_sub_k), range(0, tile_k, sub_k)):
        for i_h, start_h in zip(range(n_sub_h), range(0, tile_h, sub_h)):
            for i_w, start_w in zip(range(n_sub_w), range(0, tile_w, sub_w)):
                subtile = tile[start_h:start_h + sub_h, start_w:start_w + sub_w, start_k:start_k + sub_k]
                subtile_acc = tile_acc[i_k, i_h, i_w]
                print(f"Checking subtile {i_subtile}: ", end="")
                is_equal_subtile = subtile_check(subtile, subtile_acc)
                print("OK!" if is_equal_subtile else "ERROR! Not the same")
                i_subtile += 1
                is_equal = is_equal and is_equal_subtile
    
    return is_equal


out_h, out_w, out_k = Y_SHAPE
tile_h, tile_w, tile_k = Y_TILE_SHAPE
subtile_k, subtile_h, subtile_w = OUT_BUFFER_SHAPE

start_n_sub = 0
i_tile = 0

is_equal = True

acc_interm = acc_interm.transpose(0, 2, 3, 1)

for start_ko in range(0, out_k, tile_k):
    for start_h in range(0, out_h, tile_h):
        for start_w in range(0, out_w, tile_w):
            tile = interm[start_h:start_h + tile_h, start_w:start_w + tile_w, start_ko:start_ko + tile_k]

            n_sub_h = div_and_ceil(min(tile_h, out_h - start_h), subtile_h)
            n_sub_w = div_and_ceil(min(tile_w, out_w - start_w), subtile_w)
            n_sub_ko = div_and_ceil(min(tile_k, out_k - start_ko), subtile_k)
            n_sub = n_sub_ko * n_sub_h * n_sub_w
            tile_acc = acc_interm[start_n_sub:start_n_sub + n_sub]
            start_n_sub += n_sub

            print(f"Checking tile {i_tile}:")
            is_equal_tile = tile_check(tile, tile_acc)
            if is_equal_tile:
                print(f"Tile {i_tile} is equal.\n")
            else:
                print(f"Tile {i_tile} is NOT equal.\n")
            i_tile += 1

            is_equal = is_equal and is_equal_tile

if is_equal:
    print("OK: interm_layer0.txt and interm.log are the same")
else:
    print("ERROR: interm_layer0.txt and interm.log differ")