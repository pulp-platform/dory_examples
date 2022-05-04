import numpy as np
import util
from util import div_and_ceil, number_of_tiles, number_of_subtiles

CONV_OVERLAP = 2
SUBTILE_SHAPE = (5, 5, 16)
OUT_BUFFER_SHAPE = (32, 3, 3)

root = "/home/lmacan/pulp/dory/dory_examples/"
X_SHAPE, Y_SHAPE, W_SHAPE, X_TILE_SHAPE, Y_TILE_SHAPE, W_TILE_SHAPE = util.loadtiling(root + "logs/Tiling_profiling.log")
interm = util.loadtxt(root + "MiniNet/interm_layer0.txt", Y_SHAPE)
acc_interm = util.loadlog(root + "logs/interm_cleaned.log", OUT_BUFFER_SHAPE)

print(f'Intermediates shape: {interm.shape}')
print(f'Accumulator intermediates shape: {acc_interm.shape}')

def subtile_check(subtile, subtile_acc):
    h, w, k = subtile.shape
    return np.array_equal(subtile, subtile_acc[:h, :w, :k])

def tile_check(tile, tile_acc):
    is_equal = True
    i_subtile = 0
    n_correct = 0

    n_subtiles, sub_h, sub_w, sub_k = tile_acc.shape
    subtile_shape = (sub_h, sub_w, sub_k)
    n_sub_h, n_sub_w, n_sub_k = number_of_tiles(tile.shape, subtile_shape, conv_overlap=0)

    assert n_sub_h * n_sub_w * n_sub_k == n_subtiles, \
            f'n_sub_h = {n_sub_h}, n_sub_w = {n_sub_w}, n_sub_k = {n_sub_k}, tile.shape = {tile.shape}, subtile_shape = {subtile_shape}, n_subtiles = {n_subtiles}'

    tile_acc = tile_acc.reshape((n_sub_k, n_sub_h, n_sub_w) + subtile_shape)

    for i_k, start_k in zip(range(n_sub_k), range(0, tile_k, sub_k)):
        for i_h, start_h in zip(range(n_sub_h), range(0, tile_h, sub_h)):
            for i_w, start_w in zip(range(n_sub_w), range(0, tile_w, sub_w)):
                subtile = tile[start_h:start_h + sub_h, start_w:start_w + sub_w, start_k:start_k + sub_k]
                subtile_acc = tile_acc[i_k, i_h, i_w]
                print(f"Checking subtile {i_subtile}: ", end="")
                is_equal_subtile = subtile_check(subtile, subtile_acc)
                if is_equal_subtile:
                    print("OK!")
                    n_correct += 1
                else:
                    print("ERROR!")
                i_subtile += 1
                is_equal = is_equal and is_equal_subtile
    
    print(f'Score: {n_correct}/{i_subtile}')
    return (is_equal, n_correct, i_subtile)


subtile_k, subtile_h, subtile_w = OUT_BUFFER_SHAPE
subtile_shape = (subtile_h, subtile_w, subtile_k)

n_subtiles = number_of_subtiles(Y_SHAPE, Y_TILE_SHAPE, subtile_shape, conv_overlap=0)

assert n_subtiles == acc_interm.shape[0], \
    f'n_subtiles({n_subtiles}) != acc_interm.shape[0]({acc_interm.shape[0]})'

start_n_sub = 0
i_tile = 0
n_correct = 0
n_subtiles = 0

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

            is_equal_tile, n_correct_tile, n_subtile_tile = tile_check(tile, tile_acc)

            n_correct += n_correct_tile
            n_subtiles += n_subtile_tile

            if is_equal_tile:
                print(f"Tile {i_tile} is equal.\n")
            else:
                print(f"Tile {i_tile} is NOT equal.\n")
            i_tile += 1

            is_equal = is_equal and is_equal_tile

print(f"Final score: {n_correct}/{n_subtiles}")

if is_equal:
    print("OK: interm_layer0.txt and interm.log are the same")
else:
    print("ERROR: interm_layer0.txt and interm.log differ")