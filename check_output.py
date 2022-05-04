import numpy as np
import util
from util import div_and_ceil, number_of_tiles, number_of_subtiles

CONV_OVERLAP = 2
SUBTILE_SHAPE = (5, 5, 16)
OUT_BUFFER_SHAPE = (32, 3, 3)

CHECK_POSITION = (0, 33, 0)
EXPECTED_VALUE = 0x00
PRODUCED_VALUE = 0x39

root = "/home/lmacan/pulp/dory/dory_examples/"
X_SHAPE, Y_SHAPE, W_SHAPE, X_TILE_SHAPE, Y_TILE_SHAPE, W_TILE_SHAPE = util.loadtiling(root + "logs/Tiling_profiling.log")
interm = util.loadtxt(root + "MiniNet/out_layer0.txt", Y_SHAPE)
acc_out = util.loadlog(root + "logs/output_cleaned.log", OUT_BUFFER_SHAPE)

print(interm.shape)
print(acc_out.shape)

sub_k, sub_h, sub_w = OUT_BUFFER_SHAPE
subtile_shape = (sub_h, sub_w, sub_k)
n_subtiles = number_of_subtiles(Y_SHAPE, Y_TILE_SHAPE, subtile_shape, conv_overlap=0)

assert n_subtiles == acc_out.shape[0], \
    f'n_subtiles({n_subtiles}) != acc_out.shape[0]({acc_out.shape[0]})'

def subtile_check(subtile, subtile_acc):
    h, w, k = subtile.shape
    return np.array_equal(subtile, subtile_acc[:h, :w, :k])

def tile_check(tile, tile_acc, position = None):
    is_equal = True
    i_subtile = 0

    tile_h, tile_w, tile_k = tile.shape
    n_subtiles, sub_h, sub_w, sub_k = tile_acc.shape
    subtile_shape = (sub_h, sub_w, sub_k)

    n_sub_h, n_sub_w, n_sub_k = number_of_tiles(tile.shape, subtile_shape, conv_overlap=0)

    assert n_sub_h * n_sub_w * n_sub_k == n_subtiles, \
            f'n_sub_h = {n_sub_h}, n_sub_w = {n_sub_w}, n_sub_k = {n_sub_k}, tile.shape = {tile.shape}, subtile_shape = {subtile_shape}, n_subtiles = {n_subtiles}'

    tile_acc = tile_acc.reshape(n_sub_k, n_sub_h, n_sub_w, sub_h, sub_w, sub_k)

    for i_k, start_k in zip(range(n_sub_k), range(0, tile_k, sub_k)):
        for i_h, start_h in zip(range(n_sub_h), range(0, tile_h, sub_h)):
            for i_w, start_w in zip(range(n_sub_w), range(0, tile_w, sub_w)):
                end_h = start_h + sub_h
                end_w = start_w + sub_w
                end_k = start_k + sub_k

                subtile = tile[start_h:end_h, start_w:end_w, start_k:end_k]
                subtile_acc = tile_acc[i_k, i_h, i_w]

                print(f"Checking subtile {i_subtile}: ", end="")
                is_equal_subtile = subtile_check(subtile, subtile_acc)
                print("OK!" if is_equal_subtile else "ERROR! Not the same")

                is_check_position = (
                    position is not None
                    and (position[0] >= start_h and position[0] < end_h)
                    and (position[1] >= start_w and position[1] < end_w)
                    and (position[2] >= start_k and position[2] < end_k)
                )


                if is_check_position:
                    position = np.subtract(position, (start_h, start_w, start_k))
                    value = subtile_acc[position[0], position[1], position[2]]
                    if   value != EXPECTED_VALUE and value != PRODUCED_VALUE:
                        print(f"Checked value at position {CHECK_POSITION} differs from both expected and produced value!")
                    elif value == EXPECTED_VALUE and value != PRODUCED_VALUE:
                        print(f"Checked value at position {CHECK_POSITION} is the same as the expected value! There is something wrong with storing them from L1 -> L2 or from acc -> L1")
                    elif value != EXPECTED_VALUE and value == PRODUCED_VALUE:
                        print(f"Checked value at position {CHECK_POSITION} is the same as the produced value! Output and acc are aligned.")
                    else:
                        print(f"Checked value at position {CHECK_POSITION} is the same as expected and produced.")

                i_subtile += 1
                is_equal = is_equal and is_equal_subtile
    
    return is_equal


out_h, out_w, out_k = Y_SHAPE
tile_h, tile_w, tile_k = Y_TILE_SHAPE
subtile_k, subtile_h, subtile_w = OUT_BUFFER_SHAPE

start_n_sub = 0
i_tile = 0

is_equal = True

acc_out = acc_out.transpose(0, 2, 3, 1)

for start_ko in range(0, out_k, tile_k):
    for start_h in range(0, out_h, tile_h):
        for start_w in range(0, out_w, tile_w):
            end_h = start_h + tile_h
            end_w = start_w + tile_w
            end_ko = start_ko + tile_k

            is_check_position = (
                CHECK_POSITION is not None
                and (CHECK_POSITION[0] >= start_h and CHECK_POSITION[0] < end_h)
                and (CHECK_POSITION[1] >= start_w and CHECK_POSITION[1] < end_w)
                and (CHECK_POSITION[2] >= start_ko and CHECK_POSITION[2] < end_ko)
            )

            tile = interm[start_h:end_h, start_w:end_w, start_ko:end_ko]

            n_sub_h = div_and_ceil(tile.shape[0], subtile_h)
            n_sub_w = div_and_ceil(tile.shape[1], subtile_w)
            n_sub_ko = div_and_ceil(tile.shape[2], subtile_k)
            n_sub = n_sub_ko * n_sub_h * n_sub_w

            tile_acc = acc_out[start_n_sub:start_n_sub + n_sub]
            start_n_sub += n_sub

            print(f"Checking tile {i_tile}:")
            if is_check_position:
                is_equal_tile = tile_check(tile, tile_acc, np.subtract(CHECK_POSITION, (start_h, start_w, start_ko)))
            else:
                is_equal_tile = tile_check(tile, tile_acc)

            if is_equal_tile:
                print(f"Tile {i_tile} is equal.\n")
            else:
                print(f"Tile {i_tile} is NOT equal.\n")
            i_tile += 1

            is_equal = is_equal and is_equal_tile

if is_equal:
    print("OK: out_layer0.txt and output.log are the same")
else:
    print("ERROR: out_layer0.txt and output.log differ")