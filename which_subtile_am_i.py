import argparse
import re

def div_and_ceil(a, b):
    return ((a-1) // b) + 1

def number_of_tiles(in_shape, tile_shape, conv_overlap = 2):
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

parser = argparse.ArgumentParser("Which subtile am I?")
parser.add_argument('row', type=int, help="Row of interest")
parser.add_argument('column', type=int, help="Column of interest")
parser.add_argument('channel', type=int, help="Channel of interest")

args = parser.parse_args()

h = args.row
w = args.column
k = args.channel

root = "/home/lmacan/pulp/dory/dory_examples/"
X_SHAPE, Y_SHAPE, W_SHAPE, X_TILE_SHAPE, Y_TILE_SHAPE, W_TILE_SHAPE = loadtiling(root + "logs/Tiling_profiling.log")
OUT_BUFFER_SHAPE = (3, 3, 32)

out_h, out_w, out_k = Y_SHAPE

tile_shape = Y_TILE_SHAPE
tile_h, tile_w, tile_k = tile_shape

subtile_h, subtile_w, subtile_k = OUT_BUFFER_SHAPE

i_tile_h = h // tile_h
i_tile_w = w // tile_w
i_tile_k = k // tile_k

print(f'Tile ({i_tile_h}, {i_tile_w}, {i_tile_k})')

i_subtile = 0

n_tiles_h, n_tiles_w, n_tiles_k = number_of_tiles(Y_SHAPE, Y_TILE_SHAPE, conv_overlap=0)
is_break = False

for l in range(n_tiles_k): # skipped k because it is used as the channel variable
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            if l == i_tile_k and i == i_tile_h and j == i_tile_w:
                is_break = True
                break
            curr_tile_h = min(tile_h, out_h - i * tile_h)
            curr_tile_w = min(tile_w, out_w - j * tile_w)
            curr_tile_k = min(tile_k, out_k - l * tile_k)

            n_subtile_h, n_subtile_w, n_subtile_k = number_of_tiles((curr_tile_h, curr_tile_w, curr_tile_k), OUT_BUFFER_SHAPE, conv_overlap=0)

            i_subtile += n_subtile_h * n_subtile_w * n_subtile_k
        if is_break:
            break
    if is_break:
        break

shifted_h = h - i_tile_h * tile_h
shifted_w = w - i_tile_w * tile_w
shifted_k = k - i_tile_k * tile_k

i_subtile_h = shifted_h // subtile_h
i_subtile_w = shifted_w // subtile_w
i_subtile_k = shifted_k // subtile_k

curr_tile_h = min(tile_h, out_h - i_tile_h * tile_h)
curr_tile_w = min(tile_w, out_w - i_tile_w * tile_w)
curr_tile_k = min(tile_k, out_k - i_tile_k * tile_k)

n_subtile_h, n_subtile_w, n_subtile_k = number_of_tiles((curr_tile_h, curr_tile_w, curr_tile_k), OUT_BUFFER_SHAPE, conv_overlap=0)

i_subtile += ((i_subtile_k * n_subtile_h + i_subtile_h) * n_subtile_w + i_subtile_w) + 1 # +1 is to get it from 0 initiated indexing to 1 initiated

if i_subtile % 10 == 1:
    number_sufix = 'st'
elif i_subtile % 10 == 2:
    number_sufix = 'nd'
elif i_subtile % 10 == 3:
    number_sufix = 'rd'
else:
    number_sufix = 'th'

print(f'Your output element is produced in the {i_subtile}{number_sufix} subtile.')