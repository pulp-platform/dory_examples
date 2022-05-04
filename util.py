import numpy as np
import re

def loadtxt(filename, shape):
    with open(filename, 'r') as infile:
        infile.readline() # skip first line
        l = [line[0:-2] for line in infile]
        return np.array(l, dtype=np.uint8).reshape(shape)

def loadlog(filename, buffer_shape):
    with open(filename, 'r') as infile:
        shape = (-1,) + buffer_shape
        l = [(line[0:-2] if line[-2] == ',' else line[0:-1]).split(',') for line in infile]
        return np.array(l, dtype=np.uint8).reshape(shape)

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

def div_and_ceil(a, b):
    return ((a-1) // b) + 1


def number_of_tiles(in_shape, tile_shape, conv_overlap = 2):
    h_in, w_in, k_in = in_shape
    h_tile, w_tile, k_tile = tile_shape
    n_tiles_h = div_and_ceil(h_in - conv_overlap, (h_tile - conv_overlap))
    n_tiles_w = div_and_ceil(w_in - conv_overlap, (w_tile - conv_overlap))
    n_tiles_k = div_and_ceil(k_in, k_tile)
    return (n_tiles_h, n_tiles_w, n_tiles_k)

def number_of_subtiles(shape, tile_shape, subtile_shape, conv_overlap = 2):
    out_h, out_w, out_k = shape
    tile_h, tile_w, tile_k = tile_shape

    n_tiles_h, n_tiles_w, n_tiles_k = number_of_tiles(shape, tile_shape)

    n_body_subtile_h, n_body_subtile_w, n_body_subtile_k = number_of_tiles(tile_shape, subtile_shape, conv_overlap=conv_overlap)

    rem_tile_h = out_h - (n_tiles_h - 1) * (tile_h - conv_overlap)
    rem_tile_w = out_w - (n_tiles_w - 1) * (tile_w - conv_overlap)
    rem_tile_k = out_k - (n_tiles_k - 1) *  tile_k
            
    rem_tile_shape = (rem_tile_h, rem_tile_w, rem_tile_k)

    n_rem_subtile_h, n_rem_subtile_w, n_rem_subtile_k = number_of_tiles(rem_tile_shape, subtile_shape, conv_overlap=conv_overlap)

    n_subtile_h = n_body_subtile_h * (n_tiles_h - 1) + n_rem_subtile_h
    n_subtile_w = n_body_subtile_w * (n_tiles_w - 1) + n_rem_subtile_w
    n_subtile_k = n_body_subtile_k * (n_tiles_k - 1) + n_rem_subtile_k

    n_subtiles = n_subtile_h * n_subtile_w * n_subtile_k

    return n_subtiles