# TODO: add argparse
TP_IN = 16
TP_OUT = 32
n_h = 1
n_w = 2
Ki = 1024
Ko = 32
n_ki = ((Ki - 1) // TP_IN) + 1
n_ko = 4

n_tiles = n_ko * n_h * n_w

x_h = 3
x_w = 6
y_h = 3
y_w = 6

with open("logs/intermediates_cleaned.log", 'r') as infile:
    input_lines = infile.readlines()

input_lines = [[line.strip()[:-2].replace('{', '').replace(' ', '').split(',') for line in input_lines[start:start+Ko//n_ko]] for start in range(0, n_tiles * TP_OUT, TP_OUT)]

output_lines = ['' for i in range(y_h * y_w * Ko)]

for i, acc in enumerate(input_lines):
    for j, spatial_slice in enumerate(acc):
        for k, el in enumerate(spatial_slice):
            i_n_spatial = i % (n_h * n_w)
            i_n_h = i_n_spatial // n_w
            i_n_w = i_n_spatial % n_w
            h = k // 3
            w = k % 3
            output_lines[((i_n_h*3 + h)*y_w + i_n_w*3 + w)*Ko + (i // (n_h * n_w))*(Ko // n_ko) + j] = el

with open("logs/intermediates_sorted.log", 'w') as outfile:
    outfile.write(f'# _QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0 (shape [{y_h}, {y_w}, {Ko}]),\n')
    for line in output_lines:
        outfile.write(line + ",\n")