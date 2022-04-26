# TODO: add argparse
TP_IN = 16
n_h = 1
n_w = 2
Ki = 1024
n_ki = ((Ki - 1) // TP_IN) + 1
n_ko = 4

x_h = 3
x_w = 6

output_lines = [['' for i in range(n_ki * n_w * 3 * 3)] for j in range(n_ko)]

with open("logs/inputs_cleaned.log", 'r') as infile:
    for ko in range(n_ko):
        for i in range(n_h):
            for j in range(n_w):
                for k in range(n_ki):
                    for h in range(3):
                        for w in range(3):
                            output_lines[ko][(((i*3 + h)*n_w + j)*3 + w)*n_ki + k] = infile.readline().strip().replace(",", ",\n")

for i in range(n_ko - 1):
    if output_lines[i] != output_lines[i+1]:
        print("ERROR: Inputs are not the same!")

with open("logs/input_sorted.log", 'w') as outfile:
    outfile.write(f'# input (shape [{x_h}, {x_w}, {Ki}]),\n')
    for line in output_lines[0]:
        outfile.write(line)