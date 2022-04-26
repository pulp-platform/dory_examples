import argparse
import random

parser = argparse.ArgumentParser(description="Generate a random test.")
parser.add_argument('test_count', type=int, help="Number of subtests inside the generated test.")
parser.add_argument('-fs', '--filter-size', dest='fs', type=int, default=1, help="Filter size. Supported sizes are 1 and 3. Default: 1")
parser.add_argument('-o', '--output', dest='output', type=str, default='random_test.yml', help="Output file name. Default: random_test.yml")

args = parser.parse_args()

test_count = args.test_count
fs = args.fs
output = args.output

with open(output, 'w') as outfile:
    outfile.write('pulp_nnx_tests:\n')
    for i in range(test_count):
        K_IN = random.randint(1, 50)
        K_OUT = random.randint(1, 50)
        H_IN = random.randint(1, 20)
        W_IN = random.randint(1, 20)
        test_name = f'fs{fs}_ki{K_IN}_ko{K_OUT}_hi{H_IN}_wi{W_IN}'
        command = f'make K_IN={K_IN} K_OUT={K_OUT} H_IN={H_IN} W_IN={W_IN}'
        outfile.write(f'  {test_name}:\n    path:\n    command: {command}\n')
