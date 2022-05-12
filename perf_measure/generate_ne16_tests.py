def generate_ne16_tests(tests, header_name):
    header = ''

    header_guard = '__' + header_name.replace('.', '_').upper() + '__'

    # Preamble
    header += \
f"""#ifndef {header_guard}
#define {header_guard}

#include "ne16_test.h"

#define N_GENERATED_NE16_TESTS ({len(tests)})

static ne16_test_t ne16_tests[] = {{
"""

    for test in tests:
        header += \
f"""    {{
        .k_out = {test['k_out']},
        .k_in = {test['k_in']},
        .h_in = {test['h_in']},
        .w_in = {test['w_in']},
        .fs = {test['fs']},
        .qw = {test['qw']},
        .dw = {test['dw']}
    }},
"""

    # Postamble
    header += \
f"""}};

#endif  // {header_guard}
"""

    return header


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser("NE16 tests generator",
            "Give it a yaml file with tests and it will generate the header file.")

    parser.add_argument("yml_file", help="Yaml file with tests.")
    parser.add_argument("--output", "-o", dest="output", default="generated_ne16_tests.h", help="Name of the header file to write the tests to.")

    args = parser.parse_args()

    with open(args.yml_file, 'r') as ymlfile:
        ne16_tests = yaml.full_load(ymlfile)

    with open(args.output, 'w') as outfile:
        outfile.write(generate_ne16_tests(ne16_tests, args.output))
