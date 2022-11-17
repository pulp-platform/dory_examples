from PlatformEnv import PulpEnv, Gap9Env


PULP_SDK_PATH = "/home/lmacan/pulp/sdk"
PULP_TOOLCHAIN_PATH = "/home/lmacan/pulp/riscv-gcc"
GAP9_SDK_PATH = "/home/lmacan/gap/sdk_private"
GAP9_TOOLCHAIN_PATH = "/usr/lib/gap_riscv_toolchain"

targets = ['nnx.ne16', 'nnx.gap9']


def local_env(target):
    if 'gap9' in target:
        return Gap9Env(GAP9_SDK_PATH, GAP9_TOOLCHAIN_PATH).env
    elif 'ne16' in target:
        return PulpEnv(PULP_SDK_PATH, PULP_TOOLCHAIN_PATH).env
    else:
        return None
