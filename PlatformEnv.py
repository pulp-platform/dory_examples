import os
from abc import ABC, abstractmethod


class PlatformEnv(ABC):

    def __init__(self, sdk_path, toolchain_path, vsim_path='', xcsim_path=''):
        self.sdk_path = sdk_path
        self.toolchain_path = toolchain_path
        self.vsim_path = vsim_path
        self.xcsim_path = xcsim_path
        self._init_env(os.environ.copy())


    def _init_env(self, env):
        self.env = env if env is not None else dict()
        if self.env["PATH"] is None:
            self.env["PATH"] = ""
        self._extend_env()
        self._extend_path()


    def _extend_path(self):
        self.env["PATH"] = self._extra_path_vars() + self.env["PATH"]


    def _extend_env(self):
        self.env.update(self._extra_env_vars())


    @abstractmethod
    def _extra_path_vars(self):
        pass

    @abstractmethod
    def _extra_env_vars(self):
        pass


class Gap9Env(PlatformEnv):

    def _extra_path_vars(self):
        return f"{self.sdk_path}:" \
               f"{self.sdk_path}/utils/bin:" \
               f"{self.sdk_path}/utils/gapy:" \
               f"{self.sdk_path}/utils/gaptest:" \
               f"{self.sdk_path}/install/workstation/bin:" \
               f"{self.sdk_path}/install/workstation/openocd/bin:" \
               f"{self.sdk_path}/tools/sfu_gen/install/bin:" \
               f"{self.sdk_path}/tools/nntool/scripts:" \
               f"{self.toolchain_path}/bin:"

    def _extra_env_vars(self):
        return {
            "GAP_SDK_HOME": f"{self.sdk_path}",
            "OPENOCD_SCRIPTS": f"{self.sdk_path}/utils/openocd_tools",
            "TARGET_CHIP_FAMILY": "GAP9",
            "TARGET_CHIP": "GAP9_V2",
            "TARGET_NAME": "gap9_v2",
            "GVSOC_CONFIG": "gap9_v2",
            "BOARD_NAME": "gap9_evk",
            "BOARD_FEATURES": "audio_addon",
            "PULP_CURRENT_CONFIG": "gap9_v2@config_file=config/gap9_v2.json",
            "PULPOS_BOARD": "gap9_evk",
            "PULPOS_BOARD_VERSION": "gap9_evk",
            "PULPOS_BOARD_PROFILE": "gap9_evk",
            "PULPOS_TARGET": "gap9_v2",
            "PULPOS_SYSTEM": "gap9_v2",
            "GAPY_TARGET": "gap9_v2",
            "GAPY_NEW_TARGET": "gap9.evk",
            "PROFILER_SIGNAL_TREE": f"{self.sdk_path}/tools/profiler/gui/images/signalstree_GAP9.txt",
            "GAPY_PY_TARGET": "Gap9_evk_audio@gap.gap9.gap9_evk",
            "OPENOCD_CABLE": f"{self.sdk_path}/utils/openocd/tcl/interface/ftdi/olimex-arm-usb-tiny-h.cfg",
            "GAPY_OPENOCD_CABLE": f"{self.sdk_path}/utils/openocd/tcl/interface/ftdi/olimex-arm-usb-tiny-h.cfg",
            "GAP_SSBL_FLASH": "target/chip/soc/mram",
            "GAP_SSBL_PATH": f"{self.sdk_path}/utils/ssbl/bin/ssbl-gap9_evk",
            "PLPTEST_DEFAULT_PROPERTIES": "chip=gap9 chip_family=gap9 board=gap9_evk duration=50 test_duration=50",
            "PULP_SDK_HOME": f"{self.sdk_path}",
            "GAP_RISCV_GCC_TOOLCHAIN": f"{self.toolchain_path}",
            "TARGET_INSTALL_DIR": f"{self.sdk_path}/install/GAP9_V2",
            "INSTALL_DIR": f"{self.sdk_path}/install/workstation",
            "DEP_DIRS": f"{self.sdk_path}/install/workstation",
            "RULES_DIR": f"{self.sdk_path}/utils/rules",
            "NNTOOL_DIR": f"{self.sdk_path}/tools/nntool",
            "NNTOOL_PATH": f"{self.sdk_path}/tools/nntool/scripts",
            "NNTOOL_KERNELS_PATH": f"{self.sdk_path}/tools/nntool/autotiler/kernels",
            "NNTOOL_MATH_PATH": f"{self.sdk_path}/tools/nntool/autotiler/math_funcs",
            "NNTOOL_GENERATOR_PATH": f"{self.sdk_path}/tools/nntool/autotiler/generators",
            "PYTHONPATH": f"{self.sdk_path}/utils/gapy:{self.sdk_path}/utils/gap_configs/python:{self.sdk_path}/tools/audio-framework/frontends/python_graph_generator:{self.sdk_path}/tools/audio-framework/components:{self.sdk_path}/gvsoc/gvsoc_gap/generators:{self.sdk_path}/gvsoc/gvsoc/generators:{self.sdk_path}/gvsoc/gvsoc/engine/python:{self.sdk_path}/gvsoc/gvsoc_gap/models:{self.sdk_path}/install/workstation/python:{self.sdk_path}/tools/nntool:",
            "OPENMP_DIR": f"{self.sdk_path}/libs/openmp",
            "PULPOS_HOME": f"{self.sdk_path}/rtos/pulp/pulpos-2",
            "PULPOS_MODULES": f"{self.sdk_path}/rtos/pmsis/implem {self.sdk_path}/rtos/pulp/pulpos-2_gap8 {self.sdk_path}/rtos/pulp/pulpos-2_gap9 {self.sdk_path}/rtos/pmsis/bsp {self.sdk_path}/libs/openmp {self.sdk_path}/rtos/sfu",
            "PULPOS_GAP8_HOME": f"{self.sdk_path}/rtos/pulp/pulpos-2_gap8",
            "PULPOS_GAP9_HOME": f"{self.sdk_path}/rtos/pulp/pulpos-2_gap9",
            "GAP_PULPOS_ARCHI": f"{self.sdk_path}/rtos/pulp/gap_archi",
            "PULPOS_ARCHI": f"{self.sdk_path}/rtos/pulp/archi_pulp",
            "PULPOS_HAL": f"{self.sdk_path}/rtos/pulp/hal_pulp",
            "PMSIS_API": f"{self.sdk_path}/rtos/pmsis/api",
            "SFU_RUNTIME": f"{self.sdk_path}/rtos/sfu",
            "PMSIS_HOME": f"{self.sdk_path}/rtos/pmsis",
            "PULP_LIB_DIR": f"{self.sdk_path}/install/GAP9_V2/lib",
            "PULP_INC_DIR": f"{self.sdk_path}/install/GAP9_V2/include",
            "RUNTIME_PATH": f"{self.sdk_path}/pulp-os",
            "FREERTOS_PATH": f"{self.sdk_path}/rtos/pmsis/os/freeRTOS",
            "LD_LIBRARY_PATH": f"{self.sdk_path}/install/workstation/lib",
            "PULP_CONFIGS_PATH": f"{self.sdk_path}/utils/gap_configs/configs:{self.sdk_path}/install/workstation/configs",
            "PULP_RISCV_GCC_TOOLCHAIN": f"{self.toolchain_path}",
            "PULP_SDK_INSTALL": f"{self.sdk_path}/install/workstation",
            "GVSOC_PATH": f"{self.sdk_path}/install/workstation/python",
            "GVSOC_ISS_PATH": f"{self.sdk_path}/gvsoc/gvsoc/models/cpu/iss",
            "XTENSOR_INCLUDE_DIR": f"{self.sdk_path}/gvsoc/ext/xtensor/include",
            "GVSOC_SRC_PATH": f"{self.sdk_path}/gvsoc/gvsoc",
            "GVSOC_GAP_SRC_PATH": f"{self.sdk_path}/gvsoc/gvsoc_gap",
            "GVSOC_SFU_PATH": f"{self.sdk_path}/gvsoc/gvsoc_gap_sfu",
            "CONFIG_GVSOC_SKIP_UDMA_BUILD": "1",
            "GAP_AUDIO_FRAMEWORK_HOME": f"{self.sdk_path}/tools/audio-framework",
            "AT_HOME": f"{self.sdk_path}/tools/autotiler_v3",
            "TILER_PATH": f"{self.sdk_path}/tools/autotiler_v3",
            "TILER_LIB": f"{self.sdk_path}/tools/autotiler_v3/Autotiler/LibTile.a",
            "TILER_INC": f"{self.sdk_path}/tools/autotiler_v3/Autotiler",
            "TILER_EMU_INC": f"{self.sdk_path}/tools/autotiler_v3/Emulation",
            "TILER_GENERATOR_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators",
            "TILER_BILINEAR_RESIZE_GENERATOR_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/BilinearResizes",
            "TILER_BILINEAR_RESIZE_KERNEL_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/BilinearResizes",
            "TILER_INTEGRAL_GENERATOR_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/IntegralImage",
            "TILER_INTEGRAL_KERNEL_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/IntegralImage",
            "TILER_FFT2D_GENERATOR_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/FFT2DModel",
            "TILER_FFT2D_KERNEL_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/FFT2DModel",
            "TILER_FFT2D_TWIDDLE_PATH": f"{self.sdk_path}/tools/autotiler_v3/Generators/FFT2DModel",
            "TILER_CNN_KERNEL_PATH": f"{self.sdk_path}/tools/autotiler_v3/CNN_Libraries",
            "TILER_CNN_GENERATOR_PATH": f"{self.sdk_path}/tools/autotiler_v3/CNN_Generators",
            "TILER_CNN_KERNEL_PATH_SQ8": f"{self.sdk_path}/tools/autotiler_v3/CNN_Libraries_SQ8",
            "TILER_CNN_KERNEL_PATH_NE16": f"{self.sdk_path}/tools/autotiler_v3/CNN_Libraries_NE16",
            "TILER_CNN_KERNEL_PATH_FP16": f"{self.sdk_path}/tools/autotiler_v3/CNN_Libraries_fp16",
            "TILER_CNN_GENERATOR_PATH_SQ8": f"{self.sdk_path}/tools/autotiler_v3/CNN_Generators_SQ8",
            "TILER_CNN_GENERATOR_PATH_NE16": f"{self.sdk_path}/tools/autotiler_v3/CNN_Generators_NE16",
            "TILER_CNN_GENERATOR_PATH_FP16": f"{self.sdk_path}/tools/autotiler_v3/CNN_Generators_fp16",
            "TILER_DSP_GENERATOR_PATH": f"{self.sdk_path}/tools/autotiler_v3/DSP_Generators",
            "TILER_DSP_KERNEL_PATH": f"{self.sdk_path}/tools/autotiler_v3/DSP_Libraries",
            "TILER_FFT_LUT_PATH": f"{self.sdk_path}/tools/autotiler_v3/DSP_Libraries/LUT_Tables",
            "TILER_MFCC_GEN_LUT_SCRIPT": f"{self.sdk_path}/tools/autotiler_v3/DSP_Libraries/LUT_Tables/gen_scripts/GenMFCCLUT.py",
            "TILER_FFT_GEN_LUT_SCRIPT": f"{self.sdk_path}/tools/autotiler_v3/DSP_Libraries/LUT_Tables/gen_scripts/GenFFTLUT.py",
            "TILER_WIN_GEN_LUT_SCRIPT": f"{self.sdk_path}/tools/autotiler_v3/DSP_Libraries/LUT_Tables/gen_scripts/GenWinLUT.py",
            "GAP_OPENOCD_TOOLS": f"{self.sdk_path}/utils/openocd_tools",
            "GAP_USE_OPENOCD": "1",
            "CROSS_COMPILE": f"{self.toolchain_path}/bin/riscv32-unknown-elf-",
            "ZEPHYR_GCC_VARIANT": "cross-compile",
            "GAP_LIB_PATH": f"{self.sdk_path}/libs/gap_lib",
            "VSIM_PATH": f"{self.vsim_path}",
            "XCSIM_PATH": f"{self.xcsim_path}",
            "XCSIM_PLATFORM": f"{self.xcsim_path}"
        }


class PulpEnv(PlatformEnv):

    def _extra_path_vars(self):
        return f"{self.sdk_path}/tools/gapy:" \
               f"{self.sdk_path}/install/workstation/bin:" \
               f"{self.sdk_path}/tools/bin:"

    def _extra_env_vars(self):
        return {
            "PULP_SDK_HOME": f"{self.sdk_path}",
            "TARGET_CHIP_FAMILY": f"PULP",
            "TARGET_CHIP": f"PULP",
            "TARGET_NAME": f"pulp",
            "BOARD_NAME": f"pulp",
            "PULP_CURRENT_CONFIG": f"pulp@config_file=config/pulp.json",
            "PULPOS_BOARD": f"pulp",
            "PULPOS_BOARD_VERSION": f"pulp",
            "PULPOS_BOARD_PROFILE": f"pulp",
            "PULPOS_TARGET": f"pulp",
            "PULPOS_SYSTEM": f"pulp",
            "GAPY_TARGET": f"pulp",
            "PULPOS_MODULES": f"{self.sdk_path}/rtos/pulpos/pulp {self.sdk_path}/rtos/pmsis/pmsis_bsp",
            "GVSOC_MODULES": f"{self.sdk_path}/tools/gvsoc/common {self.sdk_path}/tools/gvsoc/pulp/models",
            "GAP_SDK_HOME": f"{self.sdk_path}",
            "GAP_RISCV_GCC_TOOLCHAIN": f"",
            "TARGET_INSTALL_DIR": f"{self.sdk_path}/install/PULP",
            "INSTALL_DIR": f"{self.sdk_path}/install/workstation",
            "DEP_DIRS": f"{self.sdk_path}/install/workstation",
            "RULES_DIR": f"{self.sdk_path}/tools/rules",
            "LD_LIBRARY_PATH": f"{self.sdk_path}/install/workstation/lib:",
            "PYTHONPATH": f"{self.sdk_path}/tools/gapy:{self.sdk_path}/tools/gap-configs/python:{self.sdk_path}/install/workstation/python:",
            "PULP_CONFIGS_PATH": f"{self.sdk_path}/tools/gap-configs/configs:",
            "PULP_SDK_INSTALL": f"{self.sdk_path}/install/workstation",
            "GVSOC_PATH": f"{self.sdk_path}/install/workstation/python",
            "XTENSOR_INCLUDE_DIR": f"{self.sdk_path}/ext/xtensor/include",
            "GVSOC_INC_PATHS": f"{self.sdk_path}/rtos/pulpos/gap_archi/include/archi/chips/pulp {self.sdk_path}/rtos/pulpos/gap_archi/include {self.sdk_path}/rtos/pulpos/pulp_archi/include",
            "PULPOS_HOME": f"{self.sdk_path}/rtos/pulpos/common",
            "PULPOS_PULP_HOME": f"{self.sdk_path}/rtos/pulpos/pulp",
            "PULPOS_GAP8_HOME": f"{self.sdk_path}/rtos/pulpos/gap8",
            "PULPOS_GAP9_HOME": f"{self.sdk_path}/rtos/pulpos/gap9",
            "GAP_PULPOS_ARCHI": f"{self.sdk_path}/rtos/pulpos/gap_archi",
            "PULPOS_ARCHI": f"{self.sdk_path}/rtos/pulpos/pulp_archi",
            "PULPOS_HAL": f"{self.sdk_path}/rtos/pulpos/pulp_hal",
            "PMSIS_API": f"{self.sdk_path}/rtos/pmsis/pmsis_api",
            "PULP_EXT_LIBS": f"{self.sdk_path}/ext_libs",
            "PULPOS_OPENMP_DIR": f"{self.sdk_path}/rtos/pulpos/common/lib/omp",
            "RISCV": f"{self.toolchain_path}",
            "PULP_RUNTIME_GCC_TOOLCHAIN": f"{self.toolchain_path}",
            "PULP_RISCV_GCC_TOOLCHAIN": f"{self.toolchain_path}",
        }
