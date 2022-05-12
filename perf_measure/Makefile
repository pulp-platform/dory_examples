APP = perf_measure

APP_SRCS += pulp_minimal.c
APP_SRCS += ne16_test.c

DORY_NETWORK_SRCDIR = ../application/DORY_network/src
APP_SRCS += $(DORY_NETWORK_SRCDIR)/pulp_nnx_hal.c

CORE ?= 1

DORY_NETWORK_INCDIR = ../application/DORY_network/inc

APP_CFLAGS += -DNUM_CORES=$(CORE) -I$(DORY_NETWORK_INCDIR) -O3 -w -fno-tree-loop-distribute-patterns -flto
APP_LDFLAGS += -flto

include $(RULES_DIR)/pmsis_rules.mk
