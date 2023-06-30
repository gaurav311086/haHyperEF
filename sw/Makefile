PROJ_ROOT = ..
IMPL := float_stall
HMETIS := "ibm01.hgr"
numbers:= 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18
DEVICE := /opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm

ifeq ($(IMPL), float_pob)
	HW_DIR = $(PROJ_ROOT)/spmv-fp
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv-fp/libfpga
	CXXFLAGS += -D FP_POB
else ifeq ($(IMPL), float_stall)
	HW_DIR = $(PROJ_ROOT)/spmv-fp
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv-fp/libfpga
	CXXFLAGS += -D FP_STALL
else
	HW_DIR = $(PROJ_ROOT)/spmv
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv/libfpga
endif

HOST_ARCH = x86

CXX := g++

CXXFLAGS += -Wall -std=c++11

CXXFLAGS += -I$(CNPY_INCLUDE)
LDFLAGS += -L$(CNPY_LIB) -lcnpy

CXXFLAGS += -I$(PROJ_ROOT)/sw
CXXFLAGS += -I$(LIBFPGA_DIR)

include $(PROJ_ROOT)/xrt/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)

include $(PROJ_ROOT)/xrt/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

LDFLAGS += -lrt -lstdc++

CXXFLAGS += -Wno-maybe-uninitialized
CXXFLAGS += -Wno-uninitialized
CXXFLAGS += -Wno-int-in-bool-context

$(HW_DIR)/build_dir.hw_emu/spmv.xclbin:
	cd $(HW_DIR); make -j5 build TARGET=hw_emu IMPL=$(IMPL) DEVICE=$(DEVICE)

$(HW_DIR)/build_dir.hw/spmv.xclbin:
	cd $(HW_DIR); make -j5 build TARGET=hw IMPL=$(IMPL) DEVICE=$(DEVICE)

.PHONY: demo host hw_emu hw benchmark
host: host.cpp
	$(CXX) $(CXXFLAGS) -g $(xcl2_SRCS) $< -o host $(LDFLAGS)

demo: benchmark.cpp ../demo_spmv.xclbin
	$(CXX) $(CXXFLAGS) -O3 $(xcl2_SRCS) $< -o benchmark $(LDFLAGS)
	./bm.sh ../demo_spmv.xclbin fixed

benchmark: benchmark.cpp $(HW_DIR)/build_dir.hw/spmv.xclbin
	$(CXX) $(CXXFLAGS) -O3 $(xcl2_SRCS) $< -o benchmark $(LDFLAGS)
	./bm.sh $(HW_DIR)/build_dir.hw/spmv.xclbin $(IMPL)

hw_emu: host $(HW_DIR)/build_dir.hw_emu/spmv.xclbin
	cp $(HW_DIR)/emconfig.json .
	./host hw_emu $(HW_DIR)/build_dir.hw_emu/spmv.xclbin

hw: host $(HW_DIR)/build_dir.hw/spmv.xclbin
	./host hw $(HW_DIR)/build_dir.hw/spmv.xclbin

hyperef: hyperef_filter.cpp
	$(CXX) $(CXXFLAGS)-g $(xcl2_SRCS) $< -o hyperef $(LDFLAGS)

hyperef_demo: hyperef $(HW_DIR)/build_dir.hw/spmv.xclbin
	time -f "Program: %C\nTotal time: %E\nUser Mode (s) %U\nKernel Mode (s) %S\nCPU: %P" ./hyperef hw $(HW_DIR)/build_dir.hw/spmv.xclbin $(HMETIS)
	echo "Done"

hyperef_all: hyperef $(HW_DIR)/build_dir.hw/spmv.xclbin
	time -f "Program: %C\nTotal time: %E\nUser Mode (s) %U\nKernel Mode (s) %S\nCPU: %P" ./hyperef hw $(HW_DIR)/build_dir.hw/spmv.xclbin all
	echo "Done"

hyperef_regress: hyperef $(HW_DIR)/build_dir.hw/spmv.xclbin
	$(foreach var,$(numbers),time -f "Program: %C\nTotal time: %E\nUser Mode (s) %U\nKernel Mode (s) %S\nCPU: %P"  ./hyperef hw $(HW_DIR)/build_dir.hw/spmv.xclbin ibm$(var).hgr;)

.PHONY: clean cleanall
clean:
	rm -rf emconfig.json
	rm -rf .run *.log *.csv

cleanall: clean
	rm -rf host benchmark
