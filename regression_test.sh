#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

echo "Test last data: 23/12/2021"

echo "8 bits 2D conv-network test"
rm -rf network_tested
mkdir network_tested

echo "MobilenetV1 MAC/cycle: 8.759356"
python3 network_generate.py --sdk gap_sdk                 
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV1.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV1.txt 

echo "PenguiNet_32 MAC/cycle: 4.790517"
python3 network_generate.py --network_dir './examples/8-bits-2D/PenguiNet_32/' --Bn_Relu_Bits 32 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/PenguiNet_32.txt
grep -FR 'Checksum final :' ./network_tested/PenguiNet_32.txt 

echo "PenguiNet_64 MAC/cycle: 3.983492"
python3 network_generate.py --network_dir './examples/8-bits-2D/PenguiNet_64/' --Bn_Relu_Bits 64 --sdk gap_sdk 
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/PenguiNet_64.txt
grep -FR 'Checksum final :' ./network_tested/PenguiNet_64.txt 

echo "dronet_complete MAC/cycle: 5.754563"
python3 network_generate.py --network_dir './examples/8-bits-2D/dronet_complete/' --Bn_Relu_Bits 64 --sdk gap_sdk --l2_buffer_size 420000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dronet_complete.txt
grep -FR 'Checksum final :' ./network_tested/dronet_complete.txt 

echo "dronet_no_pooling MAC/cycle: 5.810254"
python3 network_generate.py --network_dir './examples/8-bits-2D/dronet_no_pooling/' --Bn_Relu_Bits 64 --sdk gap_sdk --l2_buffer_size 420000 --l1_buffer_size 34000 --master_stack 3900 --slave_stack 3800
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dronet_no_pooling.txt
grep -FR 'Checksum final :' ./network_tested/dronet_no_pooling.txt 

echo "Residual_connection_test_1 MAC/cycle: 6.442216"
python3 network_generate.py --network_dir './examples/8-bits-2D/Residual_connection_test_1/' --Bn_Relu_Bits 32 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/Residual_connection_test_1.txt
grep -FR 'Checksum final :' ./network_tested/Residual_connection_test_1.txt 

echo "Residual_connection_test_2 MAC/cycle: 4.667706"
python3 network_generate.py --network_dir './examples/8-bits-2D/Residual_connection_test_2/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/Residual_connection_test_2.txt
grep -FR 'Checksum final :' ./network_tested/Residual_connection_test_2.txt 

echo "SmartAgri-net MAC/cycle: 5.930013"
python3 network_generate.py --network_dir './examples/8-bits-2D/SmartAgri-net/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/SmartAgri-net.txt
grep -FR 'Checksum final :' ./network_tested/SmartAgri-net.txt 

echo "MobilenetV2 MAC/cycle: 3.860595"
python3 network_generate.py --network_dir './examples/8-bits-2D/MV2-128/' --l2_buffer_size 360000 --Bn_Relu_Bits 64 --sdk gap_sdk --master_stack 4096 --slave_stack 3500 --l1_buffer_size 34000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV2.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV2.txt 

echo "dscnn_test MAC/cycle: 2.975287"
python3 network_generate.py --network_dir './examples/8-bits-2D/dscnn_test/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dscnn_test.txt
grep -FR 'Checksum final :' ./network_tested/dscnn_test.txt 

####### 1D networks ######

echo "TCN_T MAC/cycle: 4.291659" NOT WORKING SINCE DUPLICATION OF MUL-DIV AFTER ADD!!! INTERFACE BETWEEN LAYER AND KERNEL BROKEN!! LETS FIX IT WITH A COMMON INTERFACE
# python3 network_generate.py --network_dir examples/8-bits-1D/TCN_T/ --Bn_Relu_Bits 64 --sdk gap_sdk
# make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/TCN_T.txt
# grep -FR 'Checksum final :' ./network_tested/TCN_T.txt 

echo "TCN_test_library MAC/cycle: 8.815123" BLOCK MUL-MUL-DIV NOT PRESENT IN THE NEMO RULES
# python3 network_generate.py --network_dir examples/8-bits-1D/TCN_test_library/ --Bn_Relu_Bits 64 --sdk gap_sdk
# make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/TCN_test_library.txt
# grep -FR 'Checksum final :' ./network_tested/TCN_test_library.txt 

####### mixed-precision networks ######

echo "MobilenetV1_4bits MAC/cycle: 2.058823"
python3 network_generate.py --network_dir examples/Quantlab_examples/mnv1_4b4b_fixed/ --Bn_Relu_Bits 64 --sdk gap_sdk --frontend Quantlab
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV1_4bits.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV1_4bits.txt 

###### Occamy tests ######
python3 network_generate.py --network_dir ./examples/8-bits-2D/PenguiNet_32/ --backend Occamy --l1_buffer_size 25000 --l2_buffer_size 2000000 --number_of_clusters 2

