#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

echo "Test last data: 23/12/2021"

echo "8 bits 2D conv-network test"
rm -rf network_tested
mkdir network_tested

echo "MobilenetV1 MAC/cycle: 7.797345"
python3 network_generate.py --sdk gap_sdk                 
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV1.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV1.txt 

echo "PenguiNet_32 MAC/cycle: 4.480294"
python3 network_generate.py --network_dir './examples/8-bits-2D/PenguiNet_32/' --Bn_Relu_Bits 32 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/PenguiNet_32.txt
grep -FR 'Checksum final :' ./network_tested/PenguiNet_32.txt 

echo "PenguiNet_64 MAC/cycle: 3.873038"
python3 network_generate.py --network_dir './examples/8-bits-2D/PenguiNet_64/' --Bn_Relu_Bits 64 --sdk gap_sdk 
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/PenguiNet_64.txt
grep -FR 'Checksum final :' ./network_tested/PenguiNet_64.txt 

echo "dronet_complete MAC/cycle: 5.490148"
python3 network_generate.py --network_dir './examples/8-bits-2D/dronet_complete/' --Bn_Relu_Bits 64 --sdk gap_sdk --l2_buffer_size 420000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dronet_complete.txt
grep -FR 'Checksum final :' ./network_tested/dronet_complete.txt 

echo "dronet_no_pooling MAC/cycle: 6.268372" --> STACK PROBLEMS
python3 network_generate.py --network_dir './examples/8-bits-2D/dronet_no_pooling/' --Bn_Relu_Bits 64 --sdk gap_sdk --l2_buffer_size 420000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dronet_no_pooling.txt
grep -FR 'Checksum final :' ./network_tested/dronet_no_pooling.txt 

echo "Residual_connection_test_1 MAC/cycle: 6.028218"
python3 network_generate.py --network_dir './examples/8-bits-2D/Residual_connection_test_1/' --Bn_Relu_Bits 32 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/Residual_connection_test_1.txt
grep -FR 'Checksum final :' ./network_tested/Residual_connection_test_1.txt 

echo "Residual_connection_test_2 MAC/cycle: 4.361165"
python3 network_generate.py --network_dir './examples/8-bits-2D/Residual_connection_test_2/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/Residual_connection_test_2.txt
grep -FR 'Checksum final :' ./network_tested/Residual_connection_test_2.txt 

echo "eegnet_1d_8b_1xW MAC/cycle: 0.510354" --> using default 1D Now -- NOT TESTED
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnet_1d_8b_1xW/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnet_1d_8b_1xW.txt
grep -FR 'Checksum final :' ./network_tested/eegnet_1d_8b_1xW.txt 

echo "eegnet_1d_8b_Hx1 MAC/cycle: 2.750987" --> using default 1D Now -- NOT TESTED
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnet_1d_8b_Hx1/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnet_1d_8b_Hx1.txt
grep -FR 'Checksum final :' ./network_tested/eegnet_1d_8b_Hx1.txt 

echo "eegnetDW_1d_8b_1xW 0.531470" --> using default 1D Now -- NOT TESTED
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnetDW_1d_8b_1xW/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnetDW_1d_8b_1xW.txt
grep -FR 'Checksum final :' ./network_tested/eegnetDW_1d_8b_1xW.txt 

echo "eegnetDW_1d_8b_Hx1 MAC/cycle: 2.518593" --> using default 1D Now -- NOT TESTED
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnetDW_1d_8b_Hx1/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnetDW_1d_8b_Hx1.txt
grep -FR 'Checksum final :' ./network_tested/eegnetDW_1d_8b_Hx1.txt 

echo "SmartAgri-net MAC/cycle: 5.741506"
python3 network_generate.py --network_dir './examples/8-bits-2D/SmartAgri-net/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/SmartAgri-net.txt
grep -FR 'Checksum final :' ./network_tested/SmartAgri-net.txt 

echo "MobilenetV2 MAC/cycle: 3.847712"
python3 network_generate.py --network_dir './examples/8-bits-2D/MV2-128/' --l2_buffer_size 360000 --Bn_Relu_Bits 64 --sdk gap_sdk --master_stack 4096 --slave_stack 3500 --l1_buffer_size 34000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV2.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV2.txt 

echo "dscnn_test MAC/cycle: 2.842320"
python3 network_generate.py --network_dir './examples/8-bits-2D/dscnn_test/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dscnn_test.txt
grep -FR 'Checksum final :' ./network_tested/dscnn_test.txt 

####### 1D networks ######

echo "TCN_T MAC/cycle: 4.291659" NOT WORKING SINCE DUPLICATION OF MUL-DIV AFTER ADD!!! INTERFACE BETWEEN LAYER AND KERNEL BROKEN!! LETS FIX IT WITH A COMMON INTERFACE
python3 network_generate.py --network_dir examples/8-bits-1D/TCN_T/ --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/TCN_T.txt
grep -FR 'Checksum final :' ./network_tested/TCN_T.txt 

echo "TCN_test_library MAC/cycle: 8.815123" BLOCK MUL-MUL-DIV NOT PRESENT IN THE NEMO RULES
python3 network_generate.py --network_dir examples/8-bits-1D/TCN_test_library/ --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/TCN_test_library.txt
grep -FR 'Checksum final :' ./network_tested/TCN_test_library.txt 

####### mixed-precision networks ######

echo "MobilenetV1_4bits MAC/cycle: 6.366008" WAITING FOR ANNOTATION OF GRAPHS
python3 network_generate.py --network_dir examples/mixed-2D/MobilenetV1_4bits/ --Bn_Relu_Bits 64 --optional mixed-sw --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV1_4bits.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV1_4bits.txt 