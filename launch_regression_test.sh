#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

echo "Test last data: 05/05/2021"

echo "8 bits 2D conv-network test"
rm -rf network_tested
mkdir network_tested

echo "MobilenetV1 MAC/cycle: 8.596562"
python3 network_generate.py --sdk gap_sdk                 
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV1.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV1.txt 

echo "PenguiNet_32 MAC/cycle: 4.875062"
python3 network_generate.py --network_dir './examples/8-bits-2D/PenguiNet_32/' --Bn_Relu_Bits 32 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/PenguiNet_32.txt
grep -FR 'Checksum final :' ./network_tested/PenguiNet_32.txt 

echo "PenguiNet_64 MAC/cycle: 4.058853"
python3 network_generate.py --network_dir './examples/8-bits-2D/PenguiNet_64/' --Bn_Relu_Bits 64 --sdk gap_sdk 
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/PenguiNet_64.txt
grep -FR 'Checksum final :' ./network_tested/PenguiNet_64.txt 

echo "dronet_complete MAC/cycle: 5.768442"
python3 network_generate.py --network_dir './examples/8-bits-2D/dronet_complete/' --Bn_Relu_Bits 64 --sdk gap_sdk --l2_buffer_size 420000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dronet_complete.txt
grep -FR 'Checksum final :' ./network_tested/dronet_complete.txt 

echo "dronet_no_pooling MAC/cycle: 6.268372"
python3 network_generate.py --network_dir './examples/8-bits-2D/dronet_no_pooling/' --Bn_Relu_Bits 64 --sdk gap_sdk --l2_buffer_size 420000
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dronet_no_pooling.txt
grep -FR 'Checksum final :' ./network_tested/dronet_no_pooling.txt 

echo "Residual_connection_test_1 MAC/cycle: 6.322388"
python3 network_generate.py --network_dir './examples/8-bits-2D/Residual_connection_test_1/' --Bn_Relu_Bits 32 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/Residual_connection_test_1.txt
grep -FR 'Checksum final :' ./network_tested/Residual_connection_test_1.txt 

echo "Residual_connection_test_2 MAC/cycle: 4.615693"
python3 network_generate.py --network_dir './examples/8-bits-2D/Residual_connection_test_2/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/Residual_connection_test_2.txt
grep -FR 'Checksum final :' ./network_tested/Residual_connection_test_2.txt 

echo "eegnet_1d_8b_1xW MAC/cycle: 0.510354"
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnet_1d_8b_1xW/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnet_1d_8b_1xW.txt
grep -FR 'Checksum final :' ./network_tested/eegnet_1d_8b_1xW.txt 

echo "eegnet_1d_8b_Hx1 MAC/cycle: 2.750987"
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnet_1d_8b_Hx1/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnet_1d_8b_Hx1.txt
grep -FR 'Checksum final :' ./network_tested/eegnet_1d_8b_Hx1.txt 

echo "eegnetDW_1d_8b_1xW 0.531470"
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnetDW_1d_8b_1xW/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnetDW_1d_8b_1xW.txt
grep -FR 'Checksum final :' ./network_tested/eegnetDW_1d_8b_1xW.txt 

echo "eegnetDW_1d_8b_Hx1 MAC/cycle: 2.518593"
python3 network_generate.py --network_dir './examples/8-bits-2D/eegnetDW_1d_8b_Hx1/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/eegnetDW_1d_8b_Hx1.txt
grep -FR 'Checksum final :' ./network_tested/eegnetDW_1d_8b_Hx1.txt 

echo "SmartAgri-net MAC/cycle: 6.036715"
python3 network_generate.py --network_dir './examples/8-bits-2D/SmartAgri-net/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/SmartAgri-net.txt
grep -FR 'Checksum final :' ./network_tested/SmartAgri-net.txt 

echo "MobilenetV2 MAC/cycle: 4.192413"
python3 network_generate.py --network_dir './examples/8-bits-2D/MV2-128/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV2.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV2.txt 

echo "dscnn_test MAC/cycle: 2.967550"
python3 network_generate.py --network_dir './examples/8-bits-2D/dscnn_test/' --Bn_Relu_Bits 64 --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/dscnn_test.txt
grep -FR 'Checksum final :' ./network_tested/dscnn_test.txt 

####### 1D networks ######

echo "TCN_T MAC/cycle: 4.291659"
python3 network_generate.py --network_dir examples/8-bits-1D/TCN_T/ --Bn_Relu_Bits 64 --sdk gap_sdk --optional 1D_Conv
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/TCN_T.txt
grep -FR 'Checksum final :' ./network_tested/TCN_T.txt 

echo "TCN_test_library MAC/cycle: 8.815123"
python3 network_generate.py --network_dir examples/8-bits-1D/TCN_test_library/ --Bn_Relu_Bits 64 --sdk gap_sdk --optional 1D_Conv
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/TCN_test_library.txt
grep -FR 'Checksum final :' ./network_tested/TCN_test_library.txt 

####### mixed-precision networks ######

echo "MobilenetV1_4bits MAC/cycle: 6.366008"
python3 network_generate.py --network_dir examples/mixed-2D/MobilenetV1_4bits/ --Bn_Relu_Bits 64 --optional mixed-sw --sdk gap_sdk
make -C ./application/ clean all run CORE=8 platform=gvsoc > ./network_tested/MobilenetV1_4bits.txt
grep -FR 'Checksum final :' ./network_tested/MobilenetV1_4bits.txt 
