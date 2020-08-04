DORY examples and tutorial
==========================

Topology present in tutorial
----------------------------
* MobilenetV1-128
* 4 Custom networks

Requirements
------------

### DORY
See Dory folder for all the requirements. gap_sdk and python packages

Installation
------------
The execution of the dory example requires the following folders:
1. dory_example: contains examples to launch dory.
2. dory: repository with framework (submodule of dory_exampe)

Execute the following commands to clone all 
```
git clone https://github.com/pulp-platform/dory_examples
```

## Update modules
Before continuining, update the modules to the latest working version. Refer to the following branches as the most updated:
* dory_example --> *master*
* dory --> *master*

Execution
---------
*network_generate.py*: generate files for a whole network.
The file can be directly executed with:
```
python3 network_generate.py
```
Input example networks are included in the *Test_suite_DORY* folder.
Folder *application* contains the whole application to be executed on the GAP8 platform.

There are many parameters which you can specificy in the network_generate.py.
Use

```
python3 network_generate.py -h
```
to see all the parameters which can be specified.

Logs file are produced inside *logs* folder, with network parsed and layers tiling.

Install and build a new network from scratch
--------------------------------------------
First, source in the gap_sdk the gapv2

```
source sourceme.sh
```
Execute the following commands to generate the files and test a network on gvsoc with latest version of GAP8 sdk.
```
python3 network_generate.py
cd application
make clean all run CORE=1 platform=gvsoc
```

Some examples of files generation networks:

```
python3 network_generate.py --network_dir './Test_suite_DORY/PenguiNet_32/' --Mobilenet_correction 0 --Bn_Relu_Bits 32
python3 network_generate.py --network_dir './Test_suite_DORY/PenguiNet_64/' --Mobilenet_correction 0 --Bn_Relu_Bits 64
python3 network_generate.py --network_dir './Test_suite_DORY/MobilenetV1/' --Mobilenet_correction 1 --Bn_Relu_Bits 32
```