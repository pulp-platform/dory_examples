DORY examples and tutorial
==========================

Topology present in tutorial
----------------------------
* Mobilenets V1/V2 - 4/8 bits - 128/224
* Dronet 
* Feed-Forward Neural networks

Requirements
------------

### DORY
See DORY folder for all the requirements. gap_sdk and python packages

Installation
------------
The execution of the dory example requires the following folders:
1. dory_example: contains examples to launch DORY.
2. dory: repository with the framework (submodule of dory_exampe)

Execute the following commands to clone all 
```
git clone https://github.com/pulp-platform/dory_examples
```

## Update modules
Before continuing, update the modules to the latest working version. Refer to the following branches as the most updated:
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

There are many parameters that you can specify in the network_generate.py.
Use

```
python3 network_generate.py -h
```
to see all the parameters which can be specified.

Logs files are produced inside *logs* folder, with network parsed and layers tiling.

Install and build a new network from scratch
--------------------------------------------
First, source in the gap_sdk the platform you prefer

```
source sourceme.sh
```
Execute the following commands to generate the files and test a network on gvsoc with the latest version of GAP8 SDK.
```
python3 network_generate.py Quantlab GAP8 ./dory_examples/config_files/config_Quantlab_MV2_4bits.json --app_dir ./dory_examples/application
cd ${application_dir}
make clean all run CORE=8 platform=gvsoc
```

The variable **CORE** define the number of the cores of the cluster used for parallelization. The default is 8 (the maximum number of cores inside the cluster of GAP8).

Some examples of files generation networks:

```
python3 network_generate.py Quantlab GAP8 ./dory_examples/config_files/config_Quantlab_MV2_4bits.json --app_dir ./dory_examples/application
python3 network_generate.py Quantlab GAP8 ./dory_examples/config_files/config_Quantlab_MV2_8bits.json --app_dir ./dory_examples/application
python3 network_generate.py NEMO GAP8 ./dory_examples/config_files/config_NEMO_dronet.json --app_dir ./dory_examples/application
```

Note that for Quantlab MV2 networks you should use mixed-sw backend that supports signed input/output activations. 