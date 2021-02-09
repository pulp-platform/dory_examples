from shutil import copyfile

#copyfile("/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/PyTorch/frontnet/golden/golden_input_.txt", "renamed_golden/input.txt")
copyfile("/home/osboxes/Documents/Drone/dory_example/net/80x48x32/golden/golden_input_.txt", "input.txt")
#files = ConvBNRelu = ['relu1', 'maxpool', 'layer1.relu', 'relu2', 'layer2.relu', 'relu3', 'layer3.relu', 'relu4', 'avg_pool', 'fc_class']
files = ConvBNRelu = ['relu1', 'maxpool', 'layer1.relu1', 'layer1.relu2', 'layer2.relu1', 'layer2.relu2', 'layer3.relu1', 'layer3.relu2', 'fc']
i = 0
for file in files:
	#copyfile("/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/PyTorch/frontnet/golden/golden_" + file +".txt", "renamed_golden/out_layer" + str(i) + ".txt")
	copyfile("/home/osboxes/Documents/Drone/dory_example/net/80x48x32/golden/golden_" + file +".txt", "out_layer" + str(i) + ".txt")
	i += 1 