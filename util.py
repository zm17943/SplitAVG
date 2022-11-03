import torch
import numpy as np
import copy

def splitavg_propagation(arg, server_net, chosen_net, generators, device, criterion, optimizers, optimizer_server):
	inputs_list, labels_list = [], []
	for generator in generators:
		inputs, labels = next(generator)
		inputs_list.append(inputs.to(device))
		labels_list.append(labels.to(device))

	inputs_m = copy.deepcopy(inputs_list)
	if not arg.splitavg_v2:
		label_cat = torch.cat(labels_list, 0)

	optimizer_server.zero_grad()
	for optimizer in optimizers:
		optimizer.zero_grad()


	# Forward propagation on client sub-net until the cut layer
	for client in range(len(chosen_net)):
		for name0, midlayer0 in chosen_net[client]._modules.items():
			inputs_m[client] = midlayer0(inputs_m[client])
			if name0 == arg.cut:
				break

	# Concatenate intermediate outputs at cut layer 
	feature_cat = torch.cat([mid_output.detach() for mid_output in inputs_m])
	feature_cat.requires_grad = True
	feature_cat.retain_grad()


	server_net.train()
	for client in range(len(chosen_net)):
		chosen_net[client].train()


	inputs_server = torch.randn_like(inputs_list[0])                             # A dummy input tensor for server net, replaced at cut layer
	for name0, midlayer0 in server_net._modules.items():
		if name0 != arg.cut:
			inputs_server = midlayer0(inputs_server)
			if(name0 == 'avgpool'):                               
				inputs_server = torch.flatten(inputs_server, 1)      # Only for ResNet architecture
		if name0 == arg.cut:
			inputs_server = midlayer0(inputs_server)
			inputs_server = feature_cat

	# Server net backward propagation
	if not arg.splitavg_v2:
		loss = criterion(np.squeeze(inputs_server, axis=1), label_cat)
	else:
		loss_from_clients = []
		for i in range(arg.sample_num):
			loss_from_clients.append(criterion(inputs_server[i*arg.batch_size:(i+1)*arg.batch_size], labels_list[i]))
		loss = torch.mean(torch.tensor(loss_from_clients))
	
	loss.backward()
	running_loss = loss.item()
	

    # Server send the gradients at cut layer back to clients
	grad_to_send = feature_cat.grad[0*arg.batch_size:(0+1)*arg.batch_size]
	for i in range(1, arg.sample_num):
		grad_to_send += feature_cat.grad[i*arg.batch_size:(i+1)*arg.batch_size]
	grad_to_send /= arg.sample_num


	optimizer_server.step()
	optimizer_server.zero_grad()


	# Clients receive gradients and backpropagate
	for client in range(arg.sample_num):
		inputs_m[client].grad = grad_to_send
		inputs_m[client].backward(torch.ones_like(inputs_m[client]))
		optimizers[client].step()

	return running_loss


def val(arg, epoch, acc_site, best_acc_site, nets, server_net, test_set_loader, test_len, device, criterion):
	acc_site = {}
	test_loss_site = {}
	server_net.eval()
	for site in range(arg.site_num):
		nets[site].eval()
		acc = 0
		test_loss = 0
		with torch.no_grad():
			for data, target in test_set_loader:
				data = data.to(device)
				target = target.to(device)

				if_server_net = False
				for name0, midlayer0 in nets[site]._modules.items():
					if if_server_net:
						data = server_net._modules[name0](data)
						if(name0 == 'avgpool'):
							data = torch.flatten(data, 1)
					else:
						data = nets[site]._modules[name0](data)
						if (name0 == arg.cut):
							if_server_net = True

				loss = criterion(np.squeeze(data, axis=1), target) 
				test_loss += loss.item()
				# predict_idx = data.argmax(1, keepdim=True)
				# for h in range(arg.batch_size):
				# 	if(predict_idx[h] == target[h]):
				# 		acc += 1

		# acc_site[str(site)] = acc/test_len
		test_loss_site[str(site)] = test_loss/test_len


		# if (acc_site[str(site)] > best_acc_site[str(site)]):
		# 	best_acc_site[str(site)] = acc_site[str(site)]
		# 	if (arg.save_best):
		# 		torch.save(nets[site], './checkpoint/site{}_ckpt_best.pth'.format(site))
		# torch.save(nets[site], './checkpoint/site{}_epoch{}_ckpt.pth'.format(site, epoch))

	return test_loss_site


