import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
from data_loader import get_data_loader_for_chosen
from data_loader import get_data_loader_for_evaluation
from util import splitavg_propagation
from util import val
import random
from sklearn.utils import shuffle
import math
import numpy as np




def split_net(arg, nets, server_net):
  for client_net in nets:
    if_server_net = False
    for name0, midlayer0 in client_net._modules.items():
      if if_server_net:
        for param in midlayer0.parameters():
          param.requires_grad = False
      else:
        for param in server_net._modules[name0].parameters():
          param.requires_grad = False
      if (name0 == arg.cut):
        if_server_net = True
  return nets, server_net

def setup_models(arg, acc_site, best_acc_site, test_loss_site):
  nets = []
  for site in range(arg.site_num):
    if (arg.arch == "res34"):
      nets.append(models.resnet34(pretrained=True))
      nets[-1].fc = nn.Linear(512, arg.num_class)
    if (arg.arch == "mobile"):
      nets.append(models.mobilenet_v2(pretrained=True))
      nets[-1].classifier[1] = nn.Linear(1280, arg.num_class)
    acc_site[str(site)] = 0
    test_loss_site[str(site)] = 0
    best_acc_site[str(site)] = 0

  if (arg.arch == "res34"):
    server_net = models.resnet34(pretrained=True)
    server_net.fc = nn.Linear(512, arg.num_class)
  if (arg.arch == "mobile"):
    server_net = models.mobilenet_v2(pretrained=True)
    server_net.classifier[1] = nn.Linear(1280, arg.num_class)

  # Leave only client sub-network and server sub-network trainable
  nets, server_net = split_net(arg, nets, server_net)
  return nets, server_net.to(arg.device)


def main():

  model_names = ['res34', 'mobile']

  acc_site, best_acc_site, test_loss_site = {}, {}, {}


  parser = argparse.ArgumentParser(description="PyTorch DDCNN")
  parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
  parser.add_argument("--num_class", type=int, default=2)
  parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
  parser.add_argument("--fineSize", type=int, default=224, help="The size of processed image")
  parser.add_argument("--loadSize", type=int, default=256, help="The size of original image")
  parser.add_argument("--eval_freq", type=int, default=2, help="Validation frequent")
  parser.add_argument("--save_best", type=bool, default=True, help="If save the best validation model")
  parser.add_argument("--epoch_num", type=int, default=60, help="The number of training round")
  parser.add_argument("--iter_per_epoch", type=int, default=71, help="The number of batch forward/backward propagation per training round")
  parser.add_argument("--train_file", type=str, default="./data/boneS1.h5", help="The path of training data split")
  parser.add_argument("--val_file", type=str, default="./data/val.h5", help="The path of validation data")
  parser.add_argument('--site_num', default=4, type=int, help="The total number of participating local sites")
  parser.add_argument('--sample_num', default=4, type=int, help="The number of local sites sampled in each round")
  parser.add_argument('--seed', default=2556, type=int)
  parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='Set the GPU device')
  parser.add_argument('--arch', default='res34', choices=model_names, help='model architecture')
  parser.add_argument('--cut', default='conv1', help='The name of cut layer, specified according to the used model')
  parser.add_argument('--splitavg_v2', type=bool, default=False, help='If to run SplitAVG-v2, no label sharing')


  arg = parser.parse_args()

  # Set the random seed
  random.seed(arg.seed)
  np.random.seed(arg.seed)
  torch.manual_seed(arg.seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed(arg.seed)


  # Set up local client models and server model
  nets, server_net = setup_models(arg, acc_site, best_acc_site, test_loss_site)



  #Training
  for epoch in range(arg.epoch_num):
    chosen = random.sample(list(range(arg.site_num)), arg.sample_num)
    chosen_net = [nets[index].to(arg.device) for index in chosen]
    
    # Load data splits from local sites
    generators = get_data_loader_for_chosen(arg, chosen)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    optimizers = [torch.optim.SGD(net.parameters(), lr=arg.lr, momentum=0.9) for net in chosen_net]
    optimizer_server = torch.optim.SGD(server_net.parameters(), lr=arg.lr, momentum=0.9)

    mean_loss = 0
    for iteration in range(arg.iter_per_epoch):
      running_loss = splitavg_propagation(arg, server_net, chosen_net, generators, arg.device, criterion, optimizers, optimizer_server)
      mean_loss += running_loss

    mean_loss /= arg.iter_per_epoch
    print('At epoch: {} Step: {} AVERAGE TRAIN loss : {:.4f}'.
                  format(epoch, arg.iter_per_epoch, mean_loss))


    if (epoch % arg.eval_freq == 0):
      test_set_loader, test_len = get_data_loader_for_evaluation(arg)
      test_loss_site = val(arg, epoch, acc_site, best_acc_site, nets, server_net, test_set_loader, test_len, arg.device, criterion)
      print('-----------------------------------------------------')
      print('At epoch: {:03d} CURRENT loss for each site: '.format(epoch) + str(test_loss_site))
      # print('At epoch: {:03d} BEST accuracy for each site: '.format(epoch) +  best_acc_site)
      # print('At epoch: {:03d} CURRENT accuracy for each site: '.format(epoch) + acc_site)
      print('-----------------------------------------------------')


 

      

if __name__ == "__main__":
  main()
