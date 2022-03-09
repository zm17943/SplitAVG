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


model_names = ['res34', 'mobile']

acc_site, best_acc_site, test_loss_site = {}, {}, {}


parser = argparse.ArgumentParser(description="PyTorch DDCNN")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--num_class", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
parser.add_argument("--fineSize", type=int, default=224, help="The size of processed image")
parser.add_argument("--loadSize", type=int, default=256, help="The size of original image")
parser.add_argument("--eval_freq", type=int, default=1, help="Validation frequent")
parser.add_argument("--save_best", type=bool, default=True, help="If save the best validation model")
parser.add_argument("--epoch_num", type=str, default="60", help="The number of training round")
parser.add_argument("--iter_per_epoch", type=str, default="80", help="The number of batch forward/backward propagation per training round")
parser.add_argument("--train_file", type=str, default="./data/boneS1.h5", help="The path of training data split")
parser.add_argument("--val_file", type=str, default="./data/val.h5", help="The path of validation data")
parser.add_argument('--site_num', default=4, type=int, help="The total number of participating local sites")
parser.add_argument('--sample_num', default=4, type=int, help="The number of local sites sampled in each round")
parser.add_argument('--seed', default=2556, type=int)
parser.add_argument('--arch', default='res34', choices=model_names, help='model architecture')
parser.add_argument('--cut', default='conv1', help='The name of cut layer, specified according to the used model')


def split_net(nets, server_net):
  for client_net in nets:
    if_server_net = False
    for name0, midlayer0 in client_net._modules.items():
      if if_server_net:
        for param in midlayer0.parameters():
          param.requires_grad = False
      else:
        for param in server_net._modules[name0].parameters():
          param.requires_grad = False
      if (name0 == name):
        if_server_net = True
  return nets, server_net

def setup_models(arg):
  nets = []
  acc_site, test_loss_site = {}
  for site in range(arg.site_num):
    if (arch == "res34"):
      nets.append(models.resnet34(pretrained=True))
      nets[-1].fc = nn.Linear(512, arg.num_class)
    if (arch == "mobile"):
      nets.append(models.mobilenet_v2(pretrained=True))
      nets[-1].classifier[1] = nn.Linear(1280, arg.num_class)
    acc_site[str(site)] = 0
    test_loss_site[str(site)] = 0
    best_acc_site[str(site)] = 0

  if (arch == "res34"):
    server_net = models.resnet34(pretrained=True)
    server_net[-1].fc = nn.Linear(512, arg.num_class)
  if (arch == "mobile"):
    server_net = models.mobilenet_v2(pretrained=True)
    server_net[-1].classifier[1] = nn.Linear(1280, arg.num_class)

  # Leave only client sub-network and server sub-network trainable
  nets, server_net = split_net(nets, server_net)
  return nets, server_net.to(device)


def main():

  # Set the random seed
  random.seed(arg.seed)
  np.random.seed(arg.seed)
  torch.manual_seed(arg.seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed(arg.seed)


  # Set the GPU device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


  # Set up local client models and server model
  nets, server_net = setup_models(arg)



  #Training
  for epoch in range(epoch_num):
    chosen = random.sample(list(range(arg.site_num)), arg.sample_num)
    chosen_net = [nets[index].to(device) for index in chosen]
    
    # Load data splits from local sites
    generators = get_data_loader_for_chosen(arg, chosen)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()
    optimizers = [torch.optim.SGD(net.parameters(), lr=arg.lr, momentum=0.9) for net in chosen_net]
    optimizer_server = torch.optim.SGD(server_net.parameters(), lr=arg.lr, momentum=0.9)

    running_loss = 0
    for iteration in range(arg.iter_per_epoch):
      running_loss = splitavg_propagation(arg, running_loss, chosen_net, generators, device, criterion, optimizers, optimizer_server)


    print('At epoch: {:03d} Step: {:03d}*{:03d} AVERAGE TRAIN loss : {:.4f}'.
                  format(epoch, epoch, arg.iter_per_epoch, running_loss))


    if (epoch % arg.eval_freq == 0):
      test_set_loader, test_len = get_data_loader_for_evaluation(arg)
      val(arg, epoch, acc_site, best_acc_site, test_loss_site, nets, server_net, test_set_loader, test_len, device, criterion)
      print('-----------------------------------------------------')
      print('At epoch: {:03d} CURRENT loss for each site: '.format(epoch) + test_loss_site)
      print('At epoch: {:03d} BEST accuracy for each site: '.format(epoch) +  best_acc_site)
      print('At epoch: {:03d} CURRENT accuracy for each site: '.format(epoch) + acc_site)
      print('-----------------------------------------------------')

 

      

if __name__ == "__main__":
  args = parser.parse_args()
  main()