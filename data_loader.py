import h5py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data=None, target=None, imageSize=None, loadSize=None, transform=None):
        self.data = data
        self.target = target
        self.imageSize = imageSize
        self.loadSize = loadSize

        
    def __getitem__(self, index):

        x = self.data[index]
        y = self.target[index]
        x = x.transpose((2,0,1))
        r = np.random.randint(0, loadSize-imageSize)
        r2 = np.random.randint(0, loadSize-imageSize)
        x = x[:, r:(self.imageSize+r), r2:(self.imageSize+r2)]

        rotateNum = np.random.randint(0, 4)

        for j in range(3):
            for i in range(rotateNum):
                x[j] = np.rot90(x[j])

        if random.random() < 0.5:
          x = np.fliplr(x).copy()
        else: 
          x = np.array(x)
        
        return torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))
    
    def __len__(self):
        return len(self.data)


class MyDataset_test(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):

        x = self.data[index]
        y = self.label[index]
        x = x.transpose((2,0,1))
        x = x[:, 16:240, 16:240]
        
        return torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))
    
    def __len__(self):
        return len(self.data)


def get_data_loader_for_chosen(arg, chosen):
    data_file = h5py.File(arg.train_file, 'r')['examples']
    generators = []
    for client in chosen:
      train_set = MyDataset(data_file[str(client)]['pixels'], data_file[str(client)]['label'], arg.fineSize, arg.loadSize)
      train_set_loader = DataLoader(dataset=train_set, batch_size=arg.batch_size, shuffle=True, drop_last=True)
      generators.append(iter(train_set_loader))
    return generators


def get_data_loader_for_evaluation(arg):
    data_file = h5py.File(arg.val_file, 'r')['examples']
    test_set = MyDataset_test(data_file['pixels'], data_file['label'])
    test_set_loader = DataLoader(dataset=test_set, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    return test_set_loader, len(data_file['label'])




