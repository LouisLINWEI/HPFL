# encoding=utf-8
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import random
import torchvision
import os
import sys
import datetime

def logging(string):
    print(str(datetime.datetime.now())+' '+str(string))
    sys.stdout.flush()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class SVMconstructor(Dataset):
    def __init__(self, transform):
        X, Y = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=5.0)
        X = (X - X.mean()) / X.std()
        Y[np.where(Y == 0)] = -1
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        data = []
        for i in range(len(X)):
            data.append((X[i], Y[i]))
        self.data = data
        self.transform = transform
        self.targets = Y

    def __getitem__(self, index):
        point, label = self.data[index]
        return point, label

    def __len__(self):
        return len(self.data)
		
class HARconstructor(Dataset):
    def __init__(self, samples, labels, t):
        self.data = samples
        self.targets = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.data)

class KWSconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0], int(info[1])) )
            self.targets.append(int(info[1]))
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        feature = np.reshape(feature, (50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)
		
class ag_newsconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0], int(info[1])) )
            self.targets.append(int(info[1]))
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        #feature = np.reshape(feature, (50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)
		
def HAR_load_data():
    data = np.load('./HAR/data_har.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)

def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]

class ML_Dataset(): 
    def __init__(self, dataset_name, world_size, rank, batch_size, group_num, group_size, sample_num_per_client, d_alpha=100.0, is_independent=True):
        self.dataset_name = dataset_name
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.d_alpha = d_alpha
        self.group_num = group_num
        self.group_size = group_size
        self.is_independent = is_independent
        self.train_data, self.test_data, self.class_num = self.get_datasets(self.dataset_name)
        #self.local_size = len(self.train_data) / self.world_size
        #self.local_train_size = len(self.train_data) / self.group_num
        self.local_train_size = sample_num_per_client
        self.local_test_size = len(self.test_data) / (self.group_num // 4)
        self.train_idxs, self.test_idxs = self.get_idxs()

    # Use Dirichlet distribution to partition dataset
    def get_train_composition(self): # generate train data (varied distribution) for clients
        tp_list = []
        for i in range(self.group_num // 4):
            tp_list.append(self.d_alpha)
        tp_list[self.rank // (self.group_size * 4)] = 100.0
        tp_list = tuple(tp_list)
        composition_ratio = np.random.dirichlet(tp_list)
        return (composition_ratio*self.local_train_size).astype(int)
		
    def get_test_composition(self):  # generate test data (varied distribution) for clients
        tp_list = []
        for i in range(self.group_num // 4):
            tp_list.append(self.d_alpha)
        tp_list[self.rank // (self.group_size * 4)] = 100.0
        tp_list = tuple(tp_list)
        composition_ratio = np.random.dirichlet(tp_list)
        return (composition_ratio*self.local_test_size).astype(int)

    def get_idxs(self):
        local_train_idxs = []
        local_test_idxs = []
        self.set_seed(0)
        if self.is_independent == True: # for large scale exp: samples on each client is independently sampled from respective class pools
            train_labels = np.array(self.train_data.targets)
            test_labels = np.array(self.test_data.targets)
            sorted_train_idxs = np.argsort(train_labels)
            sorted_test_idxs = np.argsort(test_labels)
            composition = self.get_train_composition()
            composition_test = self.get_test_composition()
            print('local train dataset composition: ' + str(composition))
            print('local test dataset composition: ' + str(composition_test))
            #class_pool_size = len(self.train_data) / self.class_num
            train_class_pool_size = len(self.train_data) / (self.group_num // 4)
            test_class_pool_size = len(self.test_data) / (self.group_num // 4)
            for i in range(len(composition)):
                temp = random.sample(list(sorted_train_idxs[int(train_class_pool_size)*i : int(train_class_pool_size)*(i+1)]),composition[i])
                for j in range(composition[i]):
                    #sample_index = sorted_idxs[int(class_pool_size*random.random()) + int(class_pool_size)*i] # randomly sampling
                    # sample_index = sorted_idxs[(class_pool_size/self.world_size*self.rank+j) % class_pool_size + class_pool_size*i]
                    #local_idxs.append(sample_index)
                    local_train_idxs.append(temp[j])
            for i in range(len(composition_test)):
                temp = random.sample(list(sorted_test_idxs[int(test_class_pool_size)*i : int(test_class_pool_size)*(i+1)]),composition_test[i])
                for j in range(composition_test[i]):
                    #sample_index = sorted_idxs[int(class_pool_size*random.random()) + int(class_pool_size)*i] # randomly sampling
                    # sample_index = sorted_idxs[(class_pool_size/self.world_size*self.rank+j) % class_pool_size + class_pool_size*i]
                    #local_idxs.append(sample_index)
                    local_test_idxs.append(temp[j])
        else:
            train_labels = np.random.rand(len(self.train_data)) if self.d_alpha >= 1 else np.array(self.train_data.targets) # alpha>1:IID; alpha<1:non-IID
            test_labels = np.random.rand(len(self.test_data)) if self.d_alpha >= 1 else np.array(self.test_data.targets)
            train_sorted_idxs = np.argsort(train_labels)
            test_sorted_idxs = np.argsort(test_labels)
            local_train_idxs = train_sorted_idxs[int(self.local_train_size*self.rank) : int(self.local_train_size*(self.rank+1))]
            local_test_idxs = test_sorted_idxs[int(self.local_test_size*self.rank) : int(self.local_test_size*(self.rank+1))]
        return local_train_idxs, local_test_idxs

    def get_datasets(self, dataset_name):
        cur_file_path = os.path.dirname(__file__)
        if dataset_name == 'Mnist':
            if not os.path.exists(os.path.join(cur_file_path,'/data/mnist/')):
                os.mkdir(os.path.join(cur_file_path,'/data/mnist/'))
            train_dataset = datasets.MNIST(root=os.path.join(cur_file_path,'/data/mnist/'), train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.MNIST(root=os.path.join(cur_file_path,'/data/mnist/'), train=False, transform=transforms.ToTensor())
            class_num = 10
        if dataset_name == 'EMnist':
            if not os.path.exists(os.path.join(cur_file_path,'/data/emnist/')):
                os.mkdir(os.path.join(cur_file_path,'/data/emnist/'))
            train_dataset = datasets.MNIST(root=os.path.join(cur_file_path,'/data/emnist/'), train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.MNIST(root=os.path.join(cur_file_path,'/data/emnist/'), train=False, transform=transforms.ToTensor())
            class_num = 62
        if dataset_name == 'Cifar10':
            if not os.path.exists(os.path.join(cur_file_path,'/data/cifar10/')):
                os.mkdir(os.path.join(cur_file_path,'/data/cifar10/'))
            train_dataset = datasets.CIFAR10(root=os.path.join(cur_file_path,'/data/cifar10/'), train=True, transform=transforms.ToTensor(), download=False)
            test_dataset = datasets.CIFAR10(root=os.path.join(cur_file_path,'/data/cifar10/'), train=False, transform=transforms.ToTensor())
            class_num = 10
        if dataset_name == 'KWS':
            train_dataset = KWSconstructor(root='./kws/index_train.txt', transform=None)
            test_dataset = KWSconstructor(root='./kws/index_test.txt', transform=None)
            class_num = 10
        if dataset_name == 'ag_news':
            train_dataset = ag_newsconstructor(root='./out/train_index.txt', transform=None)
            test_dataset = ag_newsconstructor(root='./out/test_index.txt', transform=None)
            class_num = 4
        if dataset_name == 'Points':
            train_dataset = SVMconstructor(transform=transforms.ToTensor())
            test_dataset = SVMconstructor(transform=transforms.ToTensor())
            class_num = 2
        if dataset_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            train_dataset = datasets.ImageFolder('/home/kaiwei/tiny-imagenet/tiny-imagenet-200/train', transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]))
            test_dataset = datasets.ImageFolder('/home/kaiwei/tiny-imagenet/tiny-imagenet-200/val', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,]))
            class_num = 200
        if dataset_name == 'HAR':
            x_train, y_train, x_test, y_test = HAR_load_data()
            x_train, x_test = x_train.reshape(
                (-1, 9, 1, 128)), x_test.reshape((-1, 9, 1, 128))
            transform = None
            train_dataset = HARconstructor(x_train, y_train, transform)
            test_dataset = HARconstructor(x_test, y_test, transform)
            class_num = 6
        return train_dataset, test_dataset, class_num

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def get_dataloaders(self):
        train_dataloader = DataLoader(DatasetSplit(self.train_data, self.train_idxs), batch_size=self.batch_size, shuffle=True,drop_last=True)
        test_dataloader = DataLoader(DatasetSplit(self.test_data, self.test_idxs), batch_size=self.batch_size, shuffle=True,drop_last=True)
        train1_dataloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        return train_dataloader, test_dataloader, train1_dataloader

