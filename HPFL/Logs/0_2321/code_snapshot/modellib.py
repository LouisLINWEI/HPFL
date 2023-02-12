import os
import torch
from models import CNN, LSTM, ResNet, VGG, AlexNet, DenseNet, SVM
from datetime import datetime

def logging(string):
    print(str(datetime.now())+' '+str(string))
    sys.stdout.flush()

def load_model(MODEL, DATASET, CHECKPOINT_ENABLED):
    if MODEL == 'CNN' and DATASET == 'Mnist':
        model = CNN.CNN_Mnist()
    if MODEL == 'CNN' and DATASET == 'Cifar10':
        model = CNN.LeNet_Cifar10()
    if MODEL == 'CNN' and DATASET == 'KWS':
        model = CNN.CNN_KWS()
    if MODEL == 'CNN' and DATASET == 'HAR':
        model = CNN.CNN_HAR()
    if MODEL == 'CNN' and DATASET == 'EMnist':
        model = CNN.CNN_EMnist()
    if MODEL == 'LSTM' and DATASET == 'KWS':
        model = LSTM.LSTM_KWS()
    if MODEL == 'LSTM' and DATASET == 'HAR':
        model = LSTM.LSTM_HAR()
    if MODEL == 'SVM':
        model = SVM.LinearSVM()
    if MODEL == 'ResNet' and DATASET == 'Cifar10':
        model = ResNet.ResNet18_Cifar10()
    if MODEL == 'VGG16' and DATASET == 'ImageNet':
        model = VGG.VGG16_Cifar10()
    if MODEL == 'AlexNet' and DATASET == 'ImageNet':
        model = AlexNet.AlexNet_ImageNet() 
    if MODEL == 'DenseNet121' and DATASET == 'ImageNet':
        model = DenseNet.DenseNet121_ImageNet() 
    if MODEL == 'LSTM_NLP' and DATASET == 'ag_news':
        model = LSTM.LSTM_NLP() 
    if CHECKPOINT_ENABLED:
        if  os.path.exists('autoencoder-0.t7'):
            logging('===> Resume from checkpoint')
            checkpoint = torch.load('autoencoder-0.t7')
            model.load_state_dict(checkpoint['state'])
        else:
            logging('model created')
            save_model(model, 0)
    return model
