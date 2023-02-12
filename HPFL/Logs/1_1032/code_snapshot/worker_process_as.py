import torch
import torchvision
import numpy as np
import datasource
import modellib
from datetime import datetime
from as_manager import *
import copy, argparse, time, sys, os, random
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--master_address', '-m', type=str, default='127.0.0.1')
parser.add_argument('--world_size', '-w', type=int, default=5)
parser.add_argument('--rank', '-r', type=int, default=0)
parser.add_argument('--trial_no', type=int, default=0)
parser.add_argument('--remarks', type=str, default='Remarks Missing...')

args = parser.parse_args()
MASTER_ADDRESS = args.master_address
WORLD_SIZE = args.world_size
RANK = args.rank

## Suggested hyper-parameter value of (LEARNING_RATE, WEIGHT_DECAY) for each model
HyperParams = { 
        'CNN': ('Mnist', 0.01, 0.01),
        'LogisticRegression': ('Mnist', 0.01, 0.0001),
        'SVM': ('Points', 0.01, 0.0000),
        'FixupResNet': ('Cifar10', 0.1, 0.0001),
        'ResNet': ('Cifar10', 0.1, 0.001),
        'VGG16': ('ImageNet', 0.1, 0.0005),
        'DenseNet121': ('ImageNet', 0.1, 0.0001),
        'AlexNet': ('ImageNet', 0.1, 0.0001),
        'LSTM': ('KWS', 0.05, 0.01),
        'LSTM_NLP': ('ag_news', 0.05, 0.01)
        }


MODEL, D_ALPHA, PERSONALIZED_RATIO, IS_INDEPENDENT = 'CNN', 0.1, 0.5, True
CLIENT_EDGE_SYNC_FREQ = 5
EDGE_SERVER_SYNC_FREQ = 10
CLUSTERING_FREQ = 5
GROUP_NUM = 16
EDGE_SERVER_GROUP_NUM = 4
SAMPLE_NUM_PER_CLIENT = 100
GROUP_SIZE = WORLD_SIZE // GROUP_NUM # number of clients in a edge server


BATCH_SIZE = 50
DATASET, LEARNING_RATE, WEIGHT_DECAY = HyperParams[MODEL]
MAX_ROUND = 200
CHECKPOINT_ENABLED = False
CUDA = torch.cuda.is_available()
if CUDA:
    torch.cuda.set_device(RANK % torch.cuda.device_count())

def logging(string):
    print(str(datetime.now())+' '+str(string))
    sys.stdout.flush()

def save_model(model, round_id):
    state = {
        'state': model.state_dict(),
        'round': round_id,
        }
    checkpoint_name = 'autoencoder-' + str(round_id) + '.t7'
    logging('checkpoint name: '+str(checkpoint_name))
    if not os.path.exists(checkpoint_name):
        torch.save(state, checkpoint_name)
        logging('## Model saved at round' + str(round_id))

def check_lr_decay(epoch_id, optimizer):
    if epoch_id % 100 == 0:
        logging('decay learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def test(test_loader, model):
    accuracy = 0
    positive_test_number = 0
    total_test_number = 0
    #test_h = model.init_hidden(BATCH_SIZE)
    for step, (test_x, test_y) in enumerate(test_loader):
        #logging("test_y: " + str(test_y))
        if CUDA:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        #test_x = test_x.type(torch.cuda.FloatTensor)
        test_output = model(test_x)
        if MODEL != 'SVM':
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        else:
            test_output = test_output.data.cpu().numpy()
            pred_y = np.where(test_output>0, np.ones(test_output.shape).astype(int), -1*np.ones(test_output.shape).astype(int)).reshape(-1)
        positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
        total_test_number += float(test_y.size(0))
    accuracy = positive_test_number / total_test_number
    return accuracy

def train():

    train_loader, test_loader, train1_loader = datasource.ML_Dataset(DATASET, WORLD_SIZE, RANK, BATCH_SIZE, GROUP_NUM, GROUP_SIZE, SAMPLE_NUM_PER_CLIENT, D_ALPHA, IS_INDEPENDENT).get_dataloaders()

    model = modellib.load_model(MODEL, DATASET, CHECKPOINT_ENABLED)
    local_model = modellib.load_model(MODEL, DATASET, CHECKPOINT_ENABLED)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_func = torch.nn.CrossEntropyLoss()
    local_loss_func = torch.nn.CrossEntropyLoss()
    logging('\n\n ----- start training -----')

    as_manager = AS_Manager(model, local_model, MASTER_ADDRESS, WORLD_SIZE, RANK, LEARNING_RATE, CLIENT_EDGE_SYNC_FREQ, EDGE_SERVER_SYNC_FREQ, CLUSTERING_FREQ, GROUP_NUM, GROUP_SIZE, EDGE_SERVER_GROUP_NUM, PERSONALIZED_RATIO)
    
		
    iter_id = 0
    epoch_id = 0


    while as_manager.edge_server_round_id < MAX_ROUND:  #train two model, local model and global model          
        logging('\n\n--- start epoch '+ str(epoch_id) + ' ---')
        #h = model.init_hidden(BATCH_SIZE)
             
        count = [0]*200
        for step, (b_x, b_y) in enumerate(train_loader):
            gc.collect()
            #global model
            #logging("b_y: " + str(b_y))
            for i in range(len(b_y)):
                index = int(b_y[i])
                count[index] += 1
            if CUDA:
                b_x = b_x.cuda()
                b_y = b_y.cuda()                 
						

            optimizer.zero_grad()
            output = model(b_x)

            if MODEL != 'SVM':
                loss = loss_func(output, b_y)
            else:
                loss = torch.mean(torch.clamp(1 - output.t() * b_y.float(), min=0))  # hinge loss
                # loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
            loss.backward()
            optimizer.step()
			
            #local model    					

            local_optimizer.zero_grad()
            local_output = local_model(b_x)

            if MODEL != 'SVM':
                local_loss = local_loss_func(local_output, b_y)
            else:
                local_loss = torch.mean(torch.clamp(1 - local_output.t() * b_y.float(), min=0))  # hinge loss
                # loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
            local_loss.backward()
            local_optimizer.step()
                      			
            for i, (p_global, p_local) in enumerate(zip(model.parameters(), local_model.parameters())): # get new local pesonalized model 
                p_local.data = as_manager.personalized_ratio * p_local.data + (1-as_manager.personalized_ratio) * p_global.data
            		
            iter_id += 1		

            if as_manager.sync_client_edge(model, local_model, iter_id):  # sync among clients in each edge server
                
                if epoch_id != as_manager.last_test_client_edge_epoch_id and epoch_id != 0:
                    accuracy = test(test_loader, local_model)
                    #train_accuracy = test(train1_loader, model)
                    logging('client-edge test accuracy:' + str(accuracy) + '; client-edge round_id:' + str(as_manager.client_edge_round_id) + '; epoch_id:' + str(epoch_id) + '; iter_id:' + str(iter_id) + '; loss: ' + str(loss) + '; local loss: ' + str(local_loss))				
                as_manager.last_test_client_edge_epoch_id = epoch_id
            		
            #if as_manager.sync_edge_server(model, local_model, iter_id):  # syc among edge servers and the server, applied in FedAvg and APFL  
            if as_manager.sync_among_edge(model, local_model, iter_id):    # sync among edge servers, our method
                if epoch_id != as_manager.last_test_edge_server_epoch_id and epoch_id != 0 and RANK % GROUP_SIZE == 0:
                    accuracy = test(test_loader, local_model)
                    accuracy_global = test(test_loader, model)
                    #train_accuracy = test(train1_loader, model)
                    logging('edge-server test accuracy:' + str(accuracy) + '; edge-server round_id:' + str(as_manager.edge_server_round_id) + '; epoch_id:' + str(epoch_id) + '; iter_id:' + str(iter_id) + '; loss: ' + str(loss) + '; local loss: ' + str(local_loss))	
                    logging('edge-server global test accuracy:' + str(accuracy_global))		
                as_manager.last_test_edge_server_epoch_id = epoch_id
            				
            if as_manager.sync_clustering(model, iter_id):
                if RANK % GROUP_SIZE == 0:
                    logging("new edge server groups: " + str(as_manager.kmeans_label))
                    logging("new edge server groups index: " + str(as_manager.local_edge_group_list))
                    logging("new local edge label: " + str(as_manager.local_edge_label))
            		
        epoch_id += 1
        as_manager.epoch_id = epoch_id


if __name__ == "__main__":

    logging('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks)
    logging('\nInitialization:\n\t model: ' + MODEL + '; dataset: ' + DATASET + '; batch_size: ' + str(BATCH_SIZE) + '; d_alpha: ' + str(D_ALPHA) 
        + '\n\t master_address: ' + str(MASTER_ADDRESS) + '; world_size: '+str(WORLD_SIZE) + '; rank: '+ str(RANK) 
        + '; weight_decay: ' + str(WEIGHT_DECAY)
        + ';\n\t cleint_edge_sync_frequency: 1/'+str(CLIENT_EDGE_SYNC_FREQ)
        + ';\n\t cleint_edge_sync_frequency: 1/'+str(EDGE_SERVER_SYNC_FREQ)		+ '.\n')

    train()
