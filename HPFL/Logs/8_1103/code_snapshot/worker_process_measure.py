import torch
import torchvision
import numpy as np
import datasource
import modellib
from datetime import datetime
import copy, argparse, time, sys, os, random
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist
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

HyperParams = { 
        # MODEL: (LEARNING_RATE, WEIGHT_DECAY)
        'CNN': (0.01, 0.01),
        'SVM': (0.01, 0.0000),
        #'VGG16': (0.01, 0.0001),
        'VGG16': (0.1, 0.01),
        'ResNet18': (0.01, 0.0001)
        }
# CNNMnist ## set model [CNN, VGG16, DenseNet121, AlexNet, ResNet] and dataset [Mnist, Cifar10, ImageNet]
MODEL, DATASET, BATCH_SIZE, D_ALPHA = 'SVM', 'Points', 100, 0.01
MODEL, DATASET, BATCH_SIZE, D_ALPHA = 'CNN', 'Cifar10', 100, 100.0
LEARNING_RATE, WEIGHT_DECAY = HyperParams[MODEL]
SYNC_FREQ, TEST_FREQ = 500, 500

MAX_ROUND = 20000
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
    for step, (test_x, test_y) in enumerate(test_loader):
        if CUDA:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        test_output = model(test_x)
        if MODEL != 'SVM':
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        else:
            test_output = test_output.data.cpu().numpy()
            # pred_y = torch.max(torch.Tensor(test_output), 1)[1].data.cpu().numpy()
            pred_y = np.where(test_output>0, np.ones(test_output.shape).astype(int), -1*np.ones(test_output.shape).astype(int)).reshape(-1)
        positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
        total_test_number += float(test_y.size(0))
    accuracy = positive_test_number / total_test_number
    return accuracy


def train():

    global SYNC_FREQ
    train_loader, test_loader = datasource.ML_Dataset(DATASET, WORLD_SIZE, RANK, BATCH_SIZE, D_ALPHA).get_dataloaders()
    model = modellib.load_model(MODEL, DATASET, CHECKPOINT_ENABLED)

    dist.init_process_group(backend='nccl' if CUDA else 'tcp', init_method=MASTER_ADDRESS, world_size=WORLD_SIZE, rank=RANK)
    group = dist.new_group([i for i in range(WORLD_SIZE)])
    if CUDA:
        model.cuda()
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_func = torch.nn.CrossEntropyLoss()


    logging('initial model parameters: ')
    logging(list(model.parameters())[0][0][0])
    logging('\n\n ----- start training -----')

    iter_id = 0
    epoch_id = 0
    round_id = 0
    last_test_epoch_id = -1

    while round_id < MAX_ROUND:            
        logging('\n\n--- start epoch '+ str(epoch_id) + ' ---')
        
        for step, (b_x, b_y) in enumerate(train_loader):
            gc.collect()
            if CUDA:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            # print(b_y)

            optimizer.zero_grad()
            output = model(b_x)
            if MODEL != 'SVM':
                loss = loss_func(output, b_y)
            else:
                loss = torch.mean(torch.clamp(1 - output.t() * b_y.float(), min=0))  # hinge loss
            loss.backward()
            optimizer.step()

            if iter_id % SYNC_FREQ == 0:
                for (i, p) in enumerate(model.parameters()):
                    dist.all_reduce(p.data, op=dist.reduce_op.SUM, group=group)
                    p.data /= WORLD_SIZE

                if iter_id % TEST_FREQ == 0 and RANK == 0:
                    accuracy = test(test_loader, model)
                    logging(' - test - accuracy:' + str(accuracy) + '; round_id:' + str(round_id) + '; epoch_id:' + str(epoch_id) + '; iter_id:' + str(iter_id) )
                    # logging(' - test - accuracy:' + str(accuracy) + '; round_id:' + str(round_id) + '; epoch_id:' + str(epoch_id) + '; iter_id:' + str(iter_id) + 'parameter-0: ' + str(float(list(model.parameters())[0][0][0])) + '; parameter-1: ' + str(float(list(model.parameters())[0][0][1])))

                logging('finish round: ' + str(round_id) +'; at iter_id: ' + str(iter_id))
                round_id += 1

            # logging('finish iter: ' + str(iter_id))
            iter_id += 1

        # np.save(str(epoch_id), list(model.parameters())[0].detach().cpu().numpy())
        logging('finish local epoch: ' + str(epoch_id) + '; at round_id: ' + str(round_id) + '; at iter_id: ' + str(iter_id) + '; sync_frequency: 1/' +str(SYNC_FREQ))
        epoch_id += 1


if __name__ == "__main__":

    logging('Trial ID: ' + str(args.trial_no))
    logging('Exp Setup Remarks: ' + args.remarks)
    logging('\nInitialization:\n\t model: ' + MODEL + '; dataset: ' + DATASET + '; batch_size: ' + str(BATCH_SIZE) + '; d_alpha: ' + str(D_ALPHA)
        + '\n\t master_address: ' + str(MASTER_ADDRESS) + '; world_size: '+str(WORLD_SIZE) + '; rank: '+ str(RANK) 
        + '\n\t learning_rate: ' + str(LEARNING_RATE) + '; weight_decay: ' + str(WEIGHT_DECAY)
        + ';\n\t init_sync_frequency: 1/'+str(SYNC_FREQ) + '.\n')

    train()
