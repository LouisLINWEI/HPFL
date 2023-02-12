import torch
import copy
from datetime import datetime
import sys
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist
import numpy as np
from sklearn.cluster import KMeans

CUDA = torch.cuda.is_available()
print('cuda' + str(CUDA))

def logging(string):
    print(str(datetime.now())+' '+str(string))
    sys.stdout.flush()

class AS_Manager:
    def __init__(self, model, local_model, master_address, world_size, rank, learning_rate, client_edge_sync_frequency, edge_server_sync_frequency, clustering_frequency, group_num, group_size, edge_server_group_num, personalized_ratio):
        dist.init_process_group(backend='gloo', init_method=master_address, world_size=world_size, rank=rank)
        group = dist.new_group([i for i in range(world_size)])

        self.client_edge_sync_frequency = client_edge_sync_frequency
        self.edge_server_sync_frequency = edge_server_sync_frequency
        self.clustering_frequency = clustering_frequency
        self.group = group
        self.world_size = world_size
        self.rank = rank
        self.model = model
        self.local_model = local_model
        self.personalized_ratio = personalized_ratio
        self.client_edge_round_id = 0
        self.edge_server_round_id = 0
        self.clustering_round_id = 0
        self.epoch_id = 0
        self.last_test_client_edge_epoch_id = 0
        self.last_test_edge_server_epoch_id = 0
        self.learning_rate = learning_rate
        self.group_num = group_num
        self.group_size = group_size
        self.edge_server_group_num = edge_server_group_num
        self.local_group = []
        self.local_edge_group = []
        self.local_edge_group_list = []
        self.kmeans_label = torch.zeros(self.group_num, dtype=torch.int32)
        self.local_edge_label = -1

        logging("edge_server_group: " + str([i for i in range(0, self.world_size, self.group_size)]))
        self.edge_server_group = dist.new_group([i for i in range(0, self.world_size, self.group_size)])
        for i in range(self.group_num):
            print(str([i for i in range(i * self.group_size, (i+1) * self.group_size)]))
            local_group_tmp = dist.new_group([i for i in range(i * self.group_size, (i+1) * self.group_size)])
            self.local_group.append(local_group_tmp)
        #logging('edge_server_group: ' + str([i for i in range(0, self.world_size, self.group_size)]))
        #logging('all_local_group: ' + str(self.local_group))
        #logging('local_group: ' + str(self.local_group[self.rank//self.group_size]))

        for param in model.parameters():
            dist.broadcast(param.data, src=0, group=group)
        for param in local_model.parameters():
            dist.broadcast(param.data, src=0, group=group)

        self.last_model = copy.deepcopy(model) # record the model checkpoint after the last synchronization
        self.last_local_model = copy.deepcopy(local_model) 

        self.next_client_edge_sync_iter_id = self.client_edge_sync_frequency
        self.next_edge_server_sync_iter_id = self.client_edge_sync_frequency * self.edge_server_sync_frequency
        self.next_clustering_sync_iter_id = self.client_edge_sync_frequency * self.edge_server_sync_frequency - 1
        self.world_size = world_size
        self.gathered_parameters = []
        self.gathered_parameters_tmp = []
        self.model_size = 0

        for p in model.parameters():
            if rank == 0:
                self.gathered_parameters.append([copy.deepcopy(p.data) for i in range(world_size)])
                self.model_size += p.data.numel()
            else:
                self.gathered_parameters.append([])
                self.model_size += p.data.numel()

        if CUDA:
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            self.model.cuda()
            self.local_model.cuda()
        logging('model size: ' + str(self.model_size))

        self.last_round_updates = np.zeros(self.model_size)

    def cluster_edge_server(self, grad_list):
        for i in range(len(grad_list)):
            if i == 0:
                grad_list_temp = np.array(grad_list[0])
            else:
                grad_list_temp = np.vstack((grad_list_temp, np.array(grad_list[i])))
        #logging("shape: " + str(grad_list_temp.shape))
        #logging("data: " + str(grad_list_temp))
        #logging("num: " + str(self.edge_server_group_num))
        self.kmeans = KMeans(n_clusters=self.edge_server_group_num, random_state=0).fit(grad_list_temp)
        #logging("label: " + str(self.kmeans.labels_))

    def sync_client_edge(self, model, local_model, iter_id): # sync among clients in each edge server
        if iter_id == self.next_client_edge_sync_iter_id:
            if CUDA:
                model.cpu()
                local_model.cpu()
            #logging('des: ' + str(int(self.rank//self.group_size*self.group_size)))
            for (i, p) in enumerate(model.parameters()):
                if self.rank == int(self.rank//self.group_size*self.group_size):
                    grad_list = [torch.zeros_like(p.data) for _ in range(self.group_size)]
                else:
                    grad_list = []
                dist.gather(p.data, gather_list=grad_list, dst=int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
                if self.rank % self.group_size == 0:
                    p.data = sum(grad_list) / self.group_size # reduce to average
                dist.broadcast(p.data, int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
				
            for (i, p) in enumerate(local_model.parameters()):
                if self.rank == int(self.rank//self.group_size*self.group_size):
                    grad_list = [torch.zeros_like(p.data) for _ in range(self.group_size)]
                else:
                    grad_list = []
                dist.gather(p.data, gather_list=grad_list, dst=int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
                if self.rank % self.group_size == 0:
                    p.data = sum(grad_list) / self.group_size # reduce to average
                dist.broadcast(p.data, int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])


            self.next_client_edge_sync_iter_id = iter_id + self.client_edge_sync_frequency
            self.client_edge_round_id += 1
            self.last_model = copy.deepcopy(model)
            self.last_local_model = copy.deepcopy(local_model)
            if CUDA:
                model.cuda()
                local_model.cuda()
            return True
        return False

    def sync_among_edge(self, model, local_model, iter_id): # sync among edge servers, our method
        if iter_id == self.next_edge_server_sync_iter_id:
            if CUDA:
                model.cpu()
                local_model.cpu()
            for (i, p) in enumerate(model.parameters()):
                if self.rank % self.group_size == 0 and self.rank == self.local_edge_group_list[self.local_edge_label][0]:
                    grad_list = [torch.zeros_like(p.data) for _ in range(len(self.local_edge_group_list[self.local_edge_label]))]
                else:
                    grad_list = []
                dist.gather(p.data, gather_list=grad_list, dst=self.local_edge_group_list[self.local_edge_label][0], group=self.local_edge_group[self.local_edge_label])
                
                
                if self.rank == self.local_edge_group_list[self.local_edge_label][0]:
                    p.data = sum(grad_list) / len(self.local_edge_group_list[self.local_edge_label]) # reduce to average
                dist.broadcast(p.data, self.local_edge_group_list[self.local_edge_label][0], group=self.local_edge_group[self.local_edge_label])
                            
            if self.rank % self.group_size == 0:
                for i, (p_global, p_local) in enumerate(zip(model.parameters(), local_model.parameters())):						
                    p_local.data = self.personalized_ratio * p_local.data + (1-self.personalized_ratio) * p_global.data
					
            for (i, p) in enumerate(model.parameters()):
                dist.broadcast(p.data, int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
            for (i, p) in enumerate(local_model.parameters()):
                dist.broadcast(p.data, int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
            
            self.next_edge_server_sync_iter_id = iter_id + self.client_edge_sync_frequency * self.edge_server_sync_frequency
            self.edge_server_round_id += 1
            if CUDA:
                model.cuda()
                local_model.cuda()
            return True
        return False


    def sync_edge_server(self, model, local_model, iter_id): # syc among edge servers and the server, applied in FedAvg and APFL  
        if iter_id == self.next_edge_server_sync_iter_id:
            if CUDA:
                model.cpu()
                local_model.cpu()
            for (i, p) in enumerate(model.parameters()):
                if self.rank == 0:
                    grad_list = [torch.zeros_like(p.data) for _ in range(self.group_num)]
                else:
                    grad_list = []
                dist.gather(p.data, gather_list=grad_list, dst=0, group=self.edge_server_group)
                if self.rank == 0:
                    p.data = sum(grad_list) / self.group_num # reduce to average
                dist.broadcast(p.data, 0, group=self.edge_server_group)
            
            if self.rank % self.group_size == 0:
                for i, (p_global, p_local) in enumerate(zip(model.parameters(), local_model.parameters())):						
                    p_local.data = self.personalized_ratio * p_local.data + (1-self.personalized_ratio) * p_global.data
				

					
            for (i, p) in enumerate(model.parameters()):
                dist.broadcast(p.data, int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
            for (i, p) in enumerate(local_model.parameters()):
                dist.broadcast(p.data, int(self.rank//self.group_size*self.group_size), group=self.local_group[self.rank//self.group_size])
            

            self.next_edge_server_sync_iter_id = iter_id + self.client_edge_sync_frequency * self.edge_server_sync_frequency
            self.edge_server_round_id += 1
            self.last_model = copy.deepcopy(model)
            self.last_local_model = copy.deepcopy(local_model)
            if CUDA:
                model.cuda()
                local_model.cuda()
            return True
        return False
		
    def sync_clustering(self, model, iter_id): # cluster edge servers into group for edge servers
        if iter_id == self.next_clustering_sync_iter_id:
            if CUDA:
                model.cpu()
            for (i, p) in enumerate(model.parameters()):
                if i == 5:
                    if self.rank == 0:
                        grad_list = [torch.zeros_like(p.data) for _ in range(self.group_num)]
                    else:
                        grad_list = []
                    dist.gather(p.data, gather_list=grad_list, dst=0, group=self.edge_server_group)
					
            if self.rank == 0:
                self.cluster_edge_server(grad_list)
                self.kmeans_label = torch.tensor(self.kmeans.labels_)					
                            
            dist.broadcast(self.kmeans_label, 0, group=self.group)
			
            #logging("kmeans label: " + str(self.kmeans_label))
            for i in range(self.edge_server_group_num):
                tmp = []
                for j in range(len(self.kmeans_label)):
                    if i == self.kmeans_label[j]:
                        tmp.append(int(j*self.group_size))
                #logging("tmp: " + str(tmp))
                local_edge_group_tmp = dist.new_group(tmp)
                self.local_edge_group_list.append(tmp)
                self.local_edge_group.append(local_edge_group_tmp)
				
            if self.rank % self.group_size == 0:
                self.local_edge_label = self.kmeans_label[self.rank // self.group_size]
			
            self.next_clustering_sync_iter_id = iter_id + self.client_edge_sync_frequency * self.edge_server_sync_frequency * self.clustering_frequency
            self.clustering_round_id += 1
            if CUDA:
                model.cuda()
            return True
        return False