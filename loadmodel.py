import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
from utils import TSPLoss,edge_overlap,get_heat_map
import pickle
from torch.utils.data import  Dataset,DataLoader# use pytorch dataloader
from random import shuffle
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=100, help='Graph Size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='smoo')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='num of layers')
parser.add_argument('--use_smoo', action='store_true')
parser.add_argument('--EPOCHS', type=int, default=300,
                    help='epochs to train')
parser.add_argument('--topk', type=int, default=20,
                    help='top k elements per row, should equal to int Rec_Num = 20 in Search/code/include/TSP_IO.h')
parser.add_argument('--penalty_coefficient', type=float, default=2.,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=2.,
                    help='temperature for adj matrix')
parser.add_argument('--diag_penalty', type=float, default=3.,
                    help='penalty on the diag')
parser.add_argument('--rescale', type=float, default=1.,
                    help='rescale for xy plane')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
device = args.device


# tsp_instances = np.load('./data/test_tsp_instance_%d.npy'%args.num_of_nodes) # 128 instances
# from scipy.spatial import distance_matrix
# temp_dis_matrix = []
# for i in range(int(tsp_instances.shape[0])):
#     temp_dis_matrix.append(distance_matrix(tsp_instances[i],tsp_instances[i]))
# tsp_instances = np.array(temp_dis_matrix)
# NumofTestSample = tsp_instances.shape[0]
# Std = np.std(tsp_instances, axis=1)
# Mean = np.mean(tsp_instances, axis=1)


# tsp_instances = tsp_instances - Mean.reshape((NumofTestSample,1,200))

# tsp_instances = args.rescale * tsp_instances # 2.0 is the rescale

# tsp_sols = np.load('./data/test_tsp_sol_%d.npy'%args.num_of_nodes)
# total_samples = tsp_instances.shape[0]
# import json

# from models import GNN
# #scattering model
# model = GNN(input_dim=2, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
# model = model.to(device)
# from scipy.spatial import distance_matrix


# ### count model parameters
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print('Total number of parameters:')
# print(count_parameters(model))



# def coord_to_adj(dis_matrix):
#     return dis_matrix


# tsp_instances_adj = np.zeros((total_samples,args.num_of_nodes,args.num_of_nodes))
# for i in range(total_samples):
#     tsp_instances_adj[i] = coord_to_adj(tsp_instances[i])
class TSP_Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
#        tsp_instance = Data(coord=x,sol=y)
        return tuple(zip(x,y))

    def __len__(self):
        return len(self.data)

# dataset = TSP_Dataset(tsp_instances_adj,tsp_sols)
# testdata = dataset[0:] ##this is very important!
# TestData_size = len(testdata)
# batch_size = args.batch_size
# test_loader = DataLoader(testdata, batch_size, shuffle=False)
# mask = torch.ones(args.num_of_nodes, args.num_of_nodes).to(device)
# mask.fill_diagonal_(0)
# def test(loader,topk = 20):
#     avg_size = 0
#     total_cost = 0.0
#     full_edge_overlap_count = 0

#     TestData_size = len(loader.dataset)
#     Saved_indices = np.zeros((TestData_size,args.num_of_nodes,topk))
#     Saved_Values = np.zeros((TestData_size,args.num_of_nodes,topk))
#     Saved_sol = np.zeros((TestData_size,args.num_of_nodes+1))
#     Saved_pos = np.zeros((TestData_size,args.num_of_nodes,2))
#     count = 0
#     model.eval()
#     for batch in loader:
#         batch_size = batch[0].size(0)
#         xy_pos = batch[0].to(device)
#         distance_m = batch[1].to(device)
#         sol = batch[2]
#         adj = torch.exp(-1.*distance_m/args.temperature)
#         adj *= mask
#         # start here:
#         t0 = time.time()
#         output = model(xy_pos,adj)
#         t1 = time.time()
#         Heat_mat = get_heat_map(SctOutput=output,num_of_nodes=args.num_of_nodes,device = device)
#         print('It takes %.5f seconds from instance: %d to %d'%(t1 - t0,count,count + batch_size))
#         sol_indicies = torch.topk(Heat_mat,topk,dim=2).indices
#         sol_values = torch.topk(Heat_mat,topk,dim=2).values
# #        print(sol_values.size())
# #        print(batch_size)
#         Saved_indices[count:batch_size+count] = sol_indicies.detach().cpu().numpy()
#         Saved_Values[count:batch_size+count] = sol_values.detach().cpu().numpy()
#         Saved_sol[count:batch_size+count] = sol.detach().cpu().numpy()
#         Saved_pos[count:batch_size+count] = xy_pos.detach().cpu().numpy()
#         count = count + batch_size


#     return Saved_indices,Saved_Values,Saved_sol,Saved_pos


# #TSP200
# model_name = 'Saved_Models/TSP_%d/scatgnn_layer_%d_hid_%d_model_210_temp_3.500.pth'%(args.num_of_nodes,args.nlayers,args.hidden)# topk = 10
# model.load_state_dict(torch.load(model_name))
# #Saved_indices,Saved_Values,Saved_sol,Saved_pos = test(test_loader,topk = 8) # epoch=20>10 
# Saved_indices,Saved_Values,Saved_sol,Saved_pos = test(test_loader,topk = args.topk) # epoch=20>10

# print('Finish Inference!')

# import os, sys

# Q = Saved_pos
# A = Saved_sol 
# C = Saved_indices
# V = Saved_Values
# with open("1kTraning_TSP%dInstance_%d.txt"%(args.num_of_nodes,Saved_indices.shape[0]), "w") as f:
#     for i in range(Q.shape[0]):
#         for j in range(Q.shape[1]):
#             f.write(str(Q[i][j][0]) + " " + str(Q[i][j][1]) + " ")
#         f.write("output ")
#         for j in range(A.shape[1]):
#             f.write(str(int(A[i][j] + 1)) + " ")
#         f.write("indices ")
#         for j in range(C.shape[1]):
#             for k in range(args.topk):
#                 if C[i][j][k] == j:
#                     f.write("-1" + " ")
#                 else:
#                     f.write(str(int(C[i][j][k] + 1)) + " ")
#         f.write("value ")
#         for j in range(V.shape[1]):
#             for k in range(args.topk):
#                 f.write(str(V[i][j][k]) + " ")
#         f.write("\n")
#         if i == Saved_indices.shape[0] - 1:
#             break

def run_UTSP(matrix_distance, num_hidden, num_layers,temperature, topk):
    #matrix_distance is a 2D numpy array
    #args is a list of strings
    #run the UTSP
    #return the solution

    num_of_nodes = matrix_distance.shape[0]
    tsp_instances = np.array([matrix_distance])
    total_samples = tsp_instances.shape[0]
    import json
    from models import GNN
    #scattering model
    model = GNN(input_dim=num_of_nodes, hidden_dim=num_hidden, output_dim=num_of_nodes, n_layers=num_layers)
    model = model.to(device)

    dataset = TSP_Dataset(tsp_instances, None)
    testdata = dataset[0:] ##this is very important!
    TestData_size = len(testdata)
    batch_size = 1
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    mask = torch.ones(num_of_nodes, num_of_nodes).to(device)
    mask.fill_diagonal_(0)
    def test(loader,topk = 20):
        avg_size = 0
        total_cost = 0.0
        full_edge_overlap_count = 0

        TestData_size = len(loader.dataset)
        Saved_indices = np.zeros((TestData_size,num_of_nodes,topk))
        Saved_Values = np.zeros((TestData_size,num_of_nodes,topk))
        Saved_sol = np.zeros((TestData_size,num_of_nodes+1))
        Saved_pos = np.zeros((TestData_size,num_of_nodes,2))
        count = 0
        model.eval()
        for batch in loader:
            batch_size = batch[0].size(0)
            xy_pos = batch[0].to(device)
            distance_m = batch[1].to(device)
            sol = batch[2]
            adj = torch.exp(-1.*distance_m/temperature)
            adj *= mask
            # start here:
            t0 = time.time()
            output = model(xy_pos,adj)
            t1 = time.time()
            Heat_mat = get_heat_map(SctOutput=output,num_of_nodes=num_of_nodes,device = device)
            print('It takes %.5f seconds from instance: %d to %d'%(t1 - t0,count,count + batch_size))
            sol_indicies = torch.topk(Heat_mat,topk,dim=2).indices
            sol_values = torch.topk(Heat_mat,topk,dim=2).values
    #        print(sol_values.size())
    #        print(batch_size)
            Saved_indices[count:batch_size+count] = sol_indicies.detach().cpu().numpy()
            Saved_Values[count:batch_size+count] = sol_values.detach().cpu().numpy()
            Saved_sol[count:batch_size+count] = sol.detach().cpu().numpy()
            Saved_pos[count:batch_size+count] = xy_pos.detach().cpu().numpy()
            count = count + batch_size
        
        return Saved_indices,Saved_Values,Saved_sol,Saved_pos


    model_name = 'Saved_Models/TSP_%d/scatgnn_layer_%d_hid_%d_model_210_temp_3.500.pth'%(num_of_nodes,num_layers, num_hidden)# topk = 10
    model.load_state_dict(torch.load(model_name))
    Saved_indices,Saved_Values,Saved_sol,Saved_pos = test(test_loader,topk) # epoch=20>10
    print(Saved_indices)
    print(Saved_Values)
    print(Saved_sol)
    print(Saved_pos)
    

