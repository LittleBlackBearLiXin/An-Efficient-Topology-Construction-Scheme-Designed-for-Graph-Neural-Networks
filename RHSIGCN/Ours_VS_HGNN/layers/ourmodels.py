
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch_geometric.nn import GCNConv,GATConv,SAGEConv


def noramlize(A: torch.Tensor):
    D = A.sum(1)
    D_hat = torch.diag(torch.pow(D, -0.5))
    A = torch.mm(torch.mm(D_hat, A), D_hat)
    return A

def spareadjacency(adjacency):

    if not adjacency.is_sparse:
        adjacency = adjacency.to_sparse()

    adjacency = adjacency.coalesce()
    indices = adjacency.indices()

    values = adjacency.values()


    size = adjacency.size()
    adjacency = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32, device=device)


    return adjacency




class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):#
        super(GCNLayer, self).__init__()

        self.Activition = nn.LeakyReLU()
        self.bn= nn.BatchNorm1d(output_dim)


        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count =A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = noramlize(A+self.I)
        ratio=1-(torch.sum(A!=0)/(nodes_count**2))
        print(ratio)
        if ratio>10000:
            self.A = spareadjacency(A)
        else:
            self.A = A



    def forward(self, H):
        output = torch.mm(self.A, self.GCN_liner_out_1(H))
        output=self.bn(output)
        output = self.Activition(output)
        return output




class HGNN(nn.Module):#1,1,
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor):
        super(HGNN, self).__init__()

        self.class_count = class_count

        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.prelin=nn.Linear(changel,128)

        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.GCN_Branch = nn.Sequential()
        layers_count=2
        for i in range(layers_count):
            if i==0:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, self.A))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.BN = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):

        x = x.reshape([self.height * self.width, -1])
        x=self.BN(self.prelin(x))

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), x)
        H1=self.GCN_Branch(superpixels_flatten)

        GCN_result = torch.matmul(self.Q, H1)

        Y = self.Softmax_linear(GCN_result)
        Y = F.softmax(Y, -1)
        return Y





class GNN(nn.Module):
    def __init__(self, input: int, output: int,droupout,A,res=False,useln=False,MPNN='GCN'):
        super(GNN, self).__init__()

        self.Activition = nn.LeakyReLU()
        self.input = input

        self.output = output
        self.droupout = droupout
        self.BN = nn.BatchNorm1d(output)




        if MPNN=='GCN':
            self.gnn = GCNConv(input, output)
        elif MPNN=='SAGE':
            self.gnn =SAGEConv(input, output,normalize=False,aggr='sum')
        elif MPNN=='GAT':
            self.gnn = GATConv(input, output//2,heads=2,concat=True,add_self_loops=True)

        self.A = A
        self.res=res
        self.useln=useln
        if self.res:
            self.rslin=nn.Linear(input,output)#F,T,F
        if self.useln:
            self.LN = nn.LayerNorm(input)

    def forward(self, x):
        if self.useln:
            x=self.LN(x)#F,F,T
        output = self.gnn(x,self.A)
        if self.res:
            output+=self.rslin(x)
        output = self.BN(output)
        output = self.Activition(output)
        output = F.dropout(output, p=self.droupout, training=self.training)
        return output


class MLP(nn.Module):
    def __init__(self, input: int, output: int,droupout):
        super(MLP, self).__init__()

        self.Activition = nn.LeakyReLU()
        self.input = input

        self.output = output
        self.droupout = droupout
        self.BN = nn.BatchNorm1d(output)
        self.lin = nn.Linear(input, output)


    def forward(self, x):
        output = self.lin(x)
        output = self.BN(output)
        output = self.Activition(output)
        output = F.dropout(output, p=self.droupout, training=self.training)
        return output


class CNN(nn.Module):
    def __init__(self, in_ch, out_ch,droupout, kernel_size=3):
        super(CNN, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.droupout=droupout
        self.Act = nn.LeakyReLU()
        #self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        #input = self.BN(input)
        out = self.point_conv(input)
        out = self.Act(out)
        out = self.depth_conv(out)
        #out = self.BN(out)
        out = self.Act(out)
        out = F.dropout(out, p=self.droupout, training=self.training)

        return out




class MPNN(nn.Module):#0,0,0.2
    def __init__(self, height: int, width: int, changel: int, class_count: int, A, hide=128, FLAG=100,model='GNN'):
        super(MPNN, self).__init__()

        if FLAG == 1:
            layers_count = 2
            self.droupout = 0
            self.res = False
            ln = False
        elif FLAG == 2:
            layers_count = 2
            self.droupout = 0
            self.res = True
            ln = False
        elif FLAG == 3:
            layers_count = 2
            self.droupout = 0.2
            self.res = False
            ln=True
        else:
            layers_count = 0
            self.droupout = 0
            self.res = True
            ln=False

        self.class_count = class_count
        self.flag=FLAG

        self.channel = changel
        self.height = height
        self.width = width
        self.A=A


        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)

        #layers_count = 2#2,2,3

        self.NN_Branch = nn.Sequential()
        for i in range(layers_count):
            if model == 'GNN':
                print('GNN')
                if i == 0:
                    self.NN_Branch.add_module('GCN_Branch' + str(i), GNN(hide, hide, self.droupout, self.A,res=self.res,useln=ln))
                else:
                    self.NN_Branch.add_module('GCN_Branch' + str(i), GNN(hide, hide, self.droupout, self.A,res=self.res,useln=ln))
            elif model=='MLP':#2,0
                print('MLP')
                if i == 0:
                    self.NN_Branch.add_module('MLP_Branch' + str(i), MLP(hide, hide, self.droupout))
                else:
                    self.NN_Branch.add_module('MLP_Branch' + str(i), MLP(hide, hide, self.droupout))
        if model=='CNN':
            self.NN_Branch.add_module('MLP_Branch', CNNnn(self.height,self.width,changel=self.channel,class_count=self.class_count,
                                                          hide=hide,droupout=self.droupout,layers_count=1))#1,

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):

        x = x.reshape([self.height * self.width, -1])
        if self.flag==1:
            x=F.normalize(x,p=3)
            x = (self.BN(self.prelin(x)))
        elif self.flag == 2:
            x = F.normalize(x, p=3)
            x = (self.BN(self.prelin(x)))
        else:
            x=(self.BN(self.prelin(x)))

        H1 = self.NN_Branch(x)

        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)

        return Y






class CNNnn(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, hide=128, droupout=0,layers_count=2):
        super(CNNnn, self).__init__()

        self.CNN_denoise = nn.Sequential()
        self.CNN_denoise.add_module('CNN_denoise_BN1', nn.BatchNorm2d(changel))
        self.CNN_denoise.add_module('CNN_denoise_Conv1',
                                    nn.Conv2d(changel, hide, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act1', nn.LeakyReLU())
        self.CNN_denoise.add_module('CNN_denoise_BN2', nn.BatchNorm2d(hide))
        self.CNN_denoise.add_module('CNN_denoise_Conv2', nn.Conv2d(hide, hide, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act2', nn.LeakyReLU())



        self.class_count = class_count

        self.channel = changel
        self.height = height
        self.width = width
        self.droupout=droupout


        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            print('CNN')
            if i==0:
                self.CNN_Branch.add_module('MLP_Branch' + str(i), CNN(hide, hide, self.droupout))
            else:
                self.CNN_Branch.add_module('MLP_Branch' + str(i), CNN(hide, hide, self.droupout))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        x = torch.squeeze(noise, 0).permute([1, 2, 0])

        CNN_result = self.CNN_Branch(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        return CNN_result
