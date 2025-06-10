
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
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor,droupout):#
        super(GCNLayer, self).__init__()

        self.Activition = nn.LeakyReLU()
        self.bn= nn.BatchNorm1d(input_dim)


        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count =A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = noramlize(A+self.I)
        ratio=1-(torch.sum(A!=0)/(nodes_count**2))
        print(ratio)
        if ratio>0.92:
            self.A = spareadjacency(A)
        else:
            self.A = A

        self.droupout = droupout


    def forward(self, H):
        H=F.normalize(H)
        H=self.bn(H)
        output = torch.sparse.mm(self.A, self.GCN_liner_out_1(H))
        #output=self.bn(output)
        output = self.Activition(output)
        output = F.dropout(output, p=self.droupout, training=self.training)
        return output








class GNN(nn.Module):
    def __init__(self, input: int, output: int,droupout,A,res=False,MPNN='GCN'):
        super(GNN, self).__init__()

        self.Activition = nn.LeakyReLU()
        self.input = input

        self.output = output
        self.droupout = droupout
        self.BN = nn.BatchNorm1d(input)
        #self.LN=nn.LayerNorm(input)

        if MPNN=='GCN':
            self.gnn = GCNConv(input, output)
        elif MPNN=='SAGE':
            self.gnn =SAGEConv(input, output,normalize=False,aggr='sum')
        elif MPNN=='GAT':
            self.gnn = GATConv(input, output//2,heads=2,concat=True,add_self_loops=True)

        self.A = A
        self.res=res
        if self.res:
            self.rslin=nn.Linear(input,output)#

    def forward(self, x):
        x=F.normalize(x)
        #x=self.LN(x)
        x=self.BN(x)
        output = self.gnn(x,self.A)
        if self.res:
            output+=self.rslin(x)
        output = self.Activition(output)
        output = F.dropout(output, p=self.droupout, training=self.training)
        return output





class CNNCONV(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(CNNCONV, self).__init__()
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
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class MPNNCNN(nn.Module):#0,0.5,0.2
    def __init__(self, height: int, width: int, changel: int, class_count: int, A, hide=128, FLAG=100):
        super(MPNNCNN, self).__init__()
        if FLAG==1:
            droupout=0
            layers_count=2
        elif FLAG==2:
            droupout = 0.5
            layers_count = 4
        else:
            droupout = 0.2
            layers_count = 3

        self.class_count = class_count
        self.flag=FLAG

        self.channel = changel
        self.height = height
        self.width = width
        self.A = A
        self.droupout=droupout


        self.CNN_denoise = nn.Sequential()
        self.CNN_denoise.add_module('CNN_denoise_BN1', nn.BatchNorm2d(self.channel))
        self.CNN_denoise.add_module('CNN_denoise_Conv1',
                                    nn.Conv2d(self.channel, hide, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act1', nn.LeakyReLU())
        self.CNN_denoise.add_module('CNN_denoise_BN2', nn.BatchNorm2d(hide))
        self.CNN_denoise.add_module('CNN_denoise_Conv2', nn.Conv2d(hide, hide, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act2', nn.LeakyReLU())

        CNNlayers_count = 1#1,1,1
        # Pixel-level Convolutional Sub-Network
        self.CNN_Branch = nn.Sequential()
        for i in range(CNNlayers_count):
            print('CNN')
            if i < CNNlayers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), CNNCONV(128, 128, kernel_size=5))
            else:
                if self.flag == 3:
                    self.CNN_Branch.add_module('CNN_Branch' + str(i), CNNCONV(128, 128, kernel_size=5))
                else:
                    self.CNN_Branch.add_module('CNN_Branch' + str(i), CNNCONV(128, 64, kernel_size=5))

        #layers_count=2#2,4,3
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            print('GNN')
            if i < layers_count - 1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GNN(hide, 128, self.droupout, self.A))
            else:
                if self.flag==3:
                    self.GCN_Branch.add_module('GCN_Branch' + str(i), GNN(hide, 128, self.droupout, self.A))
                else:
                    self.GCN_Branch.add_module('GCN_Branch' + str(i), GNN(hide, 64, self.droupout, self.A))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))



    def forward(self, x: torch.Tensor):

        (h, w, c) = x.shape


        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 

        clean_x_flatten = clean_x.reshape([h * w, -1])



        hx = clean_x

        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])


        GCN_result = self.GCN_Branch(clean_x_flatten)


        if self.flag==3:
            Y = GCN_result + CNN_result
        else:
            Y = torch.cat([GCN_result, CNN_result], dim=-1)

        #Y = torch.cat([GCN_result, CNN_result], dim=-1)
        #Y=GCN_result+CNN_result#F,F,T

        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y




class SGNNMPNN(nn.Module):#0.5,0.5,0.2
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,AX,FLAG=100):
        super(SGNNMPNN, self).__init__()
        if FLAG==3:
            layers_count=2
            droupout=0.2
        elif FLAG==2:
            layers_count = 1
            droupout = 0.5
        else:
            layers_count = 2
            droupout=0.5


        self.class_count = class_count

        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.AX = AX
        self.prelin=nn.Linear(changel,128)
        self.droupout=droupout

        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.GCN_Branch = nn.Sequential()
        #layers_count=2#2,1,2
        for i in range(layers_count):
            if i < layers_count - 1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(input_dim=128, output_dim=128, A=self.A,droupout=self.droupout))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(input_dim=128, output_dim=128, A=self.A,droupout=self.droupout))

        MPNNlayers_count = 2  # 2,2,2
        self.MPNN_Branch = nn.Sequential()
        for i in range(MPNNlayers_count):
            print('GNN')
            if i < MPNNlayers_count - 1:
                self.MPNN_Branch.add_module('GCN_Branch' + str(i), GNN(input=128,output= 128,droupout= self.droupout,A=self.AX))
            else:
                self.MPNN_Branch.add_module('GCN_Branch' + str(i), GNN(input=128,output= 128,droupout= self.droupout,A=self.AX))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.BN = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):

        x = x.reshape([self.height * self.width, -1])
        x=self.BN(self.prelin(x))

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), x)
        H1=self.GCN_Branch(superpixels_flatten)

        GCN_result = torch.matmul(self.Q, H1)
        MPNN=self.MPNN_Branch(x)
        result=MPNN+GCN_result#T,T,T

        Y = self.Softmax_linear(result)
        Y = F.softmax(Y, -1)
        return Y



class SSMPNN(nn.Module):#0.5,0.2
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,AX,FLAG=100):
        super(SSMPNN, self).__init__()
        if FLAG==1:
            droupout=0.5
        else:
            droupout=0.2

        self.class_count = class_count

        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.AX = AX
        self.prelin=nn.Linear(changel,128)
        self.droupout=droupout
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        p=0.5
        self.CNN_Branch = nn.Sequential()
        self.CNN_Branch.add_module('CNN_BN1', nn.BatchNorm2d(self.channel))
        self.CNN_Branch.add_module('CNN_Conv1', nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
        self.CNN_Branch.add_module('CNN_Act1', nn.LeakyReLU())
        self.CNN_Branch.add_module('dp1', nn.Dropout(p))
        self.CNN_Branch.add_module('CNN', CNNCONV(128, 128, kernel_size=7))
        self.CNN_Branch.add_module('dp2', nn.Dropout(p))
        self.CNN_Branch.add_module('CNN2', CNNCONV(128, 128, kernel_size=3))
        self.CNN_Branch.add_module('dp3', nn.Dropout(p))

        self.GCN_Branch = nn.Sequential()
        layers_count=2#2,,2
        for i in range(layers_count):
            if i < layers_count - 1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(input_dim=128, output_dim=128, A=self.A,droupout=self.droupout))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(input_dim=128, output_dim=128, A=self.A,droupout=self.droupout))

        MPNNlayers_count = 2  # 2,,2
        self.MPNN_Branch = nn.Sequential()
        for i in range(MPNNlayers_count):
            print('GNN')
            if i < MPNNlayers_count - 1:
                self.MPNN_Branch.add_module('GCN_Branch' + str(i), GNN(input=128,output= 128,droupout= self.droupout,A=self.AX))
            else:
                self.MPNN_Branch.add_module('GCN_Branch' + str(i), GNN(input=128,output= 128,droupout= self.droupout,A=self.AX))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
        self.BN = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):
        cnn = self.CNN_Branch(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(cnn, 0).permute([1, 2, 0]).reshape([-1, cnn.shape[1]])

        x = x.reshape([self.height * self.width, -1])
        x=self.BN(self.prelin(x))

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), x)
        H1=self.GCN_Branch(superpixels_flatten)

        GCN_result = torch.matmul(self.Q, H1)
        MPNN=self.MPNN_Branch(x)
        result=(MPNN+GCN_result)+CNN_result
        Y = self.Softmax_linear(result)
        Y = F.softmax(Y, -1)
        return Y



class SSMPNN2(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor,AX):
        super(SSMPNN2, self).__init__()
        self.class_count = class_count  # 
        # 
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.AX=AX
        self.model = 'normal'
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 

        layers_count = 2
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())


        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), CNNCONV(128, 128, kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), CNNCONV(128, 64, kernel_size=5))

        # Superpixel-level Graph Sub-Network
        gnncout=2
        self.GCN_Branch = nn.Sequential()
        for i in range(gnncout):
            if i < gnncout - 1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, self.A,0))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A,0))


        MPNNlayers_count = 2  # ,2,
        self.MPNN_Branch = nn.Sequential()
        for i in range(MPNNlayers_count):
            print('GNN')
            if i < MPNNlayers_count - 1:
                self.MPNN_Branch.add_module('GCN_Branch' + str(i), GNN(input=128,output= 128,droupout=0.8,A=self.AX))
            else:
                self.MPNN_Branch.add_module('GCN_Branch' + str(i), GNN(input=128,output= 64,droupout=0.8,A=self.AX))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))



    def forward(self, x: torch.Tensor):

        (h, w, c) = x.shape


        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise

        clean_x_flatten = clean_x.reshape([h * w, -1])

        MPNNruslt=self.MPNN_Branch(clean_x_flatten)

        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 


        hx = clean_x


        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # 
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])


        H = superpixels_flatten
        if self.model == 'normal':
            for i in range(len(self.GCN_Branch)): H = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H, model='smoothed')

        GCN_result = torch.matmul(self.Q, H)+MPNNruslt


        Y = torch.cat([GCN_result, CNN_result], dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
