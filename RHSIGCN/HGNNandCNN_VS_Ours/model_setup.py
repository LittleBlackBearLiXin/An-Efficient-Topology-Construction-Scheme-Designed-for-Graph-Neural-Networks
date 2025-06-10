import numpy as np
import torch
import time
from SLIC.CEGCN_SLIC import LDA_SLIC
from layers import CEGCN
from SLIC.WFCG_SLIC import LDA_SLICC
from layers import WFCG
from layers.ours import construct_neighbor_matrix_with_self_loops_pyg,getA,process_sparse_A
def CEGCN_prepare_model_inputs(data,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, FLAG, superpixel_scale, device,useours):
    height, width, bands = data.shape
    m, n = height, width
    total = m * n

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)

    tic0 = time.time()
    lda = LDA_SLIC(data, train_gt, class_count - 1)
    Q, S, A, Seg = lda.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-SLIC costs time:",LDA_SLIC_Time)

    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))

    if FLAG == 1:
        net = CEGCN.CEGCN(height, width, bands, class_count, Q, A, model='normal',useours=useours)
    else:
        net = CEGCN.CEGCN(height, width, bands, class_count, Q, A,useours=useours)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)




def WFCG_prepare_model_inputs(data,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, FLAG, superpixel_scale, device,useours):
    height, width, bands = data.shape
    m, n = height, width
    total = m * n

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)
    tic0 = time.time()
    ls = LDA_SLICC(data, train_gt, class_count - 1)
    Q, S, A, Seg = ls.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-SLIC costs time: ",LDA_SLIC_Time)

    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))

    net = WFCG.WFCG(height, width, bands, class_count, Q, A,useours=useours)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)


from sklearn.decomposition import PCA
from SLIC import AMGCFN_SLIC
from layers import AMGCFN
def AMGCFN_prepare_model_inputs(data,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, FLAG, superpixel_scale, device,useours):
    height, width, bands = data.shape
    m, n = height, width
    total = m * n
    pca_bands=3
    GCN_nhid = 64  # GCN隐藏层通道数
    CNN_nhid = 64  # CNN隐藏层通道数

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)

    # 数据PCA处理 pca_bands = 3
    def pca_process(n_data, n_labels):
        n_labels = np.reshape(n_labels, [-1])
        n_idx = np.where(n_labels != 0)[0]
        x_flatt = np.reshape(n_data, [height * width, bands])
        x = x_flatt[n_idx]
        pca = PCA(n_components=pca_bands)
        pca.fit(x)
        X_new = pca.transform(x_flatt)
        print(pca.explained_variance_ratio_)
        return np.reshape(X_new, [height, width, -1])

    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'PCA processing')
    pca_data = pca_process(data, np.reshape(train_gt, [height, width]))

    # 根据训练集样本进行超像素分割
    tic0 = time.time()
    superpixel = AMGCFN_SLIC.SlicProcess(data, np.reshape(
        train_gt, [height, width]), class_count - 1)

    Q, S, W, Seg = superpixel.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    PCA_SLIC_Time = toc0 - tic0

    print('get A time:',PCA_SLIC_Time)

    # 获取不同hop的图的邻接矩阵
    superpixel_count, _ = W.shape
    pathset = []

    def DFS_get_path(start_node, n_k, get_path):
        if len(get_path) == n_k + 1:
            pathset.append(get_path[:])
        else:
            sub_node = list(np.where(W[start_node] != 0)[0])
            for next_node in sub_node:
                if next_node not in get_path:
                    get_path.append(next_node)
                    DFS_get_path(next_node, n_k, get_path)
                    get_path.pop()
        return 0

    def Get_k_hop(k_hop, n_graph):
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              ' Processing new graph ', k_hop, ' hop')
        new_graph = np.zeros_like(n_graph)
        for center in range(superpixel_count):
            path = [center]
            DFS_get_path(center, k_hop, path)
            for n_path in pathset:
                weight = 0
                for n_node in range(len(n_path) - 1):
                    weight += n_graph[n_path[n_node], n_path[n_node + 1]]
                weight = weight / k_hop
                # if weight > new_graph[n_path[0], n_path[-1]]:
                new_graph[n_path[0], n_path[-1]] = new_graph[n_path[-1], n_path[0]] \
                    = max(weight, new_graph[n_path[0], n_path[-1]], new_graph[n_path[-1], n_path[0]])

            pathset.clear()
        new_graph = new_graph + np.eye(superpixel_count)
        return new_graph

    tic1 = time.time()
    A = Get_k_hop(1, W)
    A2 = Get_k_hop(2, W)
    A3 = Get_k_hop(3, W)
    toc1 = time.time()

    Hop_Graph_Time = toc1 - tic1
    print("Hop-Graph costs time: ",Hop_Graph_Time+PCA_SLIC_Time)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), 'New graph finish')


    Q = torch.from_numpy(Q).to(device)
    S = torch.from_numpy(S.astype(np.float32)).to(device)

    A = torch.from_numpy(A.astype(np.float32)).to(device)  # 邻接矩阵hop_1
    A2 = torch.from_numpy(A2.astype(np.float32)).to(device)  # 邻接矩阵hop_2
    A3 = torch.from_numpy(A3.astype(np.float32)).to(device)  # 邻接矩阵hop_3

    nodes, channel = S.shape



    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))

    net = AMGCFN.Net(height, width, channel, class_count,
                     GCN_nhid, CNN_nhid, Q, nodes, bands,useours=useours)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("Model parameters:", count_parameters(net))

    return (net_input, S, A, A2, A3,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)



from SLIC.MSSGU_SLIC import SegmentMap
from layers.MSSGU import HiGCN
def MSSGU_prepare_model_inputs(data,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count,dataset_name, device, useours):
    height, width, bands = data.shape
    m, n = height, width
    total = m * n
    Unet_Depth=4

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)

    tic = time.time()
    SM = SegmentMap(dataset_name)
    S_list, A_list = SM.getHierarchy()
    S_list = S_list[0:int(Unet_Depth)]
    A_list = A_list[0:int(Unet_Depth)]

    for i in range(Unet_Depth):
        a = A_list[i].sum(-1)
        print(a.min(), a.max(), a.mean())


    # GPU
    S_list_gpu = []
    A_list_gpu = []
    for i in range(len(S_list)):
        S_list_gpu.append(torch.from_numpy(np.array(S_list[i], dtype=np.float32)).to(device))
        print(S_list_gpu[i].shape)
        A_list_gpu.append(torch.from_numpy(np.array(A_list[i], dtype=np.float32)).to(device))
        print(A_list_gpu[i].shape)

    toc = time.time()
    print('getHierarchy -- cost time:', (toc - tic))

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))
    Neighbors=0

    net = HiGCN(height, width, bands, class_count, S_list_gpu, A_list_gpu, Neighbors,useours=useours)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)

