import numpy as np
import torch
import time
from layers.ourmodels import HGNN,MPNN
from layers.ours import construct_neighbor_matrix_with_self_loops_pyg,dense_to_sparse_coo,process_sparse_A
from SLIC.our_SLIC import Segmenttttt
import os
from layers.Combinours import MPNNCNN,SGNNMPNN,SSMPNN,SSMPNN2
from load_data import visualize_adjacency_matrix,color_map_dict

def ours_model_inputs(data,gt,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG,layer):
    print('useing ourmodel')
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

    def getQandA(data,class_count):
        ls = Segmenttttt(data, class_count - 1)

        # 目标路径：当前路径下的 HSI 文件夹
        output_folder = './HSI'

        # 确保文件夹存在，如果没有则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存 Q 和 A 的文件路径
        q_file_path = os.path.join(output_folder, 'Q111.npy')
        a_file_path = os.path.join(output_folder, 'A111.npy')

        # 如果 Q 和 A 文件存在，则直接加载，否则重新计算 Q 和 A
        if os.path.exists(q_file_path) and os.path.exists(a_file_path):
            # 加载保存的 Q 和 A
            Q = np.load(q_file_path)
            A = np.load(a_file_path)
            print("Loaded Q and A from files.")
        else:
            # 计算 Q 和 A
            tic0 = time.perf_counter()
            Q, S, A, Seg = ls.SLIC_Process(data, scale=superpixel_scale)
            toc0 = time.perf_counter()
            LDA_SLIC_Time = toc0 - tic0
            print("LDA-SLIC costs time: {:.2f} seconds".format(LDA_SLIC_Time))

            # 保存 Q 和 A 到文件
            np.save(q_file_path, Q)
            np.save(a_file_path, A)
            print("Saved Q and A to files.")

        return Q,A

    LDA_SLIC_Time=0
    if FLAG==200:
        Q, A = getQandA(data, class_count)
        Q = torch.from_numpy(Q).to(device)
        A = torch.from_numpy(A).to(device)
    elif superpixel_scale<10000:
        ls = Segmenttttt(data, class_count - 1)
        tic0 = time.perf_counter()
        Q, S, A, Seg = ls.SLIC_Process(data, scale=superpixel_scale)
        toc0 = time.perf_counter()
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
    if FLAG == 1:
        name = 'indian_'
    elif FLAG == 2:
        name = 'paviaU_'
    elif FLAG == 3:
        name = 'salinas_'




    if layer=='HGNN':
        print('useing HGNN')
        net = HGNN(height, width, bands, class_count, Q, A)
    elif layer=='MPNN':
        ticous = time.perf_counter()
        AX = process_sparse_A(net_input, device, row=False, col=False, combin=True, FLAG=FLAG)
        #AX = construct_neighbor_matrix_with_self_loops_pyg(Q.cpu().numpy()).edge_index.to(device)#AX=dense_to_sparse_coo(AX)
        #visualize_adjacency_matrix(AX.cpu(), gt, color_map_dict, save_dir='adj', k=0.5,dataname=name,filename='ourIP.png')#0.5,0.12,0.1

        tocous = time.perf_counter()
        print('ous get A time:', (tocous - ticous)+LDA_SLIC_Time)

        print('useing MPNN')
        net = MPNN(height=height, width=width, changel=bands, class_count=class_count, A=AX,FLAG=FLAG)
    elif layer=='MPNNCNN':
        ticous = time.perf_counter()
        AX = process_sparse_A(net_input, device, row=False, col=False, combin=True, FLAG=FLAG)
        tocous = time.perf_counter()
        print('ous get A time:', (tocous - ticous) + LDA_SLIC_Time)
        print('useing MPNNCNN')
        net = MPNNCNN(height=height, width=width, changel=bands, class_count=class_count, A=AX,FLAG=FLAG)
    elif layer=='SGNNMPNN':
        ticous = time.perf_counter()
        AX = process_sparse_A(net_input, device, row=False, col=False, combin=True, FLAG=FLAG)
        tocous = time.perf_counter()
        print('ous get A time:', (tocous - ticous) + LDA_SLIC_Time)
        print('useing SGNNMPNN')
        net = SGNNMPNN(height, width, bands, class_count, Q=Q, A=A,AX=AX,FLAG=FLAG)
    elif layer=='SSMPNN':
        ticous = time.perf_counter()
        AX = process_sparse_A(net_input, device, row=False, col=False, combin=True, FLAG=FLAG)
        tocous = time.perf_counter()
        print('ous get A time:', (tocous - ticous) + LDA_SLIC_Time)
        print('useing SSMPNN')
        if FLAG==2:
            net = SSMPNN2(height, width, bands, class_count, Q=Q, A=A, AX=AX)
        else:
            net = SSMPNN(height, width, bands, class_count, Q=Q, A=A,AX=AX,FLAG=FLAG)




    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)



