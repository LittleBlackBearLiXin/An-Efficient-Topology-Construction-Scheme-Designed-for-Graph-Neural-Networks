import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

import numpy as np
import torch
import random
import time
from eval import compute_loss, evaluate_performance
from load_data import load_dataset, split_data,pltgraph
from modelpre import prepare_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def labelratio(train_gt,test_gt,val_gt,classcout):
    for i in range(classcout):
        if i>0:
            print(i, np.sum(train_gt == i),np.sum(test_gt == i),np.sum(val_gt==i))


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL='OURSCEGCN'#CEGCN,AMGCFN,WFCG*********************************

useours=False

FLAG = 1#2,3****************************************

Seed_List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

OA_ALL, AA_ALL, KPP_ALL, AVG_ALL = [], [], [], []
Train_Time_ALL, Test_Time_ALL = [], []

data, gt, class_count, dataset_name = load_dataset(FLAG,MODEL)


for curr_seed in Seed_List:
    (train_gt, val_gt, test_gt,
     train_onehot, val_onehot, test_onehot) = split_data(gt, curr_seed, class_count, FLAG,sample_type='ours',model='xxxxxxxxxxxxxx')
    fix_seed(curr_seed)
    #labelratio(train_gt,test_gt,val_gt,class_count+1)

    (net_input,S,A,A2,A3, train_gt_tensor, val_gt_tensor, test_gt_tensor,
     train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
     train_mask_tensor, val_mask_tensor, test_mask_tensor,
     net, superpixel_scale, learning_rate, WEIGHT_DECAY, max_epoch)=prepare_model(useours,MODEL, FLAG, data, train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device)


    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=WEIGHT_DECAY)
    best_loss = float('inf')
    tic1 = time.time()

    for i in range(max_epoch + 1):
        net.train()
        optimizer.zero_grad()
        if MODEL == 'AMGCFN':
            output, Cout, Gout = net(S, A, A2, A3, net_input)
            lossa = compute_loss(output, train_onehot_tensor, train_mask_tensor)
            Closs = compute_loss(Cout, train_onehot_tensor, train_mask_tensor)
            Gloss = compute_loss(Gout, train_onehot_tensor, train_mask_tensor)
            loss = lossa + Closs + Gloss
        elif MODEL=='MSSGU':
            output, _, _ = net(net_input)
            loss = compute_loss(output, train_onehot_tensor, train_mask_tensor)
        else:
            output = net(net_input)
            loss = compute_loss(output, train_onehot_tensor, train_mask_tensor)
        #print(loss.item())
        loss.backward()
        optimizer.step()

        if MODEL=='WFCG':
            i=i*10

        if i % 10 == 0:
            with torch.no_grad():
                net.eval()
                if MODEL=='AMGCFN':
                    output, _, _ = net(S, A, A2, A3, net_input)
                elif MODEL == 'MSSGU':
                    output, _, _ = net(net_input)
                else:
                    output = net(net_input)
                trainOA = evaluate_performance(output, train_gt_tensor, train_onehot_tensor, class_count, require_detail=False, printFlag=False)
                valloss = compute_loss(output, val_onehot_tensor, val_mask_tensor)
                valOA = evaluate_performance(output, val_gt_tensor, val_onehot_tensor, class_count, require_detail=False, printFlag=False)
                if MODEL == 'WFCG':
                    i = i /10
                if valloss < best_loss:
                    best_loss = valloss
                    torch.save(net.state_dict(), "model/best_model.pt")
                    #print("save model...")
                if i%100==0:
                    print(
                        f"{i + 1}\ttrain loss={loss:.4f} train OA={trainOA:.4f} val loss={valloss:.4f} val OA={valOA:.4f}")
            torch.cuda.empty_cache()

    toc1 = time.time()
    training_time = toc1 - tic1
    print('train time:',training_time)
    Train_Time_ALL.append(training_time)

    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load("model/best_model.pt", weights_only=True))
        net.eval()
        tic2 = time.time()

        if MODEL == 'AMGCFN':
            output, _, _ = net(S, A, A2, A3, net_input)
        elif MODEL=='MSSGU':
            output, _, _ = net(net_input)
        else:
            output = net(net_input)

        toc2 = time.time()

        testloss = compute_loss(output, test_onehot_tensor, test_mask_tensor)
        testOA, testAA, testKappa, acc_list = evaluate_performance(
            output, test_gt_tensor, test_onehot_tensor, class_count,
            require_detail=True, printFlag=False)

        acc_str = ', '.join([f"{x:.4f}" for x in acc_list])
        print(
            f"Training runs:{curr_seed + 1}\n[test loss={testloss:.4f} test OA={testOA:.4f} test AA={testAA:.4f} test KPA={testKappa:.4f}]\ntest peracc=[{acc_str}]")

        Test_Time_ALL.append((toc2 - tic2))

        if curr_seed == 0:
            pltgraph(gt,output,MODEL,dataset_name)
            print('------------------------graph-----------------------')

    OA_ALL.append(testOA.cpu() if torch.is_tensor(testOA) else testOA)
    AA_ALL.append(testAA)
    KPP_ALL.append(testKappa)
    AVG_ALL.append(acc_list)
    del net
    torch.cuda.empty_cache()

# 汇总统计
OA_ALL = np.array([x.cpu().numpy() if torch.is_tensor(x) else x for x in OA_ALL])
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL = np.array(Train_Time_ALL)
Test_Time_ALL = np.array(Test_Time_ALL)

print("\ntrain_ratio=auto based on flag",
      "\n==============================================================================")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
print("Average training time:", np.mean(Train_Time_ALL))
print("Average testing time:", np.mean(Test_Time_ALL))

with open(f'results/{dataset_name}_results.txt', 'a+') as f:
    f.write('\n\n************************************************' +
            "\nOA=" + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) +
            "\nAA=" + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) +
            "\nKpp=" + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) +
            "\nAVG=" + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) +
            "\nAverage training time:" + str(np.mean(Train_Time_ALL)) +
            "\nAverage testing time:" + str(np.mean(Test_Time_ALL)))
