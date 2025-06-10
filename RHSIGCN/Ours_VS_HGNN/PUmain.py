import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

import numpy as np
import torch
import random
import time
from eval import compute_loss, evaluate_performance
from load_data import load_dataset, split_data
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

MODEL='ourmodel'#11346
useours=False
FLAG = 2
Seed_List = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

OA_ALL, AA_ALL, KPP_ALL, AVG_ALL = [], [], [], []
Train_Time_ALL, Test_Time_ALL = [], []

data, gt, class_count, dataset_name = load_dataset(FLAG,MODEL)






import math


for curr_seed in Seed_List:
    (train_gt, val_gt, test_gt,
     train_onehot, val_onehot, test_onehot) = split_data(gt, curr_seed, class_count, FLAG,sample_type='ours',model=MODEL)
    fix_seed(curr_seed)
    #labelratio(train_gt,test_gt,val_gt,class_count+1)

    (net_input,S,A,A2,A3, train_gt_tensor, val_gt_tensor, test_gt_tensor,
     train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
     train_mask_tensor, val_mask_tensor, test_mask_tensor,
     net, superpixel_scale, learning_rate, WEIGHT_DECAY, max_epoch)=prepare_model(useours,MODEL, FLAG, data, gt,train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device)


    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=WEIGHT_DECAY)
    torch.cuda.empty_cache()



    best_loss = float('inf')
    tic1 = time.time()

    for i in range(max_epoch + 1):
        net.train()
        optimizer.zero_grad()

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()

        output = net(net_input)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        if i == 100:
            print(end_time - start_time)
            print(f"Memory used for inference: {(end_memory - start_memory) / (1024 ** 3):.2f} GB")

        loss = compute_loss(output, train_onehot_tensor, train_mask_tensor)
        loss.backward()
        optimizer.step()

        if math.isnan(loss):
            print("Loss is NaN, stopping training.")
            break


        with torch.no_grad():
            net.eval()
            valloss = compute_loss(output, val_onehot_tensor, val_mask_tensor)
            if valloss < best_loss:
                best_loss = valloss
                torch.save(net.state_dict(), "model/best_model.pt")
                #print("save model...")

            torch.cuda.empty_cache()
            if i % 100 == 0:
                trainOA = evaluate_performance(output, train_gt_tensor, train_onehot_tensor, class_count,
                                               require_detail=False,
                                               printFlag=False)
                valOA = evaluate_performance(output, val_gt_tensor, val_onehot_tensor, class_count,
                                             require_detail=False,
                                             printFlag=False)
                print(f"{i + 1}\ttrain loss={loss:.4f} train OA={trainOA:.4f} val loss={valloss:.4f} val OA={valOA:.4f}")

    toc1 = time.time()
    training_time = toc1 - tic1
    Train_Time_ALL.append(training_time)

    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load("model/best_model.pt", weights_only=True))
        net.eval()
        tic2 = time.time()
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



    OA_ALL.append(testOA.cpu() if torch.is_tensor(testOA) else testOA)
    AA_ALL.append(testAA)
    KPP_ALL.append(testKappa)
    AVG_ALL.append(acc_list)
    del net
    torch.cuda.empty_cache()

# 
OA_ALL = np.array([x.cpu().numpy() if torch.is_tensor(x) else x for x in OA_ALL])
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL = np.array(Train_Time_ALL)
Test_Time_ALL = np.array(Test_Time_ALL)

print("\ntrain_ratio=auto based on flag",
      "\n==============================================================================")
print('OA=', np.mean(OA_ALL)*100, '+-', np.std(OA_ALL)*100)
print('AA=', np.mean(AA_ALL)*100, '+-', np.std(AA_ALL)*100)
print('Kpp=', np.mean(KPP_ALL)*100, '+-', np.std(KPP_ALL)*100)
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
