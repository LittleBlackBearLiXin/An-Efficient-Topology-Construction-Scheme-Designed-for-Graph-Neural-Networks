
from model_setup import CEGCN_prepare_model_inputs,AMGCFN_prepare_model_inputs,\
    WFCG_prepare_model_inputs,MSSGU_prepare_model_inputs




def prepare_model(useours,MODEL, FLAG, data, train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device):
    # Initialize variables
    superpixel_scale = 0
    learning_rate = 0
    WEIGHT_DECAY = 0
    max_epoch = 0
    S=None
    A=None
    A2=None
    A3=None
    net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor = None, None, None, None
    train_onehot_tensor, val_onehot_tensor, test_onehot_tensor = None, None, None
    train_mask_tensor, val_mask_tensor, test_mask_tensor = None, None, None
    net = None

    if MODEL == 'CEGCN':
        superpixel_scale=100
        learning_rate = 5e-4
        WEIGHT_DECAY = 0
        max_epoch = 600
        net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = CEGCN_prepare_model_inputs(
            data, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, FLAG, superpixel_scale, device,useours=useours)


    elif MODEL == 'WFCG':
        if FLAG == 1:
            superpixel_scale = 100
        elif FLAG == 2:
            superpixel_scale = 100
        elif FLAG == 3:
            superpixel_scale = 100
        learning_rate = 1e-3
        WEIGHT_DECAY = 1e-3
        max_epoch = 300
        net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = WFCG_prepare_model_inputs(
            data, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, FLAG, superpixel_scale, device,useours=useours)


    elif MODEL == 'AMGCFN':
        superpixel_scale=100
        learning_rate = 5e-4
        WEIGHT_DECAY = 1e-4
        max_epoch = 500
        net_input,S,A,A2,A3, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = AMGCFN_prepare_model_inputs(
            data, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, FLAG, superpixel_scale, device,useours=useours)



    elif MODEL == 'MSSGU':
        if FLAG == 1:
            dataname = "indian_"
        elif FLAG == 2:
            dataname = "paviaU_"
        elif FLAG == 3:
            dataname = "salinas_"
        learning_rate = 5e-4
        WEIGHT_DECAY = 0
        max_epoch = 600
        net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = MSSGU_prepare_model_inputs(
            data, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, dataname, device,useours=useours)


    return net_input,S,A,A2,A3, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
           train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
           train_mask_tensor, val_mask_tensor, test_mask_tensor, \
           net, superpixel_scale,learning_rate, WEIGHT_DECAY, max_epoch
