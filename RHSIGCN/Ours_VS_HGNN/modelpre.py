
from model_setup import ours_model_inputs




def prepare_model(useours,MODEL, FLAG, data, gt,train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device):
    # Initialize variables
    layer='HGNN'
    superpixel_scale = 100
    learning_rate = 5e-4
    WEIGHT_DECAY = 0
    max_epoch = 500
    S=None
    A=None
    A2=None
    A3=None
    net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor = None, None, None, None
    train_onehot_tensor, val_onehot_tensor, test_onehot_tensor = None, None, None
    train_mask_tensor, val_mask_tensor, test_mask_tensor = None, None, None
    net = None

    if MODEL == 'ourmodel':
        if layer=='SSMPNN':
            superpixel_scale=200
        net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = ours_model_inputs(
            data, gt,train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device,FLAG,layer=layer)



    return net_input,S,A,A2,A3, train_gt_tensor, val_gt_tensor, test_gt_tensor, \
           train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
           train_mask_tensor, val_mask_tensor, test_mask_tensor, \
           net, superpixel_scale,learning_rate, WEIGHT_DECAY, max_epoch
