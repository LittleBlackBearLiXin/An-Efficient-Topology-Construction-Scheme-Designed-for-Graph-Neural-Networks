import scipy.io as sio
from sklearn import preprocessing
import os
import numpy as np
import random

import torch
import matplotlib.pyplot as plt

color_map_dict = {
    'indian_': np.array([[255, 255, 255],  # 为标签 0 增加全白色
                         [0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                         [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                         [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                         [255, 170, 0]], dtype=np.uint8),

    'paviaU_': np.array([[255, 255, 255],  # 为标签 0 增加全白色
                         [0, 0, 255], [76, 230, 0], [255, 190, 232], [255, 0, 0], [156, 156, 156],
                         [255, 255, 115], [0, 255, 197], [132, 0, 168], [0, 0, 0]], dtype=np.uint8),

    'salinas_': np.array([[255, 255, 255],  # 为标签 0 增加全白色
                          [0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                          [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                          [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                          [255, 170, 0]], dtype=np.uint8),
}




def pltgraph(gt,output,name,dataset_name,truegarph=False):
    height, width= gt.shape
    if truegarph:
        classification_map = gt

    else:
        predy = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
        classification_map = (torch.where(torch.tensor(gt) > 0, 1, 0)) * predy
    #print("Max value in classification_map:", classification_map.max())
    #print("Min value in classification_map:", classification_map.min())
    palette = color_map_dict.get(dataset_name)
    map = palette[classification_map]
    plt.figure()
    plt.imshow(map, cmap='jet')
    plt.xticks([])
    plt.yticks([])

    output_dir = dataset_name
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, name + '.png'),
        bbox_inches='tight', dpi=300)





def load_dataset(flag, MODEL):
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    if flag == 1:
        data_path = os.path.join(parent_dir, '../HyperImage_data/indian/Indian_pines_corrected.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/indian/Indian_pines_gt.mat')
        data = sio.loadmat(data_path)['indian_pines_corrected']
        gt = sio.loadmat(gt_path)['indian_pines_gt']
        class_count = 16
        dataset_name = "indian_"
        print('Useing data is:',dataset_name)
    elif flag == 2:
        data_path = os.path.join(parent_dir, '../HyperImage_data/paviaU/PaviaU.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/paviaU/PaviaU_gt.mat')
        data = sio.loadmat(data_path)['paviaU']
        gt = sio.loadmat(gt_path)['paviaU_gt']
        class_count = 9
        dataset_name = "paviaU_"
        print('Useing data is:', dataset_name)
    elif flag == 3:
        data_path = os.path.join(parent_dir, '../HyperImage_data/Salinas/Salinas_corrected.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/Salinas/Salinas_gt.mat')
        data = sio.loadmat(data_path)['salinas_corrected']
        gt = sio.loadmat(gt_path)['salinas_gt']
        class_count = 16
        dataset_name = "salinas_"
        print('Useing data is:', dataset_name)
    else:
        raise ValueError("Invalid FLAG value")


    if MODEL == 'CEGCN':
        data = standardize_data(data)
    elif MODEL == 'WFCG':
        data = normalize_data(data)
    elif MODEL == 'AMGCFN':
        print('N0 Noram')
    elif MODEL == 'MSSGU':
        data = standardize_data(data)
    elif MODEL=='ourmodel':
        data = standardize_data(data)

    return data, gt, class_count, dataset_name


def standardize_data(data):
    height, width, bands = data.shape
    reshaped = data.reshape(-1, bands)
    scaler = preprocessing.StandardScaler()
    normalized = scaler.fit_transform(reshaped).reshape(height, width, bands)
    return normalized


def normalize_data(data):

    min_val = np.min(data)
    max_val = np.max(data)


    normalized = (data - min_val) / (max_val - min_val)

    return normalized


def split_data_ratio(gt, seed, class_count, flag,model):
    if model=='CEGCN':
        if flag == 1:
            train_ratio = 0.1
        else:
            train_ratio = 0.01
        val_ratio_per_class = 0.01  #

    elif model=='WFCG':
        if flag==1:
            train_ratio = 0.01
            val_ratio_per_class = 0.01
        else:
            train_ratio = 0.001
            val_ratio_per_class = 0.001
    elif model=='AMGCFN':
        if flag==1:
            train_ratio = 0.02
            val_ratio_per_class = 0.02
        else:
            train_ratio = 0.002
            val_ratio_per_class = 0.002
    elif model=='MSSGU':
        if flag==1:
            train_ratio = 0.05
            val_ratio_per_class = 0.01
        else:
            train_ratio = 0.005
            val_ratio_per_class = 0.005


    height, width = gt.shape
    total = height * width
    gt_reshape = gt.reshape(-1)
    m, n = height, width
    random.seed(seed)

    # 1.
    train_idx = []
    for c in range(1, class_count + 1):
        idx = np.where(gt_reshape == c)[0]
        num = int(np.ceil(len(idx) * train_ratio))
        if num > 0:
            sampled = random.sample(list(idx), num)
            train_idx.extend(sampled)

    train_idx = set(train_idx)
    all_idx = set(np.where(gt_reshape > 0)[0])
    test_idx = all_idx - train_idx


    val_idx = set()
    for c in range(1, class_count + 1):
        class_test_idx = [i for i in test_idx if gt_reshape[i] == c]
        num_val = int(np.ceil(len(class_test_idx) * val_ratio_per_class))
        if num_val > 0 and len(class_test_idx) >= num_val:
            sampled_val = random.sample(class_test_idx, num_val)
            val_idx.update(sampled_val)

    test_idx -= val_idx


    def build_mask(index_set):
        mask = np.zeros(total)
        for i in index_set:
            mask[i] = gt_reshape[i]
        return mask.reshape(m, n)

    train_gt = build_mask(train_idx)
    val_gt = build_mask(val_idx)
    test_gt = build_mask(test_idx)


    def gt_to_one_hot(gt_img):
        onehot = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_img.reshape(-1)
        for i in range(total):
            if flat[i] > 0:
                onehot[i, int(flat[i]) - 1] = 1
        return onehot

    return (
        train_gt,
        val_gt,
        test_gt,
        gt_to_one_hot(train_gt),
        gt_to_one_hot(val_gt),
        gt_to_one_hot(test_gt),
    )




def split_data_pixel_sample(gt, seed, class_count, flag,model):
    train_samples_per_class = 5
    val_samples_per_class = 5
    height, width = gt.shape
    total = height * width
    gt_reshape = gt.reshape(-1)
    random.seed(seed)


    train_idx = []
    for c in range(1, class_count + 1):
        idx = np.where(gt_reshape == c)[0]
        if len(idx) == 0:
            continue
        num = min(train_samples_per_class, len(idx) // 2)
        sampled = random.sample(list(idx), num)
        train_idx.extend(sampled)

    train_idx = set(train_idx)
    all_idx = set(np.where(gt_reshape > 0)[0])
    test_idx = all_idx - train_idx


    val_idx = set()
    for c in range(1, class_count + 1):
        class_test_idx = [i for i in test_idx if gt_reshape[i] == c]
        if len(class_test_idx) >= val_samples_per_class:
            sampled_val = random.sample(class_test_idx, val_samples_per_class)
            val_idx.update(sampled_val)

    test_idx -= val_idx


    def build_mask(index_set):
        mask = np.zeros(total)
        for i in index_set:
            mask[i] = gt_reshape[i]
        return mask.reshape(height, width)

    train_gt = build_mask(train_idx)
    val_gt = build_mask(val_idx)
    test_gt = build_mask(test_idx)


    def to_one_hot(gt_img):
        onehot = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_img.reshape(-1)
        for i in range(total):
            if flat[i] > 0:
                onehot[i, int(flat[i]) - 1] = 1
        return onehot

    return train_gt, val_gt, test_gt, to_one_hot(train_gt), to_one_hot(val_gt), to_one_hot(test_gt)


def ours_split(gt, seed, class_count):
    train_samples_per_class = 5
    val_samples_per_class = 5
    height, width = gt.shape
    total = height * width
    gt_reshape = gt.reshape(-1)
    random.seed(seed)


    train_idx = []
    for c in range(1, class_count + 1):
        idx = np.where(gt_reshape == c)[0]
        if len(idx) == 0:
            continue
        num = min(train_samples_per_class, len(idx) // 2)
        sampled = random.sample(list(idx), num)
        train_idx.extend(sampled)

    train_idx = set(train_idx)
    all_idx = set(np.where(gt_reshape > 0)[0])
    test_idx = all_idx - train_idx


    val_idx = set()
    for c in range(1, class_count + 1):
        class_test_idx = [i for i in test_idx if gt_reshape[i] == c]
        if len(class_test_idx) >= val_samples_per_class:
            sampled_val = random.sample(class_test_idx, val_samples_per_class)
            val_idx.update(sampled_val)

    test_idx -= val_idx


    def build_mask(index_set):
        mask = np.zeros(total)
        for i in index_set:
            mask[i] = gt_reshape[i]
        return mask.reshape(height, width)

    train_gt = build_mask(train_idx)
    val_gt = build_mask(val_idx)
    test_gt = build_mask(test_idx)


    def to_one_hot(gt_img):
        onehot = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_img.reshape(-1)
        for i in range(total):
            if flat[i] > 0:
                onehot[i, int(flat[i]) - 1] = 1
        return onehot

    return train_gt, val_gt, test_gt, to_one_hot(train_gt), to_one_hot(val_gt), to_one_hot(test_gt)



def split_data(gt, seed, class_count,flag,sample_type,model):
    if sample_type== 'ratio':
        train_gt,val_gt,test_gt,train_onehot,val_onehot,test_onehot=split_data_ratio(gt, seed, class_count,flag,model)
    elif sample_type=='ours':
        train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot = ours_split(gt, seed, class_count)
    else:
        train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot = split_data_pixel_sample(gt, seed,class_count, flag,model)

    train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot = train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot

    return train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.colors import LinearSegmentedColormap


def visualize_adjacency_matrix(adj_matrix, gt, color_map_dict,k,dataname,
                           save_dir='adj_3d', filename='3d_adjacency.png'):
    """
    3D Adjacency Matrix Visualization with:
    1. Background node removal (label = 0)
    2. 50% random sampling per class
    3. Black edges in 3D space
    4. Node coloring by class

    Args:
        adj_matrix: Sparse adjacency matrix
        gt: Ground truth label matrix (2D numpy array)
        color_map_dict: Dictionary containing color maps
        save_dir: Directory to save visualization
        filename: Output filename
    """
    # Convert to COO format
    if hasattr(adj_matrix, 'is_sparse') and adj_matrix.is_sparse:
        adj_matrix = adj_matrix.coalesce()
        rows, cols = adj_matrix.indices().cpu().numpy()
        values = adj_matrix.values().cpu().numpy() if adj_matrix.values().numel() > 0 else None
    else:
        adj_matrix = adj_matrix.tocoo()
        rows, cols = adj_matrix.row, adj_matrix.col
        values = adj_matrix.data if adj_matrix.data.size > 0 else None

    # 1. Remove background nodes
    non_bg_coords = np.argwhere(gt != 0)
    node_labels = gt[non_bg_coords[:, 0], non_bg_coords[:, 1]]

    if len(non_bg_coords) == 0:
        print("No non-background nodes found")
        return

    # 2. 50% random sampling per class
    unique_labels = np.unique(node_labels)
    sampled_indices = []
    label_mapping = {}

    for label in unique_labels:
        label_indices = np.where(node_labels == label)[0]
        sample_size = max(1, int(k * len(label_indices)))
        sampled = np.random.choice(label_indices, sample_size, replace=False)
        sampled_indices.extend(sampled)

        for new_idx, original_idx in enumerate(sampled):
            label_mapping[original_idx] = new_idx

    sampled_indices = np.array(sampled_indices)
    sampled_coords = non_bg_coords[sampled_indices]
    sampled_labels = node_labels[sampled_indices]

    # 3. Create position arrays (using original coordinates for spatial reference)
    x = sampled_coords[:, 1]  # Using column index as x
    y = sampled_coords[:, 0]  # Using row index as y
    z = np.zeros_like(x)  # Flat in z-dimension initially

    # 4. Prepare node colors
    color_map = color_map_dict.get(dataname, [])
    if len(color_map) > 0 and 0 in unique_labels:
        color_map = color_map[1:]  # Skip background color if present
    node_colors = np.array([get_node_color(label, color_map) for label in sampled_labels]) / 255.0

    # 5. Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 6. Plot nodes
    ax.scatter(x, y, z, c=node_colors, s=10, alpha=0.8, depthshade=True, edgecolors='k')

    # 7. Filter and plot edges in 3D
    global_coords = {(r, c): i for i, (r, c) in enumerate(sampled_coords)}
    edge_count = 0

    for i in range(len(rows)):
        src = (rows[i], cols[i])
        dst = (cols[i], rows[i])  # Undirected

        if src in global_coords and dst in global_coords:
            src_idx = global_coords[src]
            dst_idx = global_coords[dst]

            # Draw line in 3D space
            ax.plot(
                [x[src_idx], x[dst_idx]],
                [y[src_idx], y[dst_idx]],
                [z[src_idx], z[dst_idx]],
                color='gray', alpha=0.4, linewidth=0.8
            )
            edge_count += 1

    # 8. Add elevation to show adjacency structure
    # Create a small random z-offset to reveal edges
    z_offset = np.random.uniform(-0.5, 0.5, size=len(z))
    ax.scatter(x, y, z_offset, c=node_colors, s=10, alpha=0.6, depthshade=True)

    # 9. Add connecting lines between original and elevated positions
    for i in range(len(x)):
        ax.plot(
            [x[i], x[i]],
            [y[i], y[i]],
            [z[i], z_offset[i]],
            color='gray', alpha=0.4, linewidth=0.8
        )

    # 10. Add labels and title
    ax.set_title(f"3D Adjacency Matrix", fontsize=14)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('')

    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)

    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"3D visualization saved to {save_path}")
    print(f"Displayed: {len(sampled_coords)} nodes, {edge_count} edges")


def get_node_color(label, label_colors):
    """Get color for node based on label"""
    try:
        label = int(label)
        if 0 <= label < len(label_colors):
            return label_colors[label]
    except (ValueError, TypeError):
        pass
    return [255, 255, 255]  # Default white


from sklearn.manifold import TSNE
import torch



def visualize_tsne(output, gt_labels, color_map_dict, filename,dataname):

    os.makedirs('tsne', exist_ok=True)
    save_path = os.path.join('tsne', filename)

    output_np = output.detach().cpu().numpy()
    gt_labels_np = gt_labels

    gt_labels_np = gt_labels_np.flatten()

    non_bg_mask = gt_labels_np != 0
    features = output_np[non_bg_mask]
    labels = gt_labels_np[non_bg_mask]

    if len(features) == 0:
        print("Warning: No non-background samples found!")
        return

    print(f"Running t-SNE on {len(features)} non-background samples...")

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features) - 1),  # 30
        n_iter=1000,  #
        random_state=42,
        verbose=1  #
    )
    tsne_results = tsne.fit_transform(features)

    color_map = color_map_dict.get(dataname, [])
    if len(color_map) > 0 and 0 in np.unique(gt_labels_np):
        color_map = color_map[1:] #

    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        color = np.array(color_map[(label-1) % len(color_map)]) / 255.0 if len(color_map) > 0 else [0.7, 0.7, 0.7]

        mask = labels == label
        plt.scatter(
            tsne_results[mask, 0],  #
            tsne_results[mask, 1],  #
            color=color,
            label=f'Class {label}',
            alpha=0.7,
            s=25,
            edgecolor='white',
            linewidth=0.3
        )

    #plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', framealpha=0.9)
    #plt.grid(alpha=0.2)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"t-SNE visualization saved to {save_path}")
    return save_path