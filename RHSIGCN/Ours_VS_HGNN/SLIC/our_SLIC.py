import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing



def SegmentsLabelProcess(labels):

 
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLICcccc(object):
    def __init__(self, HSI, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        height, width, bands = HSI.shape  #
        data = np.reshape(HSI, [height * width, bands])
        min_max = preprocessing.StandardScaler()
        data = min_max.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])


    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        # 
        # data_pca = self.pca_processing(img, 3)
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1  # =segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)



        segments_1 = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)  # 
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)  # 
        x = np.reshape(img, [-1, d])  # Flatten(x)

        x_center = np.zeros([superpixel_count], dtype=np.float32)
        y_center = np.zeros([superpixel_count], dtype=np.float32)

        for i in range(superpixel_count):
            idx = np.where(segments_1 == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

            seg_idx = np.where(segments == i)
            x_center[i] = np.mean(seg_idx[0])
            y_center[i] = np.mean(seg_idx[1])

        self.S = S
        self.Q = Q

        return Q, S, self.segments

    def get_A(self):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        #print(self.segments.shape)

        for i in range(h - 1):
            for j in range(w - 1):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                         continue
                    A[idx1, idx2] = A[idx2, idx1] = 1

        return A


class Segmenttttt(object):
    def __init__(self, data, n_component):
        self.data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape


    def SLIC_Process(self, img, scale=25):
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)

        # # IP
        myslic = SLICcccc(img, n_segments=n_segments_init, compactness=0.01, sigma=1, min_size_factor=0.1,
                      max_size_factor=10)
        # IP
        # if FLAG == 1:
        #     myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.01, sigma=1.5, min_size_factor=0.1,
        #                   max_size_factor=10)
        # # # PU
        # if FLAG == 2:
        #     myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.01, sigma=1.5, min_size_factor=0.1,
        #                   max_size_factor=10)
        # # # salinas
        # if FLAG == 3:
        # myslic = SLIC(img, FLAG, n_segments=n_segments_init, compactness=0.1, sigma=1.5, min_size_factor=0.1,
        #                   max_size_factor=10)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A = myslic.get_A()
        return Q, S, A, Segments
