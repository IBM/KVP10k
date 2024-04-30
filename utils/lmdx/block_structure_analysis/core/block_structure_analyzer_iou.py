import numpy as np
from sklearn.neighbors import KDTree
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, inconsistent, maxRstat,maxinconsts
import cv2
from scipy.ndimage import gaussian_filter

class BlockStructureAnalyzerIOU:
    # Inbar used
    def __init__(self, analyzer_config:dict):
        self.analyzer_config = analyzer_config

    #Inbar used
    def analyze(self, boxes_info_list):

        if len(boxes_info_list) == 0:
            return []

        bbox_list = []
        words_list = []
        for box_info in boxes_info_list:
            bbox_list.append(box_info['bbox'])
            words_list.append(box_info['word'])


        doc_text = self.get_doc_text(bbox_list, words_list)

        return doc_text


    # Inbar used
    def get_doc_text(self, bboxes, words):
        """

        Args:
            bboxes: list of boxes, each box is a list: [x1, y1, x2, y2]
            words: list of string, corresponds to bboxes order

        Returns:
            doc_text structure: a list of dictionaries of clusters.
                'cluster' is a dictionary with the following keys:
                    * 'bbox' - bounding box of the cluster
                    * 'lines' - a list of lines. each line is a dictionary with the following keys:
                        * 'text' - a string of the text of the line
                        * 'idx' - ids of the bounding boxes composing the line, relative to the input bboxes list
                        * 'bbox' - a bounding box of the line
                        * 'tokens': list of words in the line
                        * 'word_bboxes': list of line's word bounding-boxes
        """


        k_neighbors = self.analyzer_config['k_neighbors']
        thresh_line = self.analyzer_config['thresh_line']

        pdist_bboxes = BlockStructureAnalyzerIOU.__bboxes_pdist_area(bboxes, k_closest=int(
            BlockStructureAnalyzerIOU.__min2(k_neighbors, len(bboxes))))
        pdist_bboxes = pdist_bboxes - np.diag(np.diag(pdist_bboxes))

        thresh_cluster = self.analyzer_config['thresh_cluster']
        overlap_thresh = self.analyzer_config['overlap_thresh']
        p_cut = self.analyzer_config['p_cut']

        tmp_linkage = shc.linkage(squareform(pdist_bboxes), method='single')
        inconsistency = inconsistent(tmp_linkage, d=9)
        MR = maxRstat(tmp_linkage, inconsistency, 3)
        MI = maxinconsts(tmp_linkage, inconsistency)
        clusts = fcluster(tmp_linkage, t=thresh_cluster, criterion='monocrit', monocrit=MR)
        clusts = BlockStructureAnalyzerIOU.merge_overlapping_clusters(clusts, bboxes, overlap_thresh)
        unique_clust_idx = np.unique(clusts)
        N_clusts = len(unique_clust_idx)
        tmp_bboxes = np.array(bboxes)
        clust_bbox = []
        for l in unique_clust_idx:
            clust_bboxes = tmp_bboxes[clusts == l, :]
            bbox = BlockStructureAnalyzerIOU.__get_bounding_bbox(clust_bboxes)
            clust_bbox.append(bbox)

        sorted_idx_clust = np.argsort(np.array(clust_bbox)[:, 1])

        doc_text = []

        for clust in sorted_idx_clust:
            clust_text = {'bbox': clust_bbox[clust], 'lines': [], 'lines_idx': []}
            bbox_idx = np.where(clusts == unique_clust_idx[clust])[0]
            clust_words = [words[idx] for idx in bbox_idx]
            clust_bboxes = tmp_bboxes[bbox_idx, :]
            if len(bbox_idx) > 1:
                line1 = ''
                tokens = []
                line_idx = []
                sorted_words = np.argsort(
                    [1e2 * ((x[1] + x[3]) / 2) + 1e-2 * ((x[2] + x[0]) / 2) for x in clust_bboxes])
                score_words = np.sort(
                    [1e2 * ((x[1] + x[3]) / 2) + 1e-2 * ((x[2] + x[0]) / 2) for x in clust_bboxes])
                diff_score = np.diff(score_words)
                line_splits = np.array(np.where(diff_score > thresh_line)).ravel()
                if len(line_splits) == 0:
                    sorted_words_h = np.argsort([((x[2] + x[0]) / 2) for x in clust_bboxes])
                    score_words_h = np.sort([((x[2] + x[0]) / 2) for x in clust_bboxes])
                    for idx, score in zip(sorted_words_h, score_words_h):
                        line_idx.append(bbox_idx[idx])
                        line1 = line1 + ' ' + clust_words[idx]
                        tokens.append(clust_words[idx])
                    line_bboxes = tmp_bboxes[line_idx, :]
                    line_bbox = np.array(BlockStructureAnalyzerIOU.__get_bounding_bbox(line_bboxes))
                    clust_text['lines'].append({'text': line1, 'idx': line_idx,
                                                'bbox': line_bbox.ravel(),
                                                'word_bboxes': line_bboxes,
                                                'tokens': tokens})
                    line1 = ''
                    tokens = []
                    line_idx = []
                else:
                    line_splits = np.hstack([-1, line_splits, len(clust_bboxes) - 1])
                    line_splits = np.vstack([line_splits[:-1] + 1, line_splits[1:] + 1])

                    line_splits = line_splits.T
                    for kkk in range(len(line_splits)):

                        tmp_line_idx = sorted_words[line_splits[kkk, 0]:(line_splits[kkk, 1])]
                        tmp_line_bbox = clust_bboxes[tmp_line_idx, :]
                        sorted_words_h = tmp_line_idx[np.argsort([((x[2] + x[0]) / 2) for x in tmp_line_bbox])]
                        score_words_h = np.sort([((x[2] + x[0]) / 2) for x in tmp_line_bbox])
                        for idx, score in zip(sorted_words_h, score_words_h):
                            line_idx.append(bbox_idx[idx])
                            line1 = line1 + ' ' + clust_words[idx]
                            tokens.append(clust_words[idx])
                        line_bboxes = tmp_bboxes[line_idx, :]
                        line_bbox = np.array(BlockStructureAnalyzerIOU.__get_bounding_bbox(line_bboxes))
                        clust_text['lines'].append({'text': line1, 'idx': line_idx,
                                                    'bbox': line_bbox.ravel(),
                                                    'word_bboxes': line_bboxes,
                                                    'tokens': tokens})
                        line1 = ''
                        tokens = []
                        line_idx = []

            elif len(bbox_idx) == 1:
                clust_text['lines'].append({'text': str(clust_words),
                                            'idx': bbox_idx,
                                            'bbox': np.array(tmp_bboxes[bbox_idx, :]).ravel(),
                                            'word_bboxes': np.array(tmp_bboxes[bbox_idx, :]),  # .ravel()
                                            'tokens': [str(clust_words)]})
                line1 = ''
                tokens = []
                line_idx = []
            doc_text.append(clust_text)
        return doc_text

    # Inbar used
    @staticmethod
    def merge_overlapping_clusters(clusts, bboxes, thresh=0.5):
        done = False
        tmp_clusts = clusts.copy()
        iters = 0
        while (done == False) and iters < 5:
            iters += 1
            unique_clust_idx = np.unique(tmp_clusts)
            N_clusts = len(unique_clust_idx)
            tmp_bboxes = np.array(bboxes)
            clust_bbox = []
            for l in unique_clust_idx:
                clust_bboxes = tmp_bboxes[tmp_clusts == l, :]
                bbox = BlockStructureAnalyzerIOU.__get_bounding_bbox(clust_bboxes)
                clust_bbox.append(bbox)
            clust_intersect =BlockStructureAnalyzerIOU.clust_intersection(clust_bbox)
            overlapping_clusters = np.array(np.where(clust_intersect > thresh)).T
            if len(overlapping_clusters) > 0:
                for l in range(len(overlapping_clusters)):
                    tmp_clusts[tmp_clusts == unique_clust_idx[overlapping_clusters[l, 1]]] = unique_clust_idx[
                        overlapping_clusters[l, 0]]
                    overlapping_clusters[overlapping_clusters == overlapping_clusters[l, 1]] = overlapping_clusters[
                        l, 0]
            else:
                done = True

        return tmp_clusts

    #Inbar used
    @staticmethod
    def bbox_intersection(bbox1, bbox2):
        max_0 = BlockStructureAnalyzerIOU.__max2(bbox1[0], bbox2[0])
        max_1 = BlockStructureAnalyzerIOU.__max2(bbox1[1], bbox2[1])
        min_2 = BlockStructureAnalyzerIOU.__min2(bbox1[2], bbox2[2])
        min_3 = BlockStructureAnalyzerIOU.__min2(bbox1[3], bbox2[3])
        area = np.abs(BlockStructureAnalyzerIOU.__max2(0, (min_2 - max_0)) * BlockStructureAnalyzerIOU.__max2(0, (min_3 - max_1)))
        return area

    #Inbar used
    @staticmethod
    def clust_intersection(clust_bboxes):
        N_clusts = len(clust_bboxes)
        output = np.zeros((N_clusts, N_clusts))
        for l in range(N_clusts):
            for k in range(l + 1, N_clusts):
                bbox1_size = (clust_bboxes[l][2] - clust_bboxes[l][0]) * (clust_bboxes[l][3] - clust_bboxes[l][1])
                bbox2_size = (clust_bboxes[k][2] - clust_bboxes[k][0]) * (clust_bboxes[k][3] - clust_bboxes[k][1])
                output[l, k] = BlockStructureAnalyzerIOU.bbox_intersection(clust_bboxes[l], clust_bboxes[k]) / np.min([bbox1_size, bbox2_size])
        return output

    #Inbar used
    @staticmethod
    def __min2(x, y):
        return ((x + y - abs(x - y)) / 2)

    #Inbar used
    @staticmethod
    def __max2(x, y):
        return ((x + y + abs(x - y)) / 2)


    #Inbar Used
    @staticmethod
    def __bbox_dist_area(bbox1, bbox2):
        if type(bbox1) == dict:
            sum1 = bbox1['text_area']
            sum2 = bbox2['text_area']
            bbox1 = bbox1['bbox']
            bbox2 = bbox2['bbox']
        else:
            sum1 = (bbox1[3] - bbox1[1]) * (bbox1[2] - bbox1[0])
            sum2 = (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])
        intersection = 0  # bbox_intersection(bbox1, bbox2)
        bbox_union = [BlockStructureAnalyzerIOU.__min2(bbox1[0], bbox2[0]), BlockStructureAnalyzerIOU.__min2(bbox1[1], bbox2[1]),
                      BlockStructureAnalyzerIOU.__max2(bbox1[2], bbox2[2]), BlockStructureAnalyzerIOU.__max2(bbox1[3], bbox2[3])]
        sum_union = (bbox_union[3] - bbox_union[1]) * (bbox_union[2] - bbox_union[0])
        return BlockStructureAnalyzerIOU.__max2(0, sum_union - (sum1 + sum2 - intersection)), bbox_union

    # Inbar used
    @staticmethod
    def __get_bounding_bbox(bboxes):
        if type(bboxes) == list:
            bboxes = np.array(bboxes)
        return [np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]

    #Inbar used
    @staticmethod
    def __bboxes_pdist_area(bboxes, k_closest=10):
        bbox_centers = [[(x[0] + x[2]) / 2, (x[1] + x[3]) / 2] for x in bboxes]
        bbox_tree = KDTree(bbox_centers, leaf_size=20)
        output = np.ones((len(bboxes), len(bboxes))) * 1e9
        dist, indices = bbox_tree.query(bbox_centers, k=k_closest)

        for l in range(len(bboxes)):  # -1):
            for ind in indices[l][1:]:
                tmp_dist, _ = BlockStructureAnalyzerIOU.__bbox_dist_area(bboxes[l], bboxes[ind])
                output[l, ind] = tmp_dist
                output[ind, l] = tmp_dist
        return output

    @staticmethod
    def get_common_boundaries(boxes_info_list):
        if isinstance(boxes_info_list[0]['bbox'], list):
            left = min(boxes_info_list, key=lambda x: x['bbox'][0])['bbox'][0]
            top = min(boxes_info_list, key=lambda x: x['bbox'][1])['bbox'][1]
            right = max(boxes_info_list, key=lambda x: x['bbox'][2])['bbox'][2]
            bottom = max(boxes_info_list, key=lambda x: x['bbox'][3])['bbox'][3]
        else:  # bbox is dict of l, t, r, b
            left = min(boxes_info_list, key=lambda x: x['bbox']['l'])['bbox']['l']
            top = min(boxes_info_list, key=lambda x: x['bbox']['t'])['bbox']['t']
            right = max(boxes_info_list, key=lambda x: x['bbox']['r'])['bbox']['r']
            bottom = max(boxes_info_list, key=lambda x: x['bbox']['b'])['bbox']['b']

        return [left, top, right, bottom]