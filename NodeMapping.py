# -*- coding: utf-8 -*-
"""
@author: Hugh Bird
@copyright Copyright 2016, Hugh Bird
@lisence: MIT
@status: alpha
"""
import numpy as np

class NodeIdxMap:
    """ Maps generic node numbers to indexes. Handles enrichment too
    """
    
    def __init__(self):
        self.forward_map = {}
        self.backwards_map = {}
        self.count = 0
        
    def clear(self):
        """ Empty mapping.
        """
        self.forward_map.clear()
        self.backwards_map.clear()
        self.count = 0
        
    def tag_to_idx(self, tag):
        """ Return the idx of a tag
        
        If tag is not in mapping it will be added and an index allocated
        """
        try:
            idx = self.forward_map[tag]
        except KeyError:
            self.forward_map[tag] = self.count
            idx = self.count
            self.backwards_map[idx] = tag
            self.count += 1
        return idx
        
    def idx_to_tag(self, idx):
        """ Return the tag associated with an index
        """
        return self.backwards_map[idx]
        
    def tags_to_idxs(self, tag_iterable):
        """ Return an np.array of idx for a list of tags.
        """
        idxs = np.zeros(len(tag_iterable), dtype=np.int64)
        for i in range(0, len(tag_iterable)):
            idxs[i] = self.tag_to_idx(tag_iterable[i])
            
        return idxs
        
    def idxs_to_tags(self, idxs):
        """ Returns an list of tags given an array of idxs
        """
        tags = []
        for i in idxs:
            tags.append(self.idx_to_tag(i))
        return tags
            
    def num(self):
        """ Returns the number of indexes allocated.
        """
        return self.count
    
    def count_node_idxs(self, nid_list):
        """ Returns the number of indexes associated with a node in mappings
        
        nid_list: a list of node tags.
        Returns a list of counts correspding ot input list
        """
        nids = self.forward_map.keys()[:, 0]
        counts = []
        for node in nid_list:
            counts.append(nids.count(node))
        return counts
        
        
        
        
        
        
        
        
        
        