import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        assert x1 <= x2, 'x1 > x2, wrong coordinate.'
        assert y1 <= y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch) / 255)
    
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def set_area(self, mask, patch):
        patch_size = self.get_size()
        patch = patch.astype('float32')
        patch = cv.resize(patch, interpolation=cv.INTER_CUBIC, dsize=patch_size)
        mask[self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask
    
    def get_coord(self):
        return self.x1, self.x2, self.y1, self.y2
    
    def get_size(self):
        return self.x2 - self.x1, self.y2 - self.y1
    
    def make_square(self):
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        if width > height:
            diff = width - height
            self.y2 += diff
        elif height > width:
            diff = height - width
            self.x2 += diff
    
    def draw(self, ax, c='grey', lw=0.5, **kwargs):
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2 - self.x1, 
                                 height=self.y2 - self.y1, 
                                 linewidth=lw, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

class FixedQuadTree:
    def __init__(self, domain, fixed_length=128, build_from_info=False, meta_info=None) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        if build_from_info:
            self.nodes = self.decoder_nodes(meta_info=meta_info)
        else:
            self._build_tree()
    
    def nodes_value(self):
        meta_value = []
        for rect, v in self.nodes:
            size, _ = rect.get_size()
            meta_value += [[v, size]]
        return meta_value
    
    def encode_nodes(self):
        meta_info = []
        for rect, v in self.nodes:
            meta_info += [[rect.x1, rect.x2, rect.y1, rect.y2]]
        return meta_info
    
    def decoder_nodes(self, meta_info):
        nodes = []
        for info in meta_info:
            x1, x2, y1, y2 = info
            n = Rect(x1, x2, y1, y2)
            v = n.contains(self.domain)
            nodes += [[n, v]]
        return nodes
            
    def _build_tree(self):
        h, w = self.domain.shape
        assert h > 0 and w > 0, "Wrong img size."
        root = Rect(0, w, 0, h)
        root.make_square()  # Make the root square if necessary
        self.nodes = [[root, root.contains(self.domain)]]
        while len(self.nodes) < self.fixed_length:
            bbox, value = max(self.nodes, key=lambda x: x[1])
            idx = self.nodes.index([bbox, value])
            if min(bbox.get_size()) < 2:
                break

            x1, x2, y1, y2 = bbox.get_coord()
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            lt = Rect(x1, mid_x, mid_y, y2)
            rt = Rect(mid_x, x2, mid_y, y2)
            lb = Rect(x1, mid_x, y1, mid_y)
            rb = Rect(mid_x, x2, y1, mid_y)

            # Ensure that all new rectangles are square
            lt.make_square()
            rt.make_square()
            lb.make_square()
            rb.make_square()
            
            v1 = lt.contains(self.domain)
            v2 = rt.contains(self.domain)
            v3 = lb.contains(self.domain)
            v4 = rb.contains(self.domain)
            
            self.nodes = self.nodes[:idx] + [[lt, v1], [rt, v2], [lb, v3], [rb, v4]] + self.nodes[idx+1:]
        
        # Ensure we have exactly fixed_length nodes
        while len(self.nodes) > self.fixed_length:
            self.nodes.pop()
        while len(self.nodes) < self.fixed_length:
            self.nodes.append([Rect(0, 0, 0, 0), 0])
    
    def count_patches(self):
        return len(self.nodes)
    
    def serialize(self, img, size=(8, 8, 3)):
        seq_patch = []
        for bbox, value in self.nodes:
            area = bbox.get_area(img)
            if area.size == 0:  # Skip empty patches
                continue
            h1, w1, _ = area.shape
            if h1 != w1:
                if h1 > w1:
                    diff = h1 - w1
                    area = cv.copyMakeBorder(area, 0, 0, 0, diff, cv.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    diff = w1 - h1
                    area = cv.copyMakeBorder(area, 0, diff, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
            seq_patch.append(area)
            
        h2, w2, c2 = size
        for i in range(len(seq_patch)):
            if seq_patch[i].size == 0:  # Skip empty patches
                continue
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_CUBIC)
        if len(seq_patch) != self.fixed_length:
            seq_patch += [np.zeros(shape=(h2, w2, c2))] * (self.fixed_length - len(seq_patch))

        assert len(seq_patch) == self.fixed_length, "Not equal fixed length."
        return seq_patch
    
    def deserialize(self, seq, patch_size, channel):
        seq = np.reshape(seq, (self.fixed_length, patch_size, patch_size, channel))
        seq = seq.astype(int)
        mask = np.zeros(shape=self.domain.shape)
        for idx, (bbox, value) in enumerate(self.nodes):
            pred_mask = seq[idx, ...]
            mask = bbox.set_area(mask, pred_mask)
        return mask
    
    def draw(self, ax, c='grey', lw=1):
        for bbox, value in self.nodes:
            bbox.draw(ax=ax)