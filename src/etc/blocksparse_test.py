#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sparse

def xprop_lut(KB, cs, ks, vs, idx, max_seg, min_seg):
    locks = 0
    lockids = dict()
    seg  = list()
    segs = list()
    col  = list()
    cols = list()
    kset = set()
    # get a count of channels for each k
    channels = [0 for k in range(KB)]
    for i in idx:
        channels[ks[i]] += 1
    K = ks[idx[0]]
    seg_count = 0
    for i in idx:
        c, k, v = cs[i], ks[i], vs[i]
        kset.add(k)
        # check for new value of k
        if k != K:
            # keep track of unsegmented columns (for l2norm and testing)
            cols.append( (K, col) )
            col = list()
            # append segment for previous K and start a new one
            if len(seg):
                segs.append( (K, seg) )
                seg = list()
                seg_count += 1
            # for more than one segment we need to use spin locks to sync accumulation
            if seg_count > 1:
                locks += 1
                lockids[K] = locks
            seg_count = 0
            K = k
        col.append( (c, v) )
        seg.append( (c, v) )
        channels[k] -= 1
        # split columns up into segments, but don't let them be too small for effciency sake
        if len(seg) >= max_seg and channels[k] >= min_seg:
            segs.append( (k, seg) )
            seg = list()
            seg_count += 1
    # append last value of k
    cols.append( (k, col) )
    if len(seg):
        segs.append( (k, seg) )
        seg_count += 1
    if seg_count > 1:
        locks += 1
        lockids[k] = locks
    # add in any empty k blocks at the end
    for k in range(KB):
        if k not in kset:
            segs.append( (k, []) )
            cols.append( (k, []) )
            #else:
            #    raise ValueError("sparsity mask has empty mappings.  Not yet supported with feature_axis=0")
    #segs.sort(key=lambda x: len(x[1]), reverse=True)
    # bsmm lut
    offset = len(segs) * 4
    xp_lut = np.empty(offset + len(vs)*2, dtype=np.int32)
    xp_max = 0
    for i, (k, lut) in enumerate(segs):
        # build the lut header: int2 offset, lut_size, K, lock_id
        xp_lut[i*4:(i+1)*4] = offset//2, len(lut), k, lockids.get(k, 0)
        xp_max = max(xp_max, len(lut))
        for entry in lut:
            xp_lut[offset:offset+2] = entry
            offset += 2
    # l2 norm lut (columns not broken up into segments)
    offset = len(cols) * 4
    l2_siz = offset + len(vs)
    # we use int64 views into the lut for tf compatibility reasons..
    if l2_siz & 1:
        l2_siz += 1
    l2_lut = np.zeros(l2_siz, dtype=np.int32)
    l2_max = 0
    for i, (k, lut) in enumerate(cols):
        # build the lut header: int offset, lut_size, K
        l2_lut[i*4:(i+1)*4] = offset, len(lut), k, 0
        l2_max = max(l2_max, len(lut))
        for entry in lut:
            l2_lut[offset] = entry[1]
            offset += 1
    return cols, xp_lut, l2_lut, xp_max*8, l2_max*4, len(segs), locks


layout = np.array([[1,0,1,1,1], [0,1,1,0,1], [0,1,0,1,1]])
print("layout = \n", layout)

csr = sparse.csr_matrix(layout)
cs, ks, vs = sparse.find(csr) # ks is in sorted order by default
print("cs = ", cs)
print("ks = ", ks)
print("vs = ", vs)

blocks = len(vs)
idx = list(range(blocks))
vs = list(range(blocks))

print("blocks = ", blocks)
print("vs = ", vs)
print("idx = ", idx)

KB = 5

max_seg = 100
min_seg = 1

r = xprop_lut(KB, cs, ks, vs, idx, max_seg, min_seg)
fprop_list, fprop_lut, l2_lut, fprop_shared, l2_shared, fprop_segments, fprop_locks = r
print("fprop_list =", fprop_list)
print("fprop_lut =", fprop_lut)
print("l2_lut =", l2_lut)
print("fprop_shared =", fprop_shared)
print("l2_shared =", l2_shared)
print("fprop_segments =", fprop_segments)
print("fprop_locks =", fprop_locks)

print("r = ", r)


"""
LUT segment header structure
{
    entries start offset (divided by 2)
    number of blocks in the segment
    column index of the matrix
    lock index (zero -> lock is not needed eg. column is one segment.)
}

LUT entry structure
{
    row index of the block
    index of the block in the matrix -> where to find the block coeffs.
}

every segment has one LUT header

LUT consists of segment headers for each segment and of entries for each block.


"""

