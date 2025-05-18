# In[]
from owlready2 import get_ontology
import numpy as np
import pandas as pd
import torch

def find_cls(onto, ct_name = None, clid = None):
    xs = []
    if clid is not None:
        for x in onto.classes():
            if x.name == clid:
                xs.append(x)

    elif ct_name is not None:
        for x in onto.classes():
            if x.label == ct_name:
                xs.append(x)
    else:
        raise ValueError("Need ct_name or clid")

    return xs

def find_ancestors(cl_term, filter_cl = True):
    ancestors = cl_term.ancestors()
    ancestor_clids = []
    ancestor_names = []
    for ancestor in ancestors:
        if (filter_cl) & (ancestor.name[:2] != "CL"):
            # skip
            continue
        ancestor_clids.append(ancestor.name)
        ancestor_names.append(ancestor.label)
        
    return ancestor_clids, ancestor_names


# Test example
onto = get_ontology("/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/cl.owl").load()
clid = "CL_0011025" # exhaust T cell
cl_term = find_cls(onto, clid = clid)
print(cl_term[0].label)
ancestor_clids, ancestor_names = find_ancestors(cl_term = cl_term[0])
print(ancestor_names)


# In[]
# read in the meta-data
# data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/permuted/"
data_dir = "/data/zzhang834/hs_healthy_2025_01_30/permuted/"
sizes = np.loadtxt(data_dir + "sizes.txt")
chunk_sizes = []
meta_cells = []
for partition_idx in range(0, len(sizes)):
    meta_cells_chunk = pd.read_parquet(data_dir + f"obs_{partition_idx}_batchcode.parquet")
    chunk_sizes.append(meta_cells_chunk.shape[0])
    meta_cells.append(meta_cells_chunk)
meta_cells = pd.concat(meta_cells, axis = 0)

# select the labels
label_name = meta_cells["cell_type"].values
label_clid = meta_cells["cell_type_ontology_term_id"].values
label_fullname = ["--".join([x,y]) for x, y in zip(label_clid, label_name)]

# factorize
label_id, label_code = pd.factorize(np.array(label_fullname), sort = True)
# unique clids
clid_code = ["_".join(x.split("--")[0].split(":")) for x in label_code]
bincode_table = pd.DataFrame(index = clid_code, columns = clid_code, data = 0)

# In[]
# write the label_id into meta-data, and save the code-book
meta_cells["label_id"] = label_id
# meta_cells.drop(labels = ["batch_id"], axis = 1, inplace=True)
ptr = 0
for cum_idx, chunk_size in enumerate(chunk_sizes):
    meta_cells_chunk = meta_cells.iloc[ptr:(chunk_size + ptr), :]
    meta_cells_chunk.to_parquet(data_dir + f"obs_{cum_idx}_batchcode.parquet")
    ptr += chunk_size

# In[]
for clid in clid_code:
    if clid == "unknown":
        continue
    cl_term = find_cls(onto = onto, clid = clid)
    assert len(cl_term) == 1
    cl_term = cl_term[0]
    ancestor_clids, ancestor_names = find_ancestors(cl_term)
    # include self
    bincode_table.loc[clid, clid] = 1
    # find the intersection
    ancestor_clids = list(set(ancestor_clids).intersection(set(clid_code)))
    bincode_table.loc[clid, ancestor_clids] = 1

# In[]
# drop the unknown column, all zero anyway
bincode_table.index = label_code
bincode_table.columns = label_code
bincode_table = bincode_table.loc[:, bincode_table.columns != "unknown--unknown"]

# Save the dict for model training
label_dict = {"label_bincode": bincode_table}
torch.save(label_dict, data_dir + "label_dict.pt")

# In[]
# --------------------------------------------------------------------------------
#
# Evaluate the correctness of bincode label
#
# --------------------------------------------------------------------------------
def exact_equal(A, B):
    result = torch.all(torch.eq(A[:, None, :], B[None, :, :]), dim=2) 
    return result

def find_ancestor(A, B, chunk_size = 16):
    results = []
    for start in range(0, A.shape[0], chunk_size):
        end = start + chunk_size
        # partial = torch.sum(A[start:end, None, :].bool() | B[None, :, :].bool(), dim=2)
        partial = torch.sum(torch.logical_or(A[start:end, None, :], B[None, :, :]), dim=2)
        results.append(partial)

    result = torch.cat(results, dim=0)
    return result <= torch.sum(A, dim = 1, keepdim = True)

def find_descend(A, B):
    """ 
    Find the descendent cells
    using bincode, the descendent cells should have 1 for all 1s in the cell bincode
    including itself
    return a cell by cell binary matrix, where 1 denote descendent, 0 denote non-descendent
    """
    result = A @ B.T 
    return result == torch.sum(A, dim = 1, keepdim = True)


# NOTE: sanity check
bincode = torch.tensor(bincode_table.values)
# make sure the label is unique
pos_mtx = exact_equal(bincode, bincode)
assert torch.all(pos_mtx == torch.eye(pos_mtx.shape[0]))

ancestor_mtx = find_ancestor(bincode, bincode)
descend_mtx = find_descend(bincode, bincode)

# NOTE: the ancestor matrix is the bincode itself (both including itself, drop unknown)
# the last column of ancestor matrix is all True because known is all-zero, the ancestor of all cells
assert torch.all(ancestor_mtx[:, :-1].to(int) == bincode)
# Desendent matrix is ancestor matrix reverse 
assert torch.all(ancestor_mtx.T == descend_mtx)

# negative (remaining not related), neutral(ancestor, remove same), positive (descendent, same)
neutral = ancestor_mtx & ~exact_equal(bincode, bincode)
pos = descend_mtx & exact_equal(bincode, bincode)

assert torch.all(pos & neutral == False)




# %%
