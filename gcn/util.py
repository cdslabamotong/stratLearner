import numpy as np
import scipy.sparse as sp
import torch
import multiprocessing 
from datetime import datetime

def load_adj(filename, vNum, no_add_features=True):
    # Reading graphs
    with open(filename) as f:
        content = f.readlines()
        
    content = [x.strip() for x in content]
    if not no_add_features:
        content = [i.split(' ')[:3] for i in content]
    else:
        content = [i.split(' ')[:2] for i in content]
        
    for i, x in enumerate(content):
        content[i] = [int(j) for j in x]
        if no_add_features:
            content[i].append(1)
    
    arr = np.array(content)
    
    #shape = tuple(arr.max(axis=0)[:2]+2)
    #if shape[0] != shape[1]:
    shape = (vNum, vNum)
    
    adj = sp.coo_matrix((arr[:, 2], (arr[:, 0], arr[:, 1])), shape=shape,
                        dtype=arr.dtype)
    
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    print("Done, finished processing adj matrix...")
    return adj
    

def load_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
  
    return content


def one_hot_encoder(data, max_value):
    shape = (data.size, max_value)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    
    return one_hot


def extract_training_sets(filename, dsize=True):
    content = list(filter(None, load_file(filename)))
    if not dsize:
        X = [x for i,x in enumerate(content) if i%2==0]
        y = [x for i,x in enumerate(content) if i%2==1]
    else:
        X = [x for i,x in enumerate(content) if i%4==0]
        y = [x for i,x in enumerate(content) if i%4==1]
      
    # Transforming data format
    X = [i.split(' ') for i in X]
    y = [i.split(' ') for i in y]

    return list(zip(X,y))


def get_loader(dataset, batch_size, num_workers=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, pairPath, graphPath, vNum, total_size, train=True, dsize=True, full=False):
        self.path = "data/kro/"
        self.dsize = dsize
        self.data = self.cache(pairPath, graphPath, vNum)
        self.total_size=total_size
        
    def cache(self, pairPath, graphPath,  target_number):
        print("Processing dataset...")
        adj = load_adj(graphPath, target_number)
        sample_data = extract_training_sets(pairPath, self.dsize)
        data = []
        for datapoint in sample_data:
            # Extract input and target for training and testing
            x_train, y_train = datapoint
            x_train = [int(i) for i in x_train]
            y_train = [int(i) for i in y_train]
            #Transform the input to identity matrix
            # Getting cardnality of the sample
            if self.dsize:
                temp_card = len(x_train)
            temp_tensor = torch.zeros(target_number, target_number)
            for i in x_train:
                temp_tensor[i][i] = 1
            x_train = temp_tensor
            y_train = one_hot_encoder(np.array(y_train), target_number)
            y_train = torch.sum(torch.tensor(y_train), dim=0)#/len(y_train)
            y_train = y_train.unsqueeze(-1)
            if self.dsize:
                data.append((x_train, y_train, adj, temp_card))
            else:
                data.append((x_train, y_train, adj))

        print("Done!")
        return data
    
    def __getitem__(self, item):
        x, y, adj, cardinality = self.data[item]
        return x, y, adj, cardinality
    
    def __len__(self):
        return self.total_size


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def list_to_set(X_list):
    X_set=set()
    for x in X_list:
        X_set.add(str(x))
    return X_set

def testing(X_test,Y_test,Y_pred,instance,args, infTimes=1080):
    #print("Testing Started")

    thread = args.thread;
    block_size =int (infTimes/thread);
    p = multiprocessing.Pool(thread)
   
    influence_Xs = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes) for i in range(thread)),1)
    p.close()
    p.join()
   
    p = multiprocessing.Pool(thread)
    influence_Ys = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_test[i*block_size:(i+1)*block_size]) for i in range(thread)),1)
    p.close()
    p.join()
   
    p = multiprocessing.Pool(thread)
    influence_Y_preds = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_pred[i*block_size:(i+1)*block_size]) for i in range(thread)),1)
    p.close()
    p.join()
   
   
    influence_X=[]
    influence_Y=[]
    influence_Y_pred=[]
    for i in range(thread):
        influence_X.extend(influence_Xs[i])
        influence_Y.extend(influence_Ys[i])
        influence_Y_pred.extend(influence_Y_preds[i])
   
   
    reduce_percent_opt=[]
    reduce_percent_pre = []
    com_to_opt = []
    error_abs = []
    error_ratio = []
    for influence_x, influence_y, influence_y_pred in zip(influence_X, influence_Y, influence_Y_pred):
        #print("{} {} {} ".format(influence_x,influence_y,influence_y_pred))
        reduce_percent_opt.append((influence_x-influence_y)/influence_x)
        reduce_percent_pre.append( (influence_x-influence_y_pred)/influence_x)
        com_to_opt.append((influence_x-influence_y_pred)/(influence_x-influence_y+0.01))
        error_abs.append((influence_y_pred-influence_y))
        error_ratio.append((influence_y_pred-influence_y)/influence_y)
        #print()
    print(args.dataname)
    print("error_abs: {} +- {}".format(np.mean(np.array(error_abs)), np.std(np.array(error_abs))))
    print("error_ratio: {} +- {}".format(np.mean(np.array(error_ratio)), np.std(np.array(error_ratio))))
    print("reduce_percent_opt: {} +- {}".format(np.mean(np.array(reduce_percent_opt)), np.std(np.array(reduce_percent_opt))))
    print("reduce_percent_pre: {} +- {}".format(np.mean(np.array(reduce_percent_pre)), np.std(np.array(reduce_percent_pre))))
    print("com_to_opt: {} +- {}".format(np.mean(np.array(com_to_opt)), np.std(np.array(com_to_opt))))
    
    print("trainNum:{}, testNum:{}, infTimes:{} ".format(args.trainNum, args.testNum,  infTimes))
    
    if args.output:
        now = datetime.now()
        with open(now.strftime("%d-%m-%Y %H:%M:%S"), 'a') as the_file:
            for x_test, y_test, y_pred in zip(X_test,Y_test,Y_pred):
                for target in [x_test, y_test, y_pred]:
                    line='';
                    for a in target:
                        line += a
                        line += ' '
                    line += '\n'
                    the_file.write(line)
                the_file.write('\n')


    print("===============================================================")
