from torch.autograd import Variable
import numpy as np
import torch


def compress(train, test, model, classes=80):
    retrievalB = list([])
    retrievalL = np.ones((1, classes))
    for batch_step, (data, target) in enumerate(train):
        var_data = Variable(data.cuda())
        _,H,code= model(var_data)
        code = torch.sign(H)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL = np.concatenate((retrievalL,target.numpy()), axis=0)

    queryB = list([])
    queryL = np.ones((1, classes))
    for batch_step, (data, target) in enumerate(test):
        var_data = Variable(data.cuda())
        _,H,code = model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        queryL = np.concatenate((queryL,target.numpy()), axis=0)


    retrievalB = np.array(retrievalB)  
    retrievalL = retrievalL[1:,:]
    retrievalL = np.array(retrievalL)     


    queryB = np.array(queryB)
    queryL = queryL[1:,:]
    queryL = np.array(queryL)   
    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # tsum number of items with same label
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap




def calculate_precision_recall_k(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    trainset_len = rB.shape[0]
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    AP = np.zeros(num_query)
    total_good_pairs = 0

    map = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # tsum number of items with same label
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        
        # use another methods
        P = np.cumsum(gnd) / Ns
        AP[iter] = np.sum(P * gnd) /sum(gnd)
        sum_tp = sum_tp + np.cumsum(gnd)

        # recall total_good_pairs
        total_good_pairs = total_good_pairs + gnd.sum()


        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    precision_at_k = sum_tp / Ns / num_query
    recall_at_k = sum_tp / total_good_pairs
    pre = precision_at_k[:10000]
    rec = recall_at_k[:10000]

    return pre, rec
 

def calculate_p_r_curve(qB, rB, queryL, retrievalL):

    Wtrue = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)   # 1000 *59000
    Dhat = calculate_hamming(qB, rB)
    max_hamm = Dhat.max()
    total_good_pairs = Wtrue.sum()

    precision = np.zeros([int(max_hamm),1])
    recall = np.zeros([int(max_hamm),1])
    for n in range(int(max_hamm)):

        j = (Dhat <= (n+0.00001))
        retrieved_good_pairs = Wtrue[j].sum()

        retrieved_pairs = j.sum()
        precision[n] = retrieved_good_pairs/(retrieved_pairs+1e-24)
        recall[n]= retrieved_good_pairs/total_good_pairs

    return precision, recall



def pr_curve(query_code, retrieval_code, query_targets, retrieval_targets):
    """
    P-R curve.
    Args
        query_code(torch.Tensor): Query hash code.
        retrieval_code(torch.Tensor): Retrieval hash code.
        query_targets(torch.Tensor): Query targets.
        retrieval_targets(torch.Tensor): Retrieval targets.
        device (torch.device): Using CPU or GPU.
    Returns
        P(torch.Tensor): Precision.
        R(torch.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float()).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask

    return P, R
