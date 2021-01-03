from torch.autograd import Variable
import numpy as np
import torch


def extractab(test, model, classes=10):

    queryB = list([])
    queryH = list([])
    for batch_step, (data, target) in enumerate(test):
        data = data.view(data.size(0), -1)        
        var_data = Variable(data)
        _, H, _ = model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        queryH.extend(H.cpu().data.numpy())


    queryB = np.array(queryB)
    queryH = np.array(queryH)
    return queryB, queryH

def extractab1(test, model, classes=10):

    queryB = list([])
    queryH = list([])
    for batch_step, (data, target) in enumerate(test):
        data = data.view(data.size(0), -1)        
        var_data = Variable(data)
        _, H, _ = model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        queryH.extend(H.cpu().data.numpy())


    queryB = np.array(queryB)
    queryH = np.array(queryH)
    return queryB, queryH

def compress(train, test, model, classes=10):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, target) in enumerate(train):
        data = data.view(data.size(0), -1)
        var_data = Variable(data)
        
        _, H, _= model(var_data)
        code = torch.sign(H)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, target) in enumerate(test):
        data = data.view(data.size(0), -1)        
        var_data = Variable(data)
        _, H, _ = model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.eye(classes)[np.array(retrievalL)]

    queryB = np.array(queryB)
    queryL = np.eye(classes)[np.array(queryL)]
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


def mean_average_precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))

def precision(params):
    database_code = np.array(params['database_code'])
    validation_code = np.array(params['validation_code'])
    database_labels = np.array(params['database_labels'])
    validation_labels = np.array(params['validation_labels'])
    R = params['R']
    query_num = validation_code.shape[0]
    database_code = np.sign(database_code)
    validation_code = np.sign(validation_code)

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        APx.append(float(relevant_num) / R)

    return np.mean(np.array(APx))


def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()
    classes = np.max(tst_label) + 1
    for i in range(classes):
        if i == 0:
            tst_sample_binary = tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]
            tst_sample_label = np.array([i]).repeat(100)
            continue
        else:
            tst_sample_binary = np.concatenate([tst_sample_binary, tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]])
            tst_sample_label = np.concatenate([tst_sample_label, np.array([i]).repeat(100)])
    query_times = tst_sample_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    for i in range(query_times):
        print('Query ', i+1)
        query_label = tst_sample_label[i]
        query_binary = tst_sample_binary[i,:]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
        P = np.cumsum(buffer_yes) / Ns
        precision_radius[i] = P[np.where(np.sort(query_result)>2)[0][0]-1]
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        sum_tp = sum_tp + np.cumsum(buffer_yes)
    precision_at_k = sum_tp / Ns / query_times
    index = [100, 200, 400, 600, 800, 1000]
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    np.save('precision_at_k', precision_at_k)
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)







def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK, is_single_label):
    n_test = query_labels.size(0)
    
    Indices = retrieved_indices[:,:topK]
    if is_single_label:
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK)
        topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
        topTrainLabels = torch.cat(topTrainLabels, dim=0)
        relevances = (test_labels == topTrainLabels).type(torch.cuda.ShortTensor)
    else:
        topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
        topTrainLabels = torch.cat(topTrainLabels, dim=0).type(torch.cuda.ShortTensor)
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor)
        relevances = (topTrainLabels & test_labels).sum(dim=2)
        relevances = (relevances > 0).type(torch.cuda.ShortTensor)
        
    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(100)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k
