import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
eps = 2.2204e-16
import sys
import os

def topic_diversity(topic_matrix, top_k=10):
    """ Topic Diversity (TD) measures how diverse the discovered topics are.

    We define topic diversity to be the percentage of unique words in the top 25 words (Dieng et al., 2020)
    of the selected topics. TD close to 0 indicates redundant topics, TD close to 1 indicates more varied topics.

    Args:
        topic_matrix: shape [K, V]
        top_k:
    """
    num_topics = topic_matrix.shape[0]
    top_words_idx = np.zeros((num_topics, top_k))
    for k in range(num_topics):
        idx = np.argsort(topic_matrix[k, :])[::-1][:top_k]
        top_words_idx[k, :] = idx
    num_unique = len(np.unique(top_words_idx))
    num_total = num_topics * top_k
    td = num_unique / num_total
    # print('Topic diversity is: {}'.format(td))
    return td

def compute_npmi(corpus, word_i, word_j):
    """ Pointwise Mutual Information (PMI) measures the association of a pair of outcomes x and y.

    PMI is defined as log[p(x, y)/p(x)p(y)], which can be further normalized between [-1, +1], resulting in
    -1 (in the limit) for never occurring together, 0 for independence, and +1 for complete co-occurrence.
    The Normalized PMI is computed by PMI(x, y) / [-log(x, y)].
    """
    num_docs = len(corpus)
    num_docs_appear_wi = 0
    num_docs_appear_wj = 0
    num_docs_appear_both = 0
    for n in range(num_docs):
        doc = corpus[n]
        # doc = corpus[n].squeeze(0)
        # doc = [doc.squeeze()] if len(doc) == 1 else doc.squeeze()

        if word_i in doc:
        # if doc[word_i] > 0:
            num_docs_appear_wi += 1
        if word_j in doc:
        # if doc[word_j] > 0:
            num_docs_appear_wj += 1
        if [word_i, word_j] in doc:
        # if doc[word_i] > 0 and doc[word_j] > 0:
            num_docs_appear_both += 1

    if num_docs_appear_both == 0:
        return -1
    else:
        pmi = np.log(num_docs) + np.log(num_docs_appear_both) - \
              np.log(num_docs_appear_wi) - np.log(num_docs_appear_wj)
        if num_docs == num_docs_appear_both:
            return 1
        else:
            return pmi / (np.log(num_docs) - np.log(num_docs_appear_both))

def topic_coherence(corpus, topic_matrix, top_k=25):
    """ Topic Coherence measures the semantic coherence of top words in the discovered topics.

    We apply the widely-used Normalized Pointwise Mutual Information (NPMI) (Aletras & Stevenson, 2013; Lau et al., 2014)
    computed over the top 10 words of each topic, by the Palmetto package (Röder et al., 2015).

    Args:
        corpus:
        vocab:
        topic_matrix:
        top_k:
    """
    num_docs = len(corpus)
    # print('Number of documents: ', num_docs)

    tc_list = []
    num_topics = topic_matrix.shape[0]
    # print('Number of topics: ', num_topics)
    for k in range(num_topics):
        # print('Topic Index: {}/{}'.format(k, num_topics))
        top_words_idx = np.argsort(topic_matrix[k, :])[::-1][:top_k]
        # top_words = [vocab[idx] for idx in list(top_words_idx)]

        pairs_count = 0
        tc_k = 0
        for i, word in enumerate(top_words_idx):
            for j in range(i + 1, top_k):
                # tc_list.append(compute_npmi(corpus, word, top_words[j]))
                # tc_list.append(compute_npmi(corpus, word, top_words_idx[j]))
                tc_k += compute_npmi(corpus, word, top_words_idx[j])
                pairs_count += 1
        tc_list.append(tc_k / pairs_count)

    # tc = sum(tc_list) / (num_topics * pairs_count)
    # print('Topic coherence is: {}'.format(tc))
    return tc_list

# Load Data (X)
data_path=sys.argv[1]
model_path=sys.argv[2]
K=sys.argv[3]
save_path=sys.argv[4]

if not os.path.exists(save_path):
    os.makedirs(save_path)

train_data_diagnosis=np.load(data_path+'Data_diag.npy',allow_pickle=True).item().toarray()
train_data_diagnosis[train_data_diagnosis>1] = 1
train_data_diagnosis = np.array(train_data_diagnosis, order='C')

train_data_drug=np.load(data_path+'Data_drug.npy',allow_pickle=True).item().toarray()
train_data_drug[train_data_drug>1] = 1
train_data_drug = np.array(train_data_drug, order='C')

train_data_procedure=np.load(data_path+'Data_proc.npy',allow_pickle=True).item().toarray()
train_data_procedure[train_data_procedure>1] = 1
train_data_procedure = np.array(train_data_procedure, order='C')

# Load model (Phi, Theta)
Phi_diag= np.load(model_path+'Phi_diag_mean.npy')
Phi_procedure= np.load(model_path+'Phi_procedure_mean.npy')
Phi_drug= np.load(model_path+'Phi_drug_mean.npy')
Theta = np.load(model_path+'Theta_mean.npy')

Lambda = np.dot(Phi_diag, Theta)
P = 1 - np.exp(-Lambda)
P[P == 0] = eps
P[P == 1] = 1 - eps
like_diag = np.sum(np.sum(train_data_diagnosis * np.log(P) + (1 - train_data_diagnosis) * np.log(1 - P))) / train_data_diagnosis.shape[0]

Lambda = np.dot(Phi_drug, Theta)
P = 1 - np.exp(-Lambda)
P[P == 0] = eps
P[P == 1] = 1 - eps
like_drug = np.sum(np.sum(train_data_drug * np.log(P) + (1 - train_data_drug) * np.log(1 - P))) / train_data_drug.shape[0]

Lambda = np.dot(Phi_procedure, Theta)
P = 1 - np.exp(-Lambda)
P[P == 0] = eps
P[P == 1] = 1 - eps
like_procedure = np.sum(np.sum(train_data_procedure * np.log(P) + (1 - train_data_procedure) * np.log(1 - P))) / train_data_procedure.shape[0]

topic_coherence_diag = topic_diversity(np.transpose(Phi_diag))
topic_coherence_procedure = topic_diversity(np.transpose(Phi_procedure))
topic_coherence_drug = topic_diversity(np.transpose(Phi_drug))

# save results
out_f=save_path+'likelihood_topicCoherence.tsv'
if os.path.isfile(out_f):
    # If the file exists, open it in append mode
    with open(out_f, "a") as f:
        f.write("%s\t%f\t%f\t%f\t%f\t%f\t%f\n" % (K,like_diag,like_procedure,like_drug,topic_coherence_diag,topic_coherence_drug,topic_coherence_procedure))
else:
    # If the file does not exist, open it in write mode
    with open(out_f, "w") as f:
        f.write("topic\tlikelihood_diag\tlikelihood_drug\tlikelihood_proc\tcoherence_diag\tcoherence_drug\tcoherence_proc\n")

f.close()


