import numpy as np
import nltk
from pytorch_pretrained_bert import BertTokenizer
import time
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def bert_tokens(text_all, token_choice='bert-base-uncased'):
    """Use Bert tokenizier to tokenize the issued level text"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_article_tokenized = []
    text_flat_tokenized = []
    for text in text_all:
        text_tokenized =[]
        tmp_tokenized=[]
        for sentence in text:
            toks = tokenizer.tokenize(sentence)
            for tok in toks:
                tmp_tokenized.append(tok)
            text_tokenized.append(toks)
        text_flat_tokenized.append(tmp_tokenized)
        text_article_tokenized.append(text_tokenized)
    return text_flat_tokenized, text_article_tokenized


def get_topic_list(topic_file):
    """Return topic list from topic csv file
    csv file: first column is subject, second column is URL
    """
    df = pd.read_csv(topic_file,header = 0)
    df = df.dropna()
    col_names = df.columns
    topics = df[col_names[0]].tolist()
    topics = [topic.replace('--', ' ').replace('Antislavery', 'Anti-slavery').replace('antislavery', 'anti-slavery') for topic in topics]
    
    return topics


def get_topic_embedding(topics, port=6000, port_out=6001, model_path='/home/ubuntu/bert_tests/bert-as-service/uncased_L-12_H-768_A-12'):
    """Use bert-as-service to encode the topics into embeddings vec"""
    common = [
        '-model_dir', model_path,
        '-num_worker', '2',
        '-port', str(port),
        '-port_out', str(port_out),
        '-max_seq_len', '20',
        '-max_batch_size', '256',
        '-pooling_layer', '-2',
        '-cpu',
        #'-show_tokens_to_client',
              ]
    args = get_args_parser().parse_args(common)
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(20)
    print('encoding...')
    bc = BertClient(port=port, port_out=port_out, show_server_config=False)
    #vec = bc.encode(topics, show_tokens=True)
    vec = bc.encode(topics)
    #print(vec[1])
    bc.close()
    server.close()
    np.save('./output/topic_embedding',vec)
    return vec


def tfidf_vec(text_flat_tokenized, stop_word):
    """returns tfidf weight for each word in each issue"""    
    def dummy_doc(doc):
        return doc

    vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words=stop_word,
        tokenizer=dummy_doc,
        preprocessor=dummy_doc,
        token_pattern=None)  
    X = vectorizer.fit_transform(text_flat_tokenized)
    feature_names = vectorizer.get_feature_names()
    tfidf_biglist = []
    for issue_num in range(len(text_flat_tokenized)):
        feature_index = X[issue_num,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [X[issue_num, x] for x in feature_index])
        tfidf_dict = {}
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            tfidf_dict[w] = s
        tfidf_biglist.append(tfidf_dict)
    return tfidf_biglist


def cosine_similarity(word_emb_a, word_emb_b):
    """Cosine similarity between two embeddings"""
    cos_sim = np.dot(word_emb_a, word_emb_b)/np.linalg.norm(word_emb_a)/np.linalg.norm(word_emb_b)
    return cos_sim


def get_word_embedding(text_one_issue, port=5000, port_out=5001, model_path='/home/ubuntu/bert_tests/bert-as-service/uncased_L-12_H-768_A-12'):
    """returns word embedding for each issue,
    not for all issues since vec will be too big.""" 
    common = [
        '-model_dir', model_path,
        '-num_worker', '2',
        '-port', str(port),
        '-port_out', str(port_out),
        '-max_seq_len', '20',
        '-max_batch_size', '256',
        '-pooling_strategy', 'NONE',
        '-pooling_layer', '-2',
        '-cpu',
        '-show_tokens_to_client',
    ]
    args = get_args_parser().parse_args(common)
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(20)
    print('encoding...')
    bc = BertClient(port=port, port_out=port_out, show_server_config=False)
    vec = bc.encode(text_one_issue,show_tokens=True,is_tokenized=False)
    bc.close()
    server.close()
    word_vec = vec[0]
    #tokens= vec[1]
    np.save('../output/newspaper_embedding',word_vec)
    return vec


def get_word_embedding_server_on(text_one_issue, port=5000, port_out=5001):
    """With BertServer turned on, returns word embedding for each issue,
    not for all issues since vec will be too big.""" 
    bc = BertClient(port=port, port_out=port_out, show_server_config=False)
    vec = bc.encode(text_one_issue,show_tokens=True,is_tokenized=False)
    bc.close()
    #word_vec = vec[0]
    #tokens= vec[1]
    #np.save('../output/newspaper_embedding',word_vec)
    return vec


def get_topics_one_issue(vec,topic_embedding,topics, divide_list_issue,tfidf_biglist,
                         issue_num, n_topics):
    """Compare the cosine similarity between articles per issue to topic embeddings"""
    bert_vecs = vec[0]
    bert_tokens = vec[1]
    tmp_sum = np.zeros((vec[0].shape[0], vec[0].shape[2]))
    for num, toks in enumerate(bert_tokens):        
        for word_idx, tok in enumerate(toks):
            if tok in tfidf_biglist[issue_num]:
                weight = tfidf_biglist[issue_num][tok]
                tmp_sum[num, :] += bert_vecs[num, word_idx]*weight
    topics_issue = set()
    topics_issue_list = []
    for i in range (len(divide_list_issue)-1):
        for j in range(len(topics)):
            if i == 0:
                article_vec = np.sum(tmp_sum[0:divide_list_issue[i]],axis =0)
                sim = cosine_similarity(article_vec, topic_embedding[j,:])
                if sim > 0.7:
                    #topics_issue.add(topics[j])
                    topics_issue_list.append([topics[j], sim])
            else:
                article_vec = np.sum(tmp_sum[divide_list_issue[i]:divide_list_issue[i+1]],axis =0)
                sim = cosine_similarity(article_vec, topic_embedding[j,:])
                if sim > 0.7:
                    #topics_issue.add(topics[j])
                    topics_issue_list.append([topics[j], sim])
    sort_topic_sim = sorted(topics_issue_list, key=lambda x:x[1], reverse=True)
    tmp = zip(*sort_topic_sim)
    topics = list(tmp)[0]
    if n_topics <= len(topics):
        topics_issue = set(topics[:n_topics])
    else:
        topics_issue = set(topics)
    topics_issue = list(topics_issue)
    return topics_issue, sort_topic_sim


def expand_stopwords():
    nltk.download('stopwords')
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    symbol_list = [chr(i) for i in range(127)]
    my_stopwords = ['—','“','”','‘','’','would','could','shall']
    bert_stop = []
    #add all ##letter
    for letter in symbol_list[97:123]:
        bert_stop.append('##'+letter)
    stop_words=list(set(nltk_stopwords+symbol_list+my_stopwords+bert_stop))
    return stop_words


def get_topics_one_issue_test(vec,topic_embedding,topics, divide_list_issue,
                         issue_num, n_topics):
    """Compare the cosine similarity between articles per issue to topic embeddings"""
    bert_vecs = vec[0]
    bert_tokens = vec[1]
    tmp_sum = np.zeros((vec[0].shape[0], vec[0].shape[2]))
    for num, toks in enumerate(bert_tokens):
        for word_idx, tok in enumerate(toks):
            #if tok in tfidf_biglist[issue_num]:
                #weight = tfidf_biglist[issue_num][tok]
            weight = 1
            tmp_sum[num, :] += bert_vecs[num, word_idx]*weight
    topics_issue = set()
    topics_issue_list = []
    for i in range (len(divide_list_issue)-1):
        for j in range(len(topics)):
            if i == 0:
                article_vec = np.sum(tmp_sum[0:divide_list_issue[i]],axis =0)
                sim = cosine_similarity(article_vec, topic_embedding[j,:])
                if sim > 0.7:
                    #topics_issue.add(topics[j])
                    topics_issue_list.append([topics[j], sim])
            else:
                article_vec = np.sum(tmp_sum[divide_list_issue[i]:divide_list_issue[i+1]],axis =0)
                sim = cosine_similarity(article_vec, topic_embedding[j,:])
                if sim > 0.7:
                    #topics_issue.add(topics[j])
                    topics_issue_list.append([topics[j], sim])
    sort_topic_sim = sorted(topics_issue_list, key=lambda x:x[1], reverse=True)
    tmp = zip(*sort_topic_sim)
    topics = list(tmp)[0]
    if n_topics <= len(topics):
        topics_issue = set(topics[:n_topics])
    else:
        topics_issue = set(topics)
    topics_issue = list(topics_issue)
    return topics_issue, sort_topic_sim

