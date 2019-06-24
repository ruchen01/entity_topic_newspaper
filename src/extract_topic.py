import numpy as np
import glob
import re
import nltk
from pytorch_pretrained_bert import BertTokenizer
import time
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from sklearn.feature_extraction.text import TfidfVectorizer


def bert_tokens(text_all, token_choice='bert-base-uncased'):
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


def get_topic_embedding(topics, port=6000, port_out=6001, model_path='/home/ubuntu/bert_tests/bert-as-service/uncased_L-12_H-768_A-12'):
    common = [
        '-model_dir', model_path,
        '-num_worker', '2',
        '-port', str(port),
        '-port_out', str(port_out),
        '-max_seq_len', '20',
        '-max_batch_size', '256',
        '-pooling_layer', '-2',
        '-cpu',
    ]
    args = get_args_parser().parse_args(common)
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(20)
    print('encoding...')
    bc = BertClient(port=port, port_out=port_out, show_server_config=False)
    vec = bc.encode(topics)
    bc.close()
    server.close()
    np.save('./out/topic_embedding',vec)
    return vec


def tfidf_vec(text_flat_tokenized, stopwords):
    """returns tfidf weight for each word in each issue"""    
    def dummy_doc(doc):
        return doc

    vectorizer = TfidfVectorizer(
        analyzer='word',
        stop_words=stopwords,
        tokenizer=dummy_doc,
        preprocessor=dummy_doc,
        token_pattern=None)  
    X = vectorizer.fit_transform(text_flat_tokenized)
    feature_names=vectorizer.get_feature_names()
    tfidf_biglist=[]
    for issue_num in range(len(text_flat_tokenized)):
        feature_index =X[issue_num,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [X[issue_num, x] for x in feature_index])
        tfidf_dict={}
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            tfidf_dict[w]=s
        tfidf_biglist.append(tfidf_dict)
    return tfidf_biglist


def cosine_similarity(word_emb_a, word_emb_b):
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
        'show_tokens_to_client',
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
    word_vec= vec[0]
    #tokens= vec[1]
    np.save('./out/newspaper_embedding',word_vec)
    return vec


def get_topics_one_issue(vec,topic_embedding,topics, divide_list_issue,tfidf_biglist,
                         issue_num):
    bert_vecs = vec[0]
    bert_tokens = vec[1]
    tmp_sum = np.zeros((vec[0].shape[0], vec[0].shape[2]))
    for num, toks in enumerate(bert_tokens):        
        for word_idx, tok in enumerate(toks):
            if tok in tfidf_biglist[issue_num]:
                weight = tfidf_biglist[issue_num][tok]
                tmp_sum[num, :] += bert_vecs[num, word_idx]*weight
    topics_issue = set()
    for i in range (divide_list_issue-1):
        for j in range(len(topics)):
            if i == 0:
                article_vec = np.sum(tmp_sum[0:divide_list_issue[i]],axis =0)
                sim = cosine_similarity(article_vec, topic_embedding[j,:])
                if sim > 0.7:
                    topics_issue.add(topics[j])
            else:
                article_vec = np.sum(tmp_sum[divide_list_issue[i]:divide_list_issue[i+1]],axis =0)
                sim = cosine_similarity(article_vec, topic_embedding[j,:])
                if sim > 0.7:
                    topics_issue.add(topics[j])
    return topics_issue


def issue2articles(data):
    """Break each issue into articles based on the style of titles"""
    divide_li = []
    for i, line in enumerate(data):
        if i<len(data):
            tok = nltk.word_tokenize(line)
            tok_cap = []
            for word in tok:
                if len(word)==1:
                    cap_word = re.findall('([A-Z])', word)
                else:
                    cap_word = re.findall('([A-Z]+(?:(?!\s?[A-Z][a-z])\s?[A-Z])+)', word)
                if cap_word != []:
                    tok_cap.append(cap_word[0]) 
            if tok_cap != [] and len(tok)-len(tok_cap)<=1 and len(tok)>2:
                divide_li.append(i)
    articles = []
    for i,ind in enumerate(divide_li):
        if i+1<= len(divide_li)-1:
            if divide_li[i+1]-ind>5:
                articles.append(data[ind:divide_li[i+1]])
        else:
            articles.append(data[ind:])
    # The first article per issue is always the headline and therefore is dropped
    articles.pop(0);
    articles_list=[]
    for article in articles:
        data = " ".join(line.strip() for line in article) 
        articles_list.append(data)
    return articles_list


def articles2sentences(articles_list):
    """Break each article into sentences for each issue. issue_sentence is a flat list 
    of sentences and issue_article_sentence is sub sub list. """
    issue_sentence = []
    issue_article_sentence=[]
    for article in articles_list:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(article)
        article_sentence=[]
        for sentence in sentences:
            if len(sentence)>5:
                issue_sentence.append(sentence)
                article_sentence.append(sentence)
        issue_article_sentence.append(article_sentence)
    return issue_sentence, issue_article_sentence


def find_divide_articlue_line_number(issue_article_sentence):
    """Find the line number which divide each issue into articles """
    divide_num = 0
    divide_list = []
    for idx, article in enumerate(issue_article_sentence):
        divide_num += len(article)
        divide_list.append(divide_num)
    return divide_list


def combine_issues(issues_path):
    """combine all issues, return text_all[i][j]= a sentence, 
    where i is issue_num and j is sentence index"""
    text_all = []
    divide_list=[]
    i = 0  
    issue_files = glob.glob(issues_path+'*.txt')
    i_max = len(issue_files)
    for issue_file in issue_files:
        if i<i_max:
            with open(issue_file) as f:
                data = f.readlines()
            articles_list = issue2articles(data)
            issue_sentence, issue_article_sentence=articles2sentences(articles_list)
            divide_list_each = find_divide_articlue_line_number(issue_article_sentence)
            divide_list.append(divide_list_each)
            text_all.append(issue_sentence)
            i+=1
        else:
            break
    return text_all


def expand_stopwords():
    nltk.download('stopwords')
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    symbol_list = [chr(i) for i in range(127)]
    my_stopwords = ['—','“','”','‘','’','would','could','shall']
    bert_stop = []
    #add all ##letter
    for letter in symbol_list[97:123]:
        bert_stop.append('##'+letter)
    stopwords=list(set(nltk_stopwords+symbol_list+my_stopwords+bert_stop))
    return stopwords
