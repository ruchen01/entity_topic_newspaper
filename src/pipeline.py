import glob
import tensorflow as tf
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
import time
import os
import shutil
from src.process import Newspaper
import src.extract_topic as tp
from src.extract_entity import extract_name_all
import src.get_ocr as get_ocr
import pickle


def combine_issues(issues_path):
    """combine all issues, return text_all[i][j]= a sentence, 
    where i is issue_num and j is sentence index"""
    text_all = []
    divide_list = []
    i = 0  
    issue_files = glob.glob(issues_path+'/*.txt')
    i_max = len(issue_files)
    for issue_file in issue_files:
        if i < i_max:
            x = Newspaper()
            x.load_text(issue_file)
            article_list = x.issue2articles()
            issue_sentence, issue_article_sentence=x.articles2sentences(article_list)
            divide_list_each = x.find_divide_article_line_number(issue_article_sentence)                
                
            divide_list.append(divide_list_each)
            text_all.append(issue_sentence)
            i += 1
        else:
            break
    return text_all, divide_list


def extract_topics_all(issues_path, model_dir, topic_file, n_topics):
    """Extract topics for all issues with top n_topics topics"""
    topic_all = []
    text_all, divide_list = combine_issues(issues_path)
    topics = tp.get_topic_list(topic_file)
    topic_embedding = tp.get_topic_embedding(topics, port=3500, port_out=3501, model_path=model_dir)
    #topic_embedding = np.load('../output/topic_embedding.npy')
    print('topic embedding shape = ', topic_embedding.shape)
    stop_words = tp.expand_stopwords()
    print(len(stop_words))
    text_flat_tokenized, text_article_tokenized = tp.bert_tokens(text_all)
    tfidf_biglist = tp.tfidf_vec(text_flat_tokenized, stop_words)
    port_in = 6550
    port_out = 6551
    tmp_dir = '../output/tmp'
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    ZEROMQ_SOCK_TMP_DIR=tmp_dir
    common = [
        '-model_dir', model_dir,
        '-num_worker', '2',
        '-port', str(port_in),
        '-port_out', str(port_out),
        '-max_seq_len', '20',
        '-max_batch_size', '256',
        '-pooling_strategy', 'NONE',
        '-pooling_layer', '-2',
        '-graph_tmp_dir', tmp_dir,
        '-cpu',
        '-show_tokens_to_client',
    ]
    args = get_args_parser().parse_args(common)
    server = BertServer(args)
    server.start()
    print('wait until server is ready...')
    time.sleep(20)
    print('encoding...')        
    for issue_num in range(len(text_all)):  
        issue_num = 0
        divide_list_each = divide_list[issue_num]
        text_one_issue = text_all[issue_num]
        vec = tp.get_word_embedding_server_on(text_one_issue, port=port_in, port_out=port_out)
        topics_issue, sort_topic_sim = tp.get_topics_one_issue(vec,topic_embedding,topics, divide_list_each, 
                                               tfidf_biglist, issue_num, n_topics)
        topic_all.append(topics_issue)       
    server.close()
    topic_folder = './output/topic'
    if not os.path.isdir(topic_folder):
        os.makedirs(topic_folder)
    with open(topic_folder + '/topic.pkl', 'wb') as f:
        pickle.dump(topic_all, f)
    return topic_all


def major_pipeline(img_dir, output_text_dir, model_dir, topic_file, name_file, model_conll_dir, embedding_file, n_topics = 5):
    """end-to-end pipeline"""
    get_ocr.imgs_to_texts_save_local(img_dir=img_dir, output_dir=output_text_dir)
    topic_all = extract_topics_all(output_text_dir, model_dir, topic_file, n_topics)
    extract_name_all(output_text_dir, name_file, model_conll_dir, embedding_file)
    cleaning_file()


def test_gpu():
    """Automatically detect if system has GPU"""
    if tf.test.gpu_device_name():
        gpu = True
    else:
        gpu = False
    return gpu


def cleaning_file():
    """Delete unuseful tf graphs"""
    tmp_folders = glob.glob('./tmp*')
    if len(tmp_folders) >= 1:
        for tmp in tmp_folders:
            shutil.rmtree(tmp)


def test_pipeline(path):
    """Test model with additional images"""
    print(glob.glob(path + '/*.txt'))
    print(os.getcwd())
