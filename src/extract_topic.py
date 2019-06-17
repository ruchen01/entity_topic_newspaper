import nltk
import glob
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def generate_text_list(text_dir):
"""group all articles into a list""" 
    text_list = glob.glob(text_dir+'/*.txt')
    text_data = []
    for file_name in text_list:
        with open(file_name) as f:
            #data = f.readlines()
            data = " ".join(line.strip() for line in f)
            text_data.append(data)
    return text_data


class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        lemm = WordNetLemmatizer()
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


def process_text2vec(text):
    cnt_vectorizer = LemmaCountVectorizer(max_df=0.95,
                                         min_df=2,
                                         stop_words='english',
                                         decode_error='ignore')
    cnt_mat = cnt_vectorizer.fit_transform(text)
    return cnt_mat, cnt_vectorizer


def model(cnt_mat, n_comp = 4):
    lda = LatentDirichletAllocation(n_components=n_comp, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)
    lda.fit(cnt_mat)
    return lda


def print_top_words(model, feature_names, n_top_words=10):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)


def print_topic(text_dir, out_dir, n_top_words = 10):
    text = generate_text_li(text_dir)
    tf, tf_vectorizer = process_text2vec(text)
    lda = model(tf, n_comp = 4)
    #print("\n Assigning Topics in LDA model: ")
    feature_names = tf_vectorizer.get_feature_names()
    with open(out_dir+'/topic.txt', 'w') as f:
        for index, topic in enumerate(lda.components_):
            f.writelines("\n"+" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]]))
            
 
