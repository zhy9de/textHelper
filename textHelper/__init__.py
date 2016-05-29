# coding=utf8
import os

from sklearn import linear_model
from sklearn.externals import joblib

from src.textClassify import TextClassify
from src.textClassify.BagOfWords import BagOfWords
from src.textSimilarity.Utils import cosine_distance_nonzero
from src.textSimilarity.features import FeatureBuilder
from src.textSimilarity.isSimilar import DocFeatLoader
from src.textSimilarity.simhash_imp import SimhashBuilder, hamming_distance
from src.tokens import JiebaTokenizer

p = os.path.split(os.path.realpath(__file__))[0]
stop_words_file = os.path.realpath(p + '/data/') + '/stopwords.txt'
words_bag_path = os.path.realpath(p + '/data/wordsBag')


class TextHelper:
    def __init__(self, threshold, stop_words='', words_bag_root='', mode='c'):
        if stop_words:
            self.stop_words_file = stop_words
        else:
            self.stop_words_file = stop_words_file

        if words_bag_root:
            self.words_bag_root = words_bag_root
        else:
            self.words_bag_root = words_bag_path

        self.threshold = threshold
        self.jt = JiebaTokenizer(self.stop_words_file, mode=mode)

    def compare_similarity(self, input_tpl, compare_tpl, way=2):
        # 检测文本编码
        if not isinstance(input_tpl, unicode):
            input_tpl = input_tpl.decode('utf8')
        if not isinstance(compare_tpl, unicode):
            compare_tpl = compare_tpl.decode('utf8')

        doc_token_1 = self.jt.tokens(input_tpl)
        doc_token_2 = self.jt.tokens(compare_tpl)

        word_list = list(set(doc_token_1 + doc_token_2))

        # Build unicode string word dict
        word_dict = {}
        for idx, ascword in enumerate(word_list):
            word_dict[ascword] = idx
        # Build nonzero-feature
        fb = FeatureBuilder(word_dict)
        doc_feat_1 = fb.compute(doc_token_1)
        doc_feat_2 = fb.compute(doc_token_2)

        # Init simhash_builder
        smb = SimhashBuilder(word_list)

        doc_fl_1 = DocFeatLoader(smb, doc_feat_1)
        doc_fl_2 = DocFeatLoader(smb, doc_feat_2)

        if way == 1:
            # print 'Matching by Simhash + hamming distance'
            dist = hamming_distance(doc_fl_1.fingerprint, doc_fl_2.fingerprint)
            if dist < float(self.threshold):
                return True, dist
            else:
                return False, dist
        elif way == 2:
            # print 'Matching by VSM + cosine distance'
            dist = cosine_distance_nonzero(doc_fl_1.feat_vec, doc_fl_2.feat_vec, norm=False)
            if dist > float(self.threshold):
                return True, dist
            else:
                return False, dist

    # 初始化字典,要扔数据集进来
    def init_bag(self, coll, del_old=True):
        self.words_bag = BagOfWords(self.jt, self.words_bag_root)
        if del_old:
            self.words_bag.del_old()
        # rebuild dict
        dict_set = set()
        for data in coll.find():
            words = self.jt.tokens(data['content'])
            dict_set |= set(words)
        self.words_bag.build_dictionary(dict_set)

        train_feature, train_target = self.words_bag.transform_data(coll)

        logreg = linear_model.LogisticRegression(C=1e5)
        logreg.fit(train_feature, train_target)

        self.words_bag.save_model(logreg)

    def classify(self, text):
        # init model
        lr = joblib.load('lr.model')
        # init bow
        BOW = self.words_bag.load_dictionary()

        # TextClassify
        pred = TextClassify.find_classify(text, BOW, lr)
        return pred[0]
