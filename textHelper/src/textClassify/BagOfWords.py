# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:25:58 2015

@author: hehe
"""

import logging
import os
import shutil

import numpy
from scipy import sparse
from sklearn.externals import joblib


class BagOfWords:
    def __init__(self, jt, root_path):
        self.jt = jt
        self.path = root_path

    def del_old(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        logging.info('del old model')

    def build_dictionary(self, dict):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.dict = BagOfWords.reduce_dict(dict)

        self.save_dictionary(self.path + '/dicitionary.pkl')

    def load_dictionary(self):
        filename = self.path + '/dicitionary.pkl'
        import cPickle
        try:
            print "loaded dictionary from %s" % filename
            self.dict = cPickle.load(open(filename, 'rb'))
            print "done"
        except IOError:
            print "error while loading from %s" % filename
            return False

    def save_dictionary(self, filename):
        import cPickle
        cPickle.dump(self.dict, open(filename, 'wb'))
        print "saved dictionary to %s" % filename

    def save_model(self, logreg):
        filename = self.path + '/lr.model'
        joblib.dump(logreg, filename)

    def load_model(self):
        filename = self.path + '/lr.model'
        import os
        if os.path.isfile(filename):
            return joblib.load(filename)
        else:
            return False

    @staticmethod
    def reduce_dict(dict_set):
        dict_copy = dict_set.copy()
        for word in dict_set:
            if len(word) < 2:
                dict_copy.remove(word)
            else:
                try:
                    float(word)
                    dict_copy.remove(word)
                except ValueError:
                    continue
        dictionary = {}
        for idx, word in enumerate(dict_copy):
            dictionary[word] = idx
        return dictionary

    def transform_data(self, coll):
        print "transforming data in to bag of words vector"
        data = []
        target = []
        for art in coll.find():
            tag = art['classify_id']
            word_vector = self.trainsorm_single(art['content'])
            # data.append(sparse.csr_matrix(word_vector))
            data.append(word_vector)
            target.append(tag)
        print "done"
        return sparse.csr_matrix(numpy.asarray(data)), numpy.asarray(target)

    def trainsorm_single(self, art):
        word_vector = numpy.zeros(len(self.dict))
        words = self.jt.tokens(art)
        for word in words:
            try:
                word_vector[self.dict[word]] += 1
            except KeyError:
                pass
        return word_vector
