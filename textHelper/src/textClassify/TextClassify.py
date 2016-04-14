# -*- coding: utf-8 -*-


class TextClassify:
    def __init__(self):
        pass

    @staticmethod
    def find_classify(art, bow_model, classify_model):
        feature = bow_model.trainsorm_single(art)
        pred = classify_model.predict(feature)
        return pred
