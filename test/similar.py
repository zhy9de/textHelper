# coding=utf8

from textHelper import TextHelper

th = TextHelper(0.5)

doc1 = '我今天去了动物园非常开心啊'
doc2 = '我今天去了上方山,一点都不开心'
print th.compare_similarity(doc1, doc2)
