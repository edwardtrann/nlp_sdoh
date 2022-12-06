import numpy as np # linear algebra
import re, os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import datetime
from datetime import datetime


# BERT
import optimization
import run_classifier
import tokenization
import tensorflow_hub as hub

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
nltk_stopwords_dir = "/Users/omjahagirdar/PycharmProjects/CS221FinalProject/nltk_stopwords.txt"
STOPWORDS = set(stopword.strip() for stopword in open(nltk_stopwords_dir).readlines())


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    #     text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


def get_split(text1):
    l_total = []
    l_parcial = []
    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:200]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_parcial))
    return l_total


df = pd.read_csv("combined_data_v3.csv", usecols=["uniq_id", "text", "outcome"])
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')
df['text_split'] = df['text'].apply(get_split)

train_ids = [1, 2, 3, 4, 5, 8, 9, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 50, 52, 55, 58, 59, 60, 62, 64, 65, 66, 68, 70, 73, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 87, 89, 92, 93, 94, 96, 97, 99, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 157, 158, 160, 161, 162, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 221, 224, 225, 226, 227, 228, 230, 231, 233, 235, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 258, 259]
train_y = [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, True, True, False, True, False, False, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False, True, False, False, True, True, False, True, True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True]
val_ids = [100, 35, 34, 54, 91, 53, 61, 76, 90, 7, 72, 51, 83, 56, 63, 57, 18, 6, 67, 47, 248, 107, 250, 163, 143, 134, 223, 234, 29, 242, 257, 155, 167, 220, 232, 156, 113, 180, 222, 195]
val_y = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
test_ids = [71, 80, 49, 98, 13, 19, 69, 10, 95, 44, 236, 88, 148, 212, 130, 159, 129, 229, 203, 207]
test_y = [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False]



# Set the output directory for saving model file
OUTPUT_DIR = '/Users/omjahagirdar/PycharmProjects/CS221FinalProject/bert_output'

#@markdown Whether or not to clear/delete the directory and create a new one
# DO_DELETE = True #@param {type:"boolean"}

# if DO_DELETE:
#     try:
#         # tf.gfile.DeleteRecursively(OUTPUT_DIR)
#         tf.compat.v1.gfile.DeleteRecursively(OUTPUT_DIR)
#     except:
#         pass
#
# tf.gfile.MakeDirs(OUTPUT_DIR)
# print('***** Model output directory: {} *****'.format(OUTPUT_DIR))


# Use the InputExample class from BERT's run_classifier code to create examples from the data
DATA_COLUMN = 'text'
LABEL_COLUMN = 'outcome'
train = df[df['uniq_id'].isin(set(train_ids))][['text', 'outcome', 'text_split']]
val = df[df['uniq_id'].isin(set(val_ids))][['text', 'outcome', 'text_split']]

train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN],
                                                                   text_b = None,
                                                                   label = x[LABEL_COLUMN]), axis = 1)

val_InputExamples = val.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN],
                                                                   text_b = None,
                                                                   label = x[LABEL_COLUMN]), axis = 1)

print(train_InputExamples)
print(val_InputExamples)

# un, up, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
# for val_id, y, pred_y, pred_proba_y in zip(val_ids, val_y, val_pred_y, val_pred_proba_y):
#     print(f"{y}\t{pred_y}\t{pred_proba_y}")
#     if max(pred_proba_y) < 0.7:
#         if y:
#             up += 1
#         else:
#             un += 1
#     elif y:
#         if pred_y:
#             tp += 1
#         else:
#             fn += 1
#     else:
#         if pred_y:
#             fp += 1
#         else:
#             tn += 1
# print(f"up: {up}")
# print(f"tp: {tp}")
# print(f"fp: {fp}")
# print(f"tn: {tn}")
# print(f"fn: {fn}")
#
# print(f"positive undetermined rate = {up / (tp + fn + up)}")
# print(f"negative undetermined rate = {un / (tn + fp + un)}")
# print(f"undetermined rate = {(un + up) / (un + up + tp + fp + tn + fn)}")
# print(f"positive predictive value = {tp / (tp + fp)}")
# print(f"negative predictive value = {tn / (tn + fn)}")
# print(f"sensitivity = {tp / (tp + fn)}")
# print(f"specificity = {tn / (fp + tn)}")
