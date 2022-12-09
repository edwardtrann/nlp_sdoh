import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import re

# from bert_classifier import getTrainVal
#
#
# train_x, val_x, train_y, val_y = getTrainVal()
#
# train_x, val_x, train_y, val_y = tf.convert_to_tensor(train_x), tf.convert_to_tensor(val_x), tf.convert_to_tensor(train_y), tf.convert_to_tensor(val_y)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
nltk_stopwords_dir = "/Users/ryandwyer/Downloads/Fall 22/CS 221/Final Project/nlp_sdoh/nltk_stopwords.txt"
STOPWORDS = set(stopword.strip() for stopword in open(nltk_stopwords_dir).readlines())


# df = pd.read_csv("combined_data_v3.csv", usecols=["uniq_id", "text", "outcome"])
df = pd.read_csv("combined_data_v3.csv")
# x = tf.convert_to_tensor(df)
# print(type(x))

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

df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')

train_ids = [1, 2, 3, 4, 5, 8, 9, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 50, 52, 55, 58, 59, 60, 62, 64, 65, 66, 68, 70, 73, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 87, 89, 92, 93, 94, 96, 97, 99, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 157, 158, 160, 161, 162, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 221, 224, 225, 226, 227, 228, 230, 231, 233, 235, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 258, 259]
train_y = tf.convert_to_tensor([True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, True, True, False, True, False, False, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False, True, False, False, True, True, False, True, True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True])
val_ids = [100, 35, 34, 54, 91, 53, 61, 76, 90, 7, 72, 51, 83, 56, 63, 57, 18, 6, 67, 47, 248, 107, 250, 163, 143, 134, 223, 234, 29, 242, 257, 155, 167, 220, 232, 156, 113, 180, 222, 195]
val_y = tf.convert_to_tensor([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
test_ids = [71, 80, 49, 98, 13, 19, 69, 10, 95, 44, 236, 88, 148, 212, 130, 159, 129, 229, 203, 207]
test_y = [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False]

train_x = tf.convert_to_tensor(df[df['uniq_id'].isin(set(train_ids))][['text']])
val_x = tf.convert_to_tensor(df[df['uniq_id'].isin(set(val_ids))][['text']])

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=True)

#to test that my encoder and preprocessor work (they do)
# def get_sentence_embedding(sentences):
#     preprocessed_text = preprocessor(sentences)
#     return encoder(preprocessed_text)['pooled_output']
#
# print(get_sentence_embedding(["$500 discount. Hurry up"]))

#To test that it works on general data
# def get_sentence_embedding(sentences):
#     preprocessed_text = preprocessor(sentences)
#     return encoder(preprocessed_text)['pooled_output']
#
# for i in range(len(train_textsplit)):
#     print(get_sentence_embedding(train_textsplit[i]))

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
preprocessed_text = preprocessor(text_input)
outputs = encoder(preprocessed_text)

l = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(l)
model = tf.keras.Model(inputs=[text_input], outputs=[l])

#print(model.summary())

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

print(model.fit(train_x, train_y, epochs=10))

print(model.evaluate(val_x, val_y))

y_pred = model.predict(val_x)

print(y_pred.flatten())

