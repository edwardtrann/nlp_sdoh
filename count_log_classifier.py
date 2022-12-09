import pandas as pd
from tdf_if_log_classifier import get_uniq_id_docs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


data_df = pd.read_csv("combined_data_v3.csv", usecols=["uniq_id", "outcome", "text"])
train_ids = [1, 2, 3, 4, 5, 8, 9, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 50, 52, 55, 58, 59, 60, 62, 64, 65, 66, 68, 70, 73, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 87, 89, 92, 93, 94, 96, 97, 99, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 157, 158, 160, 161, 162, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 221, 224, 225, 226, 227, 228, 230, 231, 233, 235, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 258, 259]
train_y = [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, True, True, False, True, False, False, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, False, False, False, True, False, False, True, True, False, True, True, True, True, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True]
val_ids = [100, 35, 34, 54, 91, 53, 61, 76, 90, 7, 72, 51, 83, 56, 63, 57, 18, 6, 67, 47, 248, 107, 250, 163, 143, 134, 223, 234, 29, 242, 257, 155, 167, 220, 232, 156, 113, 180, 222, 195]
val_y = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
test_ids = [71, 80, 49, 98, 13, 19, 69, 10, 95, 44, 236, 88, 148, 212, 130, 159, 129, 229, 203, 207]
test_y = [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False]
count_vectorizer = CountVectorizer(stop_words={'english'})
train_docs = get_uniq_id_docs(data_df, train_ids)
train_X = count_vectorizer.fit_transform(train_docs).toarray()
count_clf = LogisticRegression(max_iter=1000).fit(train_X, train_y)

print(f"\n\n___training___\n")
train_pred_y = count_clf.predict(train_X)
train_pred_proba_y = count_clf.predict_proba(train_X)

threshold = 0.7

un, up, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
for train_id, y, pred_y, pred_proba_y in zip(train_ids, train_y, train_pred_y, train_pred_proba_y):
    if max(pred_proba_y) < threshold:
        if y:
            up += 1
        else:
            un += 1
    elif y:
        if pred_y:
            tp += 1
        else:
            fn += 1
    else:
        if pred_y:
            fp += 1
        else:
            tn += 1
print(f"up: {up}")
print(f"tp: {tp}")
print(f"fp: {fp}")
print(f"tn: {tn}")
print(f"fn: {fn}")

print(f"positive undetermined rate = {up / (tp + fn + up)}")
print(f"negative undetermined rate = {un / (tn + fp + un)}")
print(f"undetermined rate = {(un + up) / (un + up + tp + fp + tn + fn)}")
print(f"positive predictive value = {tp / (tp + fp)}")
print(f"negative predictive value = {tn / (tn + fn)}")
print(f"sensitivity = {tp / (tp + fn)}")
print(f"specificity = {tn / (fp + tn)}")

print(f"\n\n___validation___\n")
val_docs = get_uniq_id_docs(data_df, val_ids)
val_X = count_vectorizer.transform(val_docs).toarray()
val_pred_y = count_clf.predict(val_X)
val_pred_proba_y = count_clf.predict_proba(val_X)

un, up, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
for val_id, y, pred_y, pred_proba_y in zip(val_ids, val_y, val_pred_y, val_pred_proba_y):
    if max(pred_proba_y) < threshold:
        if y:
            up += 1
        else:
            un += 1
    elif y:
        if pred_y:
            tp += 1
        else:
            fn += 1
    else:
        if pred_y:
            fp += 1
        else:
            tn += 1
print(f"up: {up}")
print(f"tp: {tp}")
print(f"fp: {fp}")
print(f"tn: {tn}")
print(f"fn: {fn}")

print(f"positive undetermined rate = {up / (tp + fn + up)}")
print(f"negative undetermined rate = {un / (tn + fp + un)}")
print(f"undetermined rate = {(un + up) / (un + up + tp + fp + tn + fn)}")
print(f"positive predictive value = {tp / (tp + fp)}")
print(f"negative predictive value = {tn / (tn + fn)}")
print(f"sensitivity = {tp / (tp + fn)}")
print(f"specificity = {tn / (fp + tn)}")


print(f"\n\n___test___\n")
test_docs = get_uniq_id_docs(data_df, test_ids)
test_X = count_vectorizer.transform(test_docs).toarray()
test_pred_y = count_clf.predict(test_X)
test_pred_proba_y = count_clf.predict_proba(test_X)

un, up, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
for test_id, y, pred_y, pred_proba_y in zip(test_ids, test_y, test_pred_y, test_pred_proba_y):
    if max(pred_proba_y) < threshold:
        if y:
            up += 1
        else:
            un += 1
    elif y:
        if pred_y:
            tp += 1
        else:
            fn += 1
    else:
        if pred_y:
            fp += 1
        else:
            tn += 1
print(f"up: {up}")
print(f"tp: {tp}")
print(f"fp: {fp}")
print(f"tn: {tn}")
print(f"fn: {fn}")

print(f"positive undetermined rate = {up / (tp + fn + up)}")
print(f"negative undetermined rate = {un / (tn + fp + un)}")
print(f"undetermined rate = {(un + up) / (un + up + tp + fp + tn + fn)}")
print(f"positive predictive value = {tp / (tp + fp)}")
print(f"negative predictive value = {tn / (tn + fn)}")
print(f"sensitivity = {tp / (tp + fn)}")
print(f"specificity = {tn / (fp + tn)}")
