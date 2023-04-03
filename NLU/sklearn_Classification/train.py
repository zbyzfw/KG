# -*- coding:utf-8 -*-

import os
import pickle
import random
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

seed = 222
random.seed(seed)
np.random.seed(seed)

def load_data(data_path):
    X,y = [],[]
    with open(data_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            text,label = line.strip().split(',')
            text = ' '.join(list(text.lower()))
            X.append(text)
            y.append(label)

    index = np.arange(len(X))
    np.random.shuffle(index)
    X = [X[i] for i in index]
    y = [y[i] for i in index]
    return X,y

def run(data_path,model_save_path):
    X,y = load_data(data_path)

    label_set = sorted(list(set(y)))
    label2id = {label:idx for idx,label in enumerate(label_set)}
    id2label = {idx:label for label,idx in label2id.items()}

    y = [label2id[i] for i in y]

    label_names = sorted(label2id.items(), key = lambda kv:kv[1], reverse=False)
    target_names = [i[0] for i in label_names]
    labels = [i[1] for i in label_names]

    train_X, text_X, train_y, text_y = train_test_split(X, y, test_size=0.15, random_state=42)
    # TfidfVectorizer可以将文本数据转换为数值特征，使用词频-逆文档频率（TF-IDF）的方法。TF-IDF是一种衡量单词在文档中的重要性的方法，它考虑了单词在文档中出现的频率和在整个语料库中出现的频率。TfidfVectorizer可以对文本进行分词、去除停用词、归一化等预处理操作，然后计算每个单词的TF-IDF值，并返回一个稀疏矩阵。
    vec = TfidfVectorizer(ngram_range=(1,3),min_df=0, max_df=0.9,analyzer='char',use_idf=1,smooth_idf=1, sublinear_tf=1)
    train_X = vec.fit_transform(train_X)
    text_X = vec.transform(text_X)

    # svc_clf = svm.LinearSVC(tol=0.00001, C=6.0, multi_class='ovr',class_weight='balanced',random_state=122, max_iter=1500)
    # -------------LR--------------
    # LogisticRegression可以实现逻辑回归模型，用于二分类或多分类问题。逻辑回归模型是一种线性模型，它使用一个对数几率函数（logit function）将输入特征映射到一个概率值，然后根据概率值判断类别。LogisticRegression可以使用不同的损失函数和正则化方法来优化模型参数，并提供了多种求解器选项。
    LR = LogisticRegression(C=8, dual=False,n_jobs=4,max_iter=400,multi_class='ovr',random_state=122)
    LR.fit(train_X, train_y)
    pred = LR.predict(text_X)
    # classification_report可以生成一个文本报告，显示每个类别的精确度（precision）、召回率（recall）、F1分数（F1-score）和样本数（support）。这些指标可以反映模型对不同类别的预测能力和平衡性。classification_report可以接受真实标签（y_true）和预测标签（y_pred）作为输入，并返回一个字符串或一个字典。
    # confusion_matrix可以计算混淆矩阵（confusion matrix），用于评估分类模型的准确度。混淆矩阵是一个N x N的表格（N是类别数），它显示了模型对每个类别的正确和错误预测的数量。混淆矩阵可以帮助我们分析模型的误判情况和优化方向。confusion_matrix也可以接受真实标签（y_true）和预测标签（y_pred）作为输入，并返回一个数组。
    print(classification_report(text_y, pred,target_names=target_names))
    print(confusion_matrix(text_y, pred,labels=labels))

    # -------------gbdt--------------
    # GradientBoostingClassifier可以实现梯度提升树（Gradient Boosting Tree）模型，用于分类问题。梯度提升树模型是一种集成学习方法，它通过迭代地添加弱学习器（通常是回归树）来构建一个强学习器。在每一步，梯度提升树模型使用损失函数的负梯度作为残差来拟合新的弱学习器，并将其加权累加到之前的模型中。GradientBoostingClassifier可以使用不同的损失函数和树参数来调整模型性能，并提供了早停、剪枝等功能。
    gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.01,max_depth=8, random_state=24)
    gbdt.fit(train_X, train_y)
    pred = gbdt.predict(text_X)
    print(classification_report(text_y, pred,target_names=target_names))
    print(confusion_matrix(text_y, pred,labels=labels))

    # -------------融合--------------
    pred_prob1 = LR.predict_proba(text_X)
    pred_prob2 = gbdt.predict_proba(text_X)

    pred = np.argmax((pred_prob1+pred_prob2)/2, axis=1)
    print(classification_report(text_y, pred,target_names=target_names))
    print(confusion_matrix(text_y, pred,labels=labels))

    pickle.dump(id2label,open(os.path.join(model_save_path,'id2label.pkl'),'wb'))
    pickle.dump(vec,open(os.path.join(model_save_path,'vec.pkl'),'wb'))
    pickle.dump(LR,open(os.path.join(model_save_path,'LR.pkl'),'wb'))
    pickle.dump(gbdt,open(os.path.join(model_save_path,'gbdt.pkl'),'wb'))



if __name__ == '__main__':
    run("./data/intent_recog_data.txt", "./model_file/")