#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import os, time, math
import numpy as np
import pandas as pd
import zipfile, pickle, h5py, joblib, json

import librosa
import opensmile
import tensorflow_hub as hub

from math import pi
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from scipy.fftpack import fft, hilbert
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score

from config import *
from utils import *
from feature import *
from model import *

## read data csv and set up label
df = pd.read_csv(DIR_DATA + 'data.csv')
df['file_path'] = df['file_cough'] + '.wav'
df['label_symptom'] = (df['symptoms_status_choice'].map(str) != "['No']").astype(int)
df['label_abnormal'] = ((df['symptoms_status_choice'].map(str) != "['No']") | (df['cov19_status_choice'] != 'never')).astype(int)
df['label_covid'] = (df['cov19_status_choice'] != 'never').astype(int)

## create feature and cache
if not os.path.exists(OUTPUT_DIR + 'FRILL.pickle'):
    X_trill_features = get_features_of_list_audio(DIR_DATA, df)
    pickle.dump({
        'X_trill_features': X_trill_features
    }, open(OUTPUT_DIR + 'FRILL.pickle', "wb" ))
else:
    f = pickle.load(open(OUTPUT_DIR + 'FRILL.pickle', "rb" ))
    X_trill_features     = f['X_trill_features']

## Training COVID-19
print("Training COVID-19")
y = df['label_covid']
pos_scale = (y == 0).sum() / (y == 1).sum()

### load fold split
if not os.path.exists(OUTPUT_DIR + 'df_label_covid_5fold.csv'):
    folds = df.copy()
    Fold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label_covid'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    folds.to_csv(OUTPUT_DIR + 'df_label_covid_5fold.csv', index=False)
else:
    folds = pd.read_csv(OUTPUT_DIR + 'df_label_covid_5fold.csv')

### training
X = merge_feature([X_trill_features])
targets = []
preds = []
aucs = []

for fold in range(5):
    train_idx = folds['fold'] != fold
    valid_idx = folds['fold'] == fold
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[valid_idx]
    y_val = y[valid_idx]

    targets.append(y_val)

    model = get_model(pos_scale, seed)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_val)

    pred = np.array(pred)[:,1]
    preds.append(pred)
    auc = roc_auc_score(y_val, pred)
    aucs.append(auc)
    print(auc)
    del model

### evaluate
targets = np.concatenate(targets)
preds = np.concatenate(preds)

print("(!) cv5 AUC ", np.mean(aucs), np.std(aucs))
evaluate(preds, targets)

## Training abnormal
print("Training abnormal")
y = df['label_abnormal']
pos_scale = (y == 0).sum() / (y == 1).sum()
print(pos_scale)

### load fold split
if not os.path.exists(OUTPUT_DIR + 'df_label_abnormal_5fold.csv'):
    folds = df.copy()
    Fold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label_abnormal'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    folds.to_csv(OUTPUT_DIR + 'df_label_abnormal_5fold.csv', index=False)
else:
    folds = pd.read_csv(OUTPUT_DIR + 'df_label_abnormal_5fold.csv')

### training
X = merge_feature([X_trill_features])
targets = []
preds = []
aucs = []

for fold in range(5):
    train_idx = folds['fold'] != fold
    valid_idx = folds['fold'] == fold
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[valid_idx]
    y_val = y[valid_idx]

    targets.append(y_val)

    model = get_model2(pos_scale, seed)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_val)

    pred = np.array(pred)[:,1]
    preds.append(pred)
    auc = roc_auc_score(y_val, pred)
    aucs.append(auc)
    print(auc)
    del model

### evaluate
targets = np.concatenate(targets)
preds = np.concatenate(preds)

print("(!) cv5 AUC ", np.mean(aucs), np.std(aucs))
evaluate(preds, targets)

## Training symptom
print("Training symptom")

y = df['label_symptom']
pos_scale = (y == 0).sum() / (y == 1).sum()
print(pos_scale)

### load fold split
if not os.path.exists(OUTPUT_DIR + 'df_label_symptom_5fold.csv'):
    folds = df.copy()
    Fold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label_symptom'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    folds.to_csv(OUTPUT_DIR + 'df_label_symptom_5fold.csv', index=False)
else:
    folds = pd.read_csv(OUTPUT_DIR + 'df_label_symptom_5fold.csv')

### training
X = merge_feature([X_trill_features])

targets = []
preds = []
aucs = []

for fold in range(5):
    train_idx = folds['fold'] != fold
    valid_idx = folds['fold'] == fold
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[valid_idx]
    y_val = y[valid_idx]

    targets.append(y_val)

    model = get_model2(pos_scale, seed)
    model.fit(X_train, y_train)

    pred = model.predict_proba(X_val)

    pred = np.array(pred)[:,1]
    preds.append(pred)
    auc = roc_auc_score(y_val, pred)
    aucs.append(auc)
    print(auc)
    del model

### evaluate
targets = np.concatenate(targets)
preds = np.concatenate(preds)

print("(!) cv5 AUC ", np.mean(aucs), np.std(aucs))
evaluate(preds, targets)
