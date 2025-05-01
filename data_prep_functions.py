import re

import matplotlib.pyplot as plt

# from interpro_scraping import interpro_scraping_pandas
#from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import KFold
from yellowbrick.datasets import load_credit
from yellowbrick.model_selection import RFECV as RFECVyb
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
plt.style.use('ggplot')

print("updated 'scorer' function")


def scorer(df, label, model, identifier, folds):
    y = label
    X = df

    # Initialize lists to store Pearson, R2, MSE, and Spearman scores for each fold
    pearson_scores = []
    r2_scores = []
    mse = []
    spear = []
    # Split data into 10 folds
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Loop through each fold and train/test the model
    for train_index, test_index in kfold.split(X, y):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model on the train set
        model.fit(X_train, y_train)

        # Predict on the test set and evaluate the model
        y_pred = model.predict(X_test)
        pearson, _ = pearsonr(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_score = mean_squared_error(y_test, y_pred)
        spear_score, _ = spearmanr(y_test, y_pred)

        # Append the scores for this fold to the lists
        pearson_scores.append(pearson)
        r2_scores.append(r2)
        mse.append(mse_score)
        spear.append(spear_score)

    # Calculate mean and standard deviation for each metric
    pearson_mean = np.mean(pearson_scores)
    r2_mean = np.mean(r2_scores)
    mse_mean = np.mean(mse)
    spear_mean = np.mean(spear)

    pearson_std = np.std(pearson_scores)
    r2_std = np.std(r2_scores)
    mse_std = np.std(mse)
    spear_std = np.std(spear)

    feat_import = model.feature_importances_

    # Prepare data for DataFrame
    data = [[pearson_mean, pearson_std, r2_mean, r2_std, mse_mean, mse_std, spear_mean, spear_std, df.shape[1], identifier]]
    feat_scores = list(zip(df.columns.tolist(), feat_import))

    scores = pd.DataFrame(data, columns=['pearson_mean', 'pearson_std', 'r2_mean', 'r2_std', 'mse_mean', 'mse_std', 'spearman_mean', 'spearman_std', 'Number of Features', 'ID'])
    feats = pd.DataFrame(feat_scores, columns=['Features', 'Importance_' + identifier])

    plt.style.use('classic')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')

    # Plot individual scores for each fold
    fold_idx = range(len(pearson_scores))
    line1, = axs[0].plot(fold_idx, pearson_scores, 'o', color='black', markersize=4, label='Pearson')
    line2, = axs[0].plot(fold_idx, r2_scores, 's', color='grey', markersize=4, label='R2')
    line3, = axs[0].plot(fold_idx, spear, '^', color='lightgrey', markersize=4, label='Spearman')

    axs[0].legend(handles=[line1, line2, line3], loc='upper right')

    axs[0].set_title('Individual Scores\n{}'.format(identifier), fontweight='bold', color='black')
    axs[0].set_xlabel('Fold Index', fontweight='bold', color='black')
    axs[0].set_ylabel('Score', fontweight='bold', color='black')

    axs[0].tick_params(axis='both', which='both', direction='out',
                       length=6, width=2, colors='black',
                       labelsize='large', labelcolor='black',
                       bottom=True, top=False, left=True, right=False)
    axs[0].grid(False)

    # Plot average and standard deviation of the scores (excluding MSE as it is covered in the detailed plot)
    metrics = ['Pearson', 'R2', 'Spearman']
    means = [pearson_mean, r2_mean, spear_mean]
    stds = [pearson_std, r2_std, spear_std]
    colors = ['black', 'grey', 'lightgrey']
    bars = axs[1].bar(metrics, means, yerr=stds, capsize=10, color=colors, width=0.4, edgecolor='black', linewidth=1.5, error_kw=dict(elinewidth=2, capthick=2))
    
    # Change plot title here when needed
    axs[1].set_title('RFR Model: RFECV Avg. and STDEV\nfor Bovine Swiss-Prot Intensity', fontweight='bold', color='black')
    axs[1].set_ylabel('Score', fontweight='bold', color='black')
    
    # Extend y-axis range to 1.0
    axs[1].set_ylim([0, 1.0])

    # Extend x-axis range 
    axs[1].set_xlim([-0.5, len(metrics) - 0.5])

    axs[1].tick_params(axis='both', which='both', direction='out',
                       length=6, width=2, colors='black',
                       labelsize='large', labelcolor='black',
                       bottom=True, top=False, left=True, right=False)
    axs[1].grid(False)

    plt.tight_layout()
    plt.show()

    print('Scorer ran successfully')
    return scores, feats


def scorer_RFC(df, label, model, identifier, folds, output_dir):
    y = label
    X = df

    # Initialize lists to store Pearson and R2 scores for each fold
    F1_scores = []
    auroc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    # Split data into 10 folds
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Create a 2x1 subplot figure for the scores and the average/standard deviation

    # Loop through each fold and train/test a linear regression model
    for train_index, test_index in kfold.split(X, y):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit a linear regression model on the training set
        model.fit(X_train, y_train)

        # Predict on the test set and evaluate the model
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Append the scores for this fold to the lists
        F1_scores.append(f1)
        auroc_scores.append(auroc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # plot individual and aggregate scores for each scoring method
    fold_idx = range(len(F1_scores))
    axs[0].plot(fold_idx, F1_scores, 'bo', label='F1')
    axs[0].plot(fold_idx, auroc_scores, 'go', label='AUROC')
    axs[0].plot(fold_idx, accuracy_scores, 'kx', label='Accuracy')
    axs[0].legend()
    axs[0].set_title('Individual Scores\n{}'.format(identifier))
    axs[0].set_xlabel('Fold Index')
    axs[0].set_ylabel('Score')

    F1_mean = np.mean(F1_scores)
    auroc_mean =np.mean(auroc_scores)
    acc_mean = np.mean(accuracy_scores)
    precision_mean = np.mean(precision_scores)
    recall_mean = np.mean(recall_scores)

    F1_std = np.std(F1_scores)
    auroc_std =np.std(auroc_scores)
    acc_std = np.std(accuracy_scores)
    precision_std = np.std(precision_scores)
    recall_std = np.std(recall_scores)

    feat_import = model.feature_importances_
    # plot the average and standard deviation of the scores on a separate subplot
    axs[1].bar(['F1', 'AUROC', 'Accuracy','Precision','Recall'],
               [F1_mean, auroc_mean, acc_mean,precision_mean,recall_mean],
               yerr=[F1_std, auroc_std, acc_std,precision_std,recall_std])
    axs[1].set_title('Average and Standard Deviation of Scores')
    axs[1].set_ylabel('Score')

    # adjust spacing between subplots and save the figure
    fig.subplots_adjust(wspace=0.3)
    fig.set_dpi(300)
    plt.tight_layout()
    #plt.savefig(f'{output_dir}/scores_{identifier}.png', bbox_inches='tight')
    plt.show()
    #plt.close(fig)
    data=[[F1_mean,F1_std,auroc_mean,acc_std,acc_mean,acc_std,precision_mean,precision_std,recall_mean,recall_std,df.shape[1],identifier]]
    feat_scores=list(zip(df.columns.tolist(),feat_import,))
    scores=pd.DataFrame(data,columns=['F1_mean','F1_std','AUROC_mean','AUROC_std','Accuracy_mean','Accuracy_std','precision_mean','precision_std','recall_mean','recall_std','Number of Features','ID'])
    feats=pd.DataFrame(feat_scores,columns=['Features','Importance'+identifier])
    print('Scorer ran successfully')
    return scores, feats


