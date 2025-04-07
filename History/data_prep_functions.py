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

print("test! updated 'scorer' function")


def feat_elim_rand(df, labels, out_name, feats, estimator, step):
    estimator = RandomForestRegressor(n_estimators=estimator)
    selector = RFE(estimator, n_features_to_select=feats, step=step)
    selector = selector.fit(df, labels)
    selector.support_
    feat_list = selector.get_feature_names_out()
    a = pd.DataFrame(list(zip(df.columns, selector.ranking_)), columns=['feature', 'selector rankings'])
    a.to_excel('Output_data/FeatSelection' + str(out_name) + '.xlsx')
    return df[feat_list]


def rand_forest_reg_fit(df, labels, out_name, test_size, estimator):
    x_train, x_test, y_train, y_test = train_test_split(df, labels,
                                                        test_size=test_size,
                                                        random_state=42)
    rfg = RandomForestRegressor(n_estimators=estimator)
    rfg.fit(x_train, y_train)
    feat_importances = rfg.feature_importances_
    b = list(zip(df.columns, feat_importances * 100))
    score = rfg.score(x_test, y_test)
    # a = pd.DataFrame(b, columns=['feature', 'feat_importance'])
    # a.to_excel("Output_data/Featimportances"+str(out_name)+".xlsx")
    return b, score


def scram_score(df, label, model, identifier, test_percent):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    r2s.append(r2)
    pearson.append(corr)
    mse.append(mse_score)
    feat_import = model.feature_importances_
    feats.append('allfeats')
    feat_import = np.insert(feat_import, [0], 0)
    for j in x_test.columns:
        tmp = x_test.copy()
        # print(x_test)
        np.random.shuffle(tmp[j].values)
        scram_score = model.score(tmp, y_test)
        predictions = model.predict(tmp)
        r2 = r2_score(y_test, predictions)
        corr, _ = pearsonr(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        r2s.append(r2)
        pearson.append(corr)
        feats.append(j)
        mse.append(mse_score)
    # a = pd.DataFrame(list(zip(feats, pearson, r2s, feat_import)), columns=['feat', 'pearson', 'R2', 'importances'])
    # a.to_excel("Output_data/scram_loss_feats" + id + ".xlsx")
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(feats, pearson, color='tab:blue', marker='o', label='Pearson')
    ax.plot(feats, r2s, color='tab:red', marker='s', label='R2')
    ax.plot(feats, mse, color='tab:green', marker='^', label='MSE')
    ax.plot(feats, feat_import, color='tab:purple', marker='x', label='Feature Importance')

    ax.set_xticklabels(feats, rotation=90)
    ax.legend()

    ax.set_title('Score as a Function of Scrambled Feature\n{}'.format(id))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Score')
    plt.savefig('Output_data/FeatScramLoss' + id + '.png', bbox_inches='tight')
    plt.close('all')
    print('Scramble Scoring ran successfully')


def feat_drop(df, label, model, identifier, test_percent):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    corr, _ = pearsonr(y_test, predictions)
    mse_score = mean_squared_error(y_test, predictions)
    a = list(zip(model.feature_importances_, model.feature_names_in_))
    a.sort(reverse=True)
    feat_import = model.feature_importances_
    col_import = pd.DataFrame(a, columns=['importances', 'names'])
    sorted_cols = col_import['names']
    feat_import = np.insert(feat_import, [0], 0)
    feat_import = np.delete(feat_import, [-1])
    feats.append('All Feats')
    r2s.append(r2)
    pearson.append(corr)
    mse.append(mse_score)

    df_3 = df.copy()
    for i in sorted_cols:
        if i == sorted_cols.iloc[-1]:
            break
        # df_3=df_2.copy() #remove if you only want to drop each feature instead of dropping one feature at a time
        df_3.drop(columns=[i], inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(df_3, label, test_size=test_percent)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        r2 = r2_score(y_test, predictions)
        corr, _ = pearsonr(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        r2s.append(r2)
        pearson.append(corr)
        mse.append(mse_score)
        feats.append(i)

    # df_out = pd.DataFrame(list(zip(feats, pearson, r2s, mse, accuracy)), columns=['dropped feat', 'Pearson', 'r2', 'neg_mse', 'accuracy'])
    # df_out.to_excel("Output_data/feat_drop_cumulative" + id + ".xlsx")
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(feats, pearson, color='tab:blue', marker='o', label='Pearson')
    ax.plot(feats, r2s, color='tab:red', marker='s', label='R2')
    ax.plot(feats, mse, color='tab:green', marker='^', label='MSE')
    ax.plot(feats, feat_import, color='tab:purple', marker='x', label='Feature Importance')

    ax.set_xticklabels(feats, rotation=90)
    ax.legend()

    ax.set_title('Score as a Function of Feature Drop\n{}'.format(id))
    ax.set_xlabel('Features Dropped')
    ax.set_ylabel('Score')

    plt.savefig('Output_data/feat_drop_cumulative_{}.png'.format(id), bbox_inches='tight')
    plt.close(fig)
    print('Feat drop ran successfully')


def feat_drop_multifold(df, label, model, identifier, test_percent, folds):
    id = identifier
    feats = []
    r2s = []
    pearson = []
    mse = []
    feat_importances = []

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=test_percent, random_state=42)
    model.fit(x_train, y_train)
    feat_import = model.feature_importances_

    a = list(zip(feat_import, df.columns))
    a.sort(reverse=True)
    col_import = pd.DataFrame(a, columns=['importances', 'names'])
    sorted_cols = col_import['names']
    feats.append('All Feats')
    feats.extend(sorted_cols.tolist())
    feats = feats[:-1]

    for train_index, test_index in kf.split(df):
        x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        r2 = r2_score(y_test, predictions)
        corr, _ = pearsonr(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        feat_import = model.feature_importances_

        a = list(zip(feat_import, df.columns))
        a.sort(reverse=True)

        feat_import = np.insert(feat_import, [0], 0)
        feat_import = np.delete(feat_import, [-1])

        r2s.append(r2)
        pearson.append(corr)
        mse.append(mse_score)
        feat_importances.append(feat_import)

        df_3 = df.copy()
        for i in sorted_cols:
            if i == sorted_cols.iloc[-1]:
                break
            df_3.drop(columns=[i], inplace=True)
            x_train, x_test = df_3.iloc[train_index], df_3.iloc[test_index]
            y_train, y_test = label[train_index], label[test_index]

            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            r2 = r2_score(y_test, predictions)
            corr, _ = pearsonr(y_test, predictions)
            mse_score = mean_squared_error(y_test, predictions)
            r2s.append(r2)
            pearson.append(corr)
            mse.append(mse_score)

    feats = np.array(feats)
    r2s = np.array(r2s).reshape(folds, -1)
    pearson = np.array(pearson).reshape(folds, -1)
    mse = np.array(mse).reshape(folds, -1)
    feat_importances = np.array(feat_importances).reshape(folds, -1)

    import matplotlib.pyplot as plt

    mean_r2s = np.mean(r2s, axis=0)
    std_r2s = np.std(r2s, axis=0)
    mean_pearson = np.mean(pearson, axis=0)
    std_pearson = np.std(pearson, axis=0)
    mean_mse = np.mean(mse, axis=0)
    std_mse = np.std(mse, axis=0)
    mean_feat_importances = np.mean(feat_importances, axis=0)
    std_feat_importances = np.std(feat_importances, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot R2 scores
    ax.plot(feats, mean_r2s, color='tab:red', label='R2')
    ax.fill_between(feats, mean_r2s - std_r2s, mean_r2s + std_r2s, alpha=0.2, color='tab:red')

    # Plot Pearson correlation coefficients
    ax.plot(feats, mean_pearson, color='tab:blue', label='Pearson')
    ax.fill_between(feats, mean_pearson - std_pearson, mean_pearson + std_pearson, alpha=0.2, color='tab:blue')

    # Plot mean squared errors
    ax.plot(feats, mean_mse, color='tab:green', label='MSE')
    ax.fill_between(feats, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color='tab:green')

    # Plot feature importances
    ax.plot(feats, mean_feat_importances, color='tab:purple', label='Feature Importance')
    ax.fill_between(feats, mean_feat_importances - std_feat_importances, mean_feat_importances + std_feat_importances,
                    alpha=0.2, color='tab:purple')

    ax.set_xticklabels(feats, rotation=90)
    ax.legend()

    ax.set_title('Score as a Function of Feature Drop\n{}'.format(id))
    ax.set_xlabel('Features Dropped')
    ax.set_ylabel('Score')

    plt.savefig('Output_data/feat_drop_multifold_{}.png'.format(id), bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('Feat drop multifold ran successfully')


def scorer(df, label, model, identifier, folds):
    y = label
    X = df

    # Initialize lists to store Pearson, R2, MSE, and Spearman scores for each fold
    pearson_scores = []
    r2_scores = []
    mse = []
    spear = []

    # Split your data into folds using KFold
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Loop through each fold and train/test the model
    for train_index, test_index in kfold.split(X, y):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model on the training set
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

    # Plotting the results
    plt.style.use('classic')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')

    # Plot individual scores for each fold
    fold_idx = range(len(pearson_scores))
    line1, = axs[0].plot(fold_idx, pearson_scores, 'o', color='black', markersize=4, label='Pearson')
    line2, = axs[0].plot(fold_idx, r2_scores, 's', color='grey', markersize=4, label='R2')
    line3, = axs[0].plot(fold_idx, spear, '^', color='lightgrey', markersize=4, label='Spearman')

    # Manually create the legend to ensure no duplicates
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

    axs[1].set_title('RFR Model: RFECV Avg. and STDEV\nfor Bovine Swiss-Prot iBAQ', fontweight='bold', color='black')
    axs[1].set_ylabel('Score', fontweight='bold', color='black')
    
    # Extend y-axis range to 1.0
    axs[1].set_ylim([0, 1.0])

    # Extend x-axis range to add gaps
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








# def scorer(df, label, model, identifier, folds):
#     """

#     :type identifier: string
#     """
#     y = label
#     X = df

#     # Initialize lists to store Pearson and R2 scores for each fold
#     pearson_scores = []
#     r2_scores = []
#     mse = []
#     spear=[]
#     # Split your data into 10 folds using KFold
#     kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

#     # Create a 2x1 subplot figure for the scores and the average/standard deviation

#     # Loop scough each fold and train/test a linear regression model
#     for train_index, test_index in kfold.split(X, y):
#         # Split the data into training and test sets
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Fit a linear regression model on the training set
#         model.fit(X_train, y_train)

#         # Predict on the test set and evaluate the model
#         y_pred = model.predict(X_test)
#         pearson, _ = pearsonr(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         mse_score = mean_squared_error(y_test, y_pred)
#         spear_score= spearmanr(y_pred,y_test)

#         # Append the scores for this fold to the lists
#         pearson_scores.append(pearson)
#         r2_scores.append(r2)
#         mse.append(mse_score)
#         spear.append(spear_score)

#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     # plot individual and aggregate scores for each scoring method
#     fold_idx = range(len(pearson_scores))
#     axs[0].plot(fold_idx, pearson_scores, 'bo', label='Pearson')
#     axs[0].plot(fold_idx, r2_scores, 'go', label='R2')
#     axs[0].plot(fold_idx, mse, 'kx', label='MSE')
#     axs[0].legend()
#     axs[0].set_title('Individual Scores\n{}'.format(identifier))
#     axs[0].set_xlabel('Fold Index')
#     axs[0].set_ylabel('Score')

#     pearson_mean = np.mean(pearson_scores)
#     R2_mean =np.mean(r2_scores)
#     MSE_mean = np.mean(mse)
#     spear_mean = np.mean(spear)
#     pearson_std = np.std(pearson_scores)
#     R2_std = np.std(r2_scores)
#     MSE_std = np.std(mse)
#     spear_std = np.std(spear)

#     feat_import = model.feature_importances_
#     # plot the average and standard deviation of the scores on a separate subplot
#     # axs[1].bar(['Pearson', 'R2', 'MSE'], [pearson_mean, R2_mean, MSE_mean],
#     #            yerr=[pearson_std, R2_std, MSE_std])
#     # axs[1].set_title('Average and Standard Deviation of Scores')
#     # axs[1].set_ylabel('Score')

#     # adjust spacing between subplots and save the figure
#     # fig.subplots_adjust(wspace=0.3)
#     # fig.set_dpi(300)
#     # plt.tight_layout()
#     # plt.savefig('Output_data/scores_{}.png'.format(identifier), bbox_inches='tight')
#     # plt.close(fig)
#     data=[[pearson_mean,pearson_std,R2_mean,R2_std,MSE_mean,MSE_std,spear_mean,spear_std,df.shape[1],identifier]]
#     feat_scores=list(zip(df.columns.tolist(),feat_import,))
#     scores=pd.DataFrame(data,columns=['pearson_mean','pearson_std','R2_mean','R2_std','MSE_mean','MSE_std','spearman_mean','spearman_std','Number of Features','ID'])
#     feats=pd.DataFrame(feat_scores,columns=['Features','Importance'+identifier])
#     print('Scorer ran successfully')
#     return scores, feats

def scorer_RFC(df, label, model, identifier, folds, output_dir):
    y = label
    X = df

    # Initialize lists to store Pearson and R2 scores for each fold
    F1_scores = []
    auroc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    # Split your data into 10 folds using KFold
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
    plt.close(fig)
    data=[[F1_mean,F1_std,auroc_mean,acc_std,acc_mean,acc_std,precision_mean,precision_std,recall_mean,recall_std,df.shape[1],identifier]]
    feat_scores=list(zip(df.columns.tolist(),feat_import,))
    scores=pd.DataFrame(data,columns=['F1_mean','F1_std','AUROC_mean','AUROC_std','Accuracy_mean','Accuracy_std','precision_mean','precision_std','recall_mean','recall_std','Number of Features','ID'])
    feats=pd.DataFrame(feat_scores,columns=['Features','Importance'+identifier])
    print('Scorer ran successfully')
    return scores, feats


def PCA_plot(df, label, identifier):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    id = identifier
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df)
    pca = PCA(n_components=5)
    x_pca = pca.fit_transform(X_std)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=label, cmap='viridis')
    plt.colorbar()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('Output_data/PCA' + id + '.png')
    plt.close('all')
    print('PCA Ran successfully')


def RFECV_plot(df, label, model, identifier, folds, step, scoring='neg_mean_squared_error'):
    from sklearn.feature_selection import RFECV

    min_feats = 8
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    estimator = model
    selector = RFECV(estimator=estimator, cv=cv, scoring=scoring, min_features_to_select=min_feats,
                     step=step)
    selector = selector.fit(df, label)
    selector.support_
    feat_list2 = selector.get_feature_names_out()
    selected_features = df.columns[selector.support_]
    df = df[feat_list2]
    # df.to_excel("Input_data/Save_files/df_RFECV"+id+id2+".xlsx")
    # rfecv_df=pd.DataFrame(selector.cv_results_)
    # rfecv_df.to_excel("Output_data/RFECV_results"+id+id2+".xlsx")
    # label_abund_df.to_excel("Input_data/Save_files/label_abund_all.xlsx")
    n_scores = len(selector.cv_results_["mean_test_score"])
    fig, ax = plt.subplots(figsize=(8, 6))

    x = range(1, n_scores + 1)
    y = selector.cv_results_["mean_test_score"]
    err = selector.cv_results_["std_test_score"]

    ax.plot(x, y, 'k-', label=scoring)
    ax.fill_between(x, y - err, y + err, alpha=0.2, label='Standard Deviation')
    ax.legend()
    ax.set_xlabel('Number of Features Selected')
    ax.set_ylabel(scoring)
    ax.set_title('Recursive Feature Elimination with Correlated Features\n{}'.format(identifier))

    fig.set_dpi(300)
    plt.tight_layout()
    plt.savefig('Output_data/RFECV_{}.png'.format(identifier), bbox_inches='tight')
    plt.close(fig)

    print('Recursive Feature Elimination with Correlated Features ran successfully')
    return df


def RFECV_plot_yb(df, label, model, identifier, folds, step, scoring='neg_mean_squared_error'):
    min_feats = 8
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    estimator = model
    selector = RFECV(estimator=estimator, cv=cv, scoring=scoring, min_features_to_select=min_feats,
                     step=step)
    selector.fit(df, label)
    feat_list2=selector.get_feature_names_out()
    df2=df[feat_list2].copy()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot RFECV visualizer
    visualizer = RFECV(estimator, cv=cv, scoring=scoring, step=step)
    visualizer.fit(df, label)
    visualizer.show(ax=ax[0])

    # Plot feature importances
    if hasattr(estimator, 'coef_'):
        importances = estimator.coef_
    elif hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    else:
        raise AttributeError('Estimator does not have a coefficient or feature_importances_ attribute.')

    std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    names = df.columns

    ax[1].set_title("Feature importances")
    ax[1].bar(range(df.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    ax[1].set_xticks(range(df.shape[1]))
    ax[1].set_xticklabels(names[indices], rotation=90)
    ax[1].set_xlim([-1, df.shape[1]])

    fig.suptitle("Recursive Feature Elimination with Cross-Validation\n{}".format(identifier))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.savefig('Output_data/RFECV_{}.png'.format(identifier), bbox_inches='tight')
    plt.show()

    return df2



def lasso_feature_selection(df, label, identifier):
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    X = df
    y = label

    # Scale the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform Lasso regression
    lasso = Lasso(alpha=0.1, random_state=42)
    pipe = Pipeline([('scaler', scaler), ('lasso', lasso)])
    pipe.fit(X_scaled, y)

    # Get feature importances
    importances = np.abs(pipe.named_steps['lasso'].coef_)
    feature_names = X.columns

    # Sort features by importance
    sorted_idx = importances.argsort()[::-1]
    importances = importances[sorted_idx]
    feature_names = feature_names[sorted_idx]
    id = identifier
    # Create plot of feature importances
    plt.figure()
    plt.title("Feature importances using Lasso Regression for {}".format(id))
    plt.bar(range(X.shape[1]), importances)
    plt.xticks(range(X.shape[1]), feature_names, rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('Output_data/LassoReg_{}.png'.format(id), bbox_inches='tight')
    plt.close('all')

    # Reduce dataset to high-importance features
    X_reduced = X.iloc[:, sorted_idx[:10]]

    return X_reduced


if __name__ == "__main__":
    print(type(ProteinAnalysis))
    print('found biopython')
