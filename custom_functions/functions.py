#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas as pd

def subplots(dataframe, variables, kind, x, y, n_bins, figsize = (15,12)):
    '''If kind != hist, set n_bins = None.
        Use this function when you want to create many histograms
        in the same figure. This function allows to built visualizations
        with different x ranges.'''
    
    plt.figure(figsize = figsize)
    colors = list(mcolor.TABLEAU_COLORS.values())
    j = 0
    for i, name in enumerate(variables):
        plt.subplot(x, y, i+1)
        if j == len(colors):
            j = 0
        if kind == 'hist':
            dataframe[name].plot(kind = kind, bins = n_bins, color = colors[j])
        else:
            dataframe[name].plot(kind = kind, color = colors[j])
        plt.title(name)
        j += 1
    plt.tight_layout()
    plt.show()


# In[ ]:


def bar_plots_hue_categorical(dataframe, variables, hue_var, x = 4, y = 4, size_fig = (15,16)):
    
    '''Use this function to plot barplots for multiple categorical variables, which values is numeric, 
    explicitly showing the classes they belong to.'''
    
    plt.figure(figsize= size_fig)
    for i, var in enumerate(variables):
    
        # creating a dataframe
        data = dataframe.groupby(hue_var)[var].value_counts().to_frame(name = 'count').reset_index()
    
        # changing the format of var values
        data[var] = data[var].apply(lambda x: format(x, '.3g'))
    
        # plot
        plt.subplot(x, y, i+1)
        sns.barplot(x = var, y = 'count', hue = hue_var, data = data)
        plt.xlabel('')
        plt.title(var)
    
    plt.tight_layout()
    plt.show()


# In[ ]:


def bar_plot_hue_percentile(dataframe, variables, percentile, hue_var, x = 6, y = 3, size_fig = (15,20)):
    plt.figure(figsize= size_fig)
    for i, var in enumerate(variables):
    
        # plot
        plt.subplot(x, y, i+1)
        dataframe[dataframe[var].rank(pct=True) <= percentile].groupby(hue_var)[var].hist(bins = 30, grid = False)
        plt.xlabel('')
        plt.title(var)
    
    plt.tight_layout()
    plt.show()


# In[ ]:


def hist_percentile(dataframe, variables, percentile, hue_var, x = 5, y = 2, size_fig = (15,15)):
    plt.figure(figsize= size_fig)
    for i, var in enumerate(variables):
        # plot
        plt.subplot(x, y, i+1)
        dataframe[dataframe[var].rank(pct = True) > percentile].groupby(hue_var)[var].hist(bins = 30, grid = False)
        plt.xlabel('')
        plt.title(var)
    
    plt.tight_layout()
    plt.show()


# In[ ]:


# Metrics of the model

def evaluate_model_without_cv(X_test, y_test, y_pred, modelo):
    # Evaluate a model without cross validation
    
    # Accuracy
    accuracy = modelo.score(X_test, y_test)
    print('Accuracy: {0:0.3f}'.format(accuracy))

    # Precision score
    from sklearn.metrics import precision_score
    print('Precision: {0:0.3f}'.format(precision_score(y_test, y_pred)))
      
    # Recall score
    from sklearn.metrics import recall_score
    print('Recall: {0:0.3f}'.format(recall_score(y_test, y_pred)))

    # F1-score
    from sklearn.metrics import f1_score
    print('F1-score: {0:0.3f}'.format(f1_score(y_test, y_pred)))

    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    print('ROC AUC: {0:0.3f}'.format(roc_auc_score(y_test, y_pred)))
    
    # Confusion matrix
    from sklearn.metrics import plot_confusion_matrix
    disp = plot_confusion_matrix(modelo, X_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 normalize=None)


# In[ ]:


def model_cross_validation(model, X, Y, num_folds, seed):
    
    # Metrics
    metrics = ['accuracy','precision','recall','f1','roc_auc']
    
    # Creating the folds
    kfolds = StratifiedKFold(n_splits= num_folds, shuffle= True, random_state = seed)

    # Model train and test with cross_validate method
    resultado = cross_validate(model, X, Y, scoring = metrics, cv = kfolds)
    
    # Printing the results
    for key, value in resultado.items():
        print('{0} : {1:0.3f}'.format(key, value.mean()))


# In[ ]:


def compare_models(steps, labels, metrics, folds, x_train, y_train):
    
    '''Steps is a list of steps to a pipeline.
       Labels is a list of names to differentiate each model in steps.
       Metrics can be a string with only one metric or a list of metrics'''
    
    score = {}
    for i, step in enumerate(list(zip(labels, steps))):
        pipeline = Pipeline(steps=step[1])
        cv = StratifiedKFold(n_splits=folds, random_state=1)
        
        if type(metrics) is list and len(metrics) > 1:
            scores = cross_validate(pipeline, x_train, y_train, scoring = metrics, cv = cv)
            key = step[0]
            score[key] = scores
        else:
            scores = cross_val_score(pipeline, x_train, y_train, scoring=metrics, cv=cv, n_jobs=-1)
            key = step[0]
            score[key] = scores
        
    if type(metrics) is list and len(metrics) > 1:
        reform = {(outerKey, innerKey): values for outerKey, innerDict in score.items() for innerKey, values in innerDict.items()}
        return pd.DataFrame(reform)
    else:
        return pd.DataFrame(score)


# In[ ]:


def train_roc_plot(models, labels, x_train, y_train, x_test, y_test, feature_names = None):
    
    '''models is a list of steps containing a model that follow a pipeline fashion.
       labels is a list of names to differentiate each model in models.
       feature_names is the names of the features to be considered in x_train and x_test data, if None use all attributes.
       x_train, x_test must be a dataframe with columns names.
       This function uses the Youden's J statistic to estimate the best threshold.'''
    
    best_threshold = []
    plt.subplots(1, figsize=(7,5))
    plt.title('Receiver Operating Characteristic')
    
    if feature_names == None:
        names = [x_train.columns]*len(models)
        steps = list(zip(labels, models, names))
    else:
        steps = list(zip(labels, models, feature_names))
        
    for i, step in enumerate(steps):
        pipeline = Pipeline(steps=step[1])
        model = pipeline.fit(x_train[step[2]], y_train)
        pred = model.predict_proba(x_test[step[2]])
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, pred[:,1])
        J_youdens = true_positive_rate1 - false_positive_rate1
        ix = np.argmax(J_youdens)
        best_threshold.append((step[0], threshold1[ix]))
        auc_c = roc_auc_score(y_test, pred[:,1])
        plt.plot(false_positive_rate1, true_positive_rate1, label = step[0] + ' - AUC: ' + str(np.round(auc_c,3)))
        
        # show the threshold in the plot
        #plt.scatter(false_positive_rate1[ix], true_positive_rate1[ix], marker='o', color='black', 
                   # label='Best threshold' if i == 0 else "")
        
    plt.plot([0, 1], linestyle = "--", label = 'No Skill')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()
    return best_threshold


def roc_plot(predictions, labels, y_test):
    
    '''predictions: a list of predictions
       labels: a list of names to differentiate each model in models.
       y_test: observed values
       This function uses the Youden's J statistic to estimate the best threshold.'''
    
    best_threshold = []
    plt.subplots(1, figsize=(7,5))
    plt.title('Receiver Operating Characteristic')
    
    steps = list(zip(labels, predictions))
        
    for i, step in enumerate(steps):
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, step[1][:,1])
        J_youdens = true_positive_rate - false_positive_rate
        ix = np.argmax(J_youdens)
        best_threshold.append((step[0], threshold[ix]))
        auc_c = roc_auc_score(y_test, step[1][:,1])
        plt.plot(false_positive_rate, true_positive_rate, label = step[0] + ' - AUC: ' + str(np.round(auc_c,3)))
        
        # show the threshold in the plot
        #plt.scatter(false_positive_rate1[ix], true_positive_rate1[ix], marker='o', color='black', 
                   # label='Best threshold' if i == 0 else "")
        
    plt.plot([0, 1], linestyle = "--", label = 'No Skill')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()
    return best_threshold


def train_prc_plot(models, labels, x_train, y_train, x_test, y_test, feature_names = None):
    
    '''Precision-Recall curve.
       models is a list of steps containing a model in a pipeline fashion.
       labels is a list of names to differentiate each model in models.
       feature_names is the names of the features to be considered in x_train and x_test data, if None use all attributes.
       x_train, x_test must be a dataframe with columns names.
       This function uses F1-score to estimate the best threshold.'''
    
    plt.subplots(1, figsize=(7,5))
    plt.title('Precision-Recall Curve')
    best_threshold = []
    
    if feature_names == None:
        names = [x_train.columns]*len(models)
        steps = list(zip(labels, models, names))
    else:
        steps = list(zip(labels, models, feature_names))
    
    for i,step in enumerate(steps):
        pipeline = Pipeline(steps=step[1])
        model = pipeline.fit(x_train[step[2]], y_train)
        pred = model.predict_proba(x_test[step[2]])
        precision, recall, threshold = precision_recall_curve(y_test, pred[:,1])
        fscore = 2*(precision * recall)/(precision + recall)
        ix = np.argmax(fscore)
        best_threshold.append((step[0], threshold[ix]))
        ap = average_precision_score(y_test, pred[:,1])
        plt.step(recall, precision, where = 'post', label = step[0] + ' - AP: ' + str(np.round(ap,3)))
        
        # show thresholds in the plot
        #plt.scatter(recall[ix], precision[ix], marker='o', color='black', 
                   # label='Best threshold' if i==0 else "")

    no_skill = y_train.value_counts(normalize=True)[1]
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend()
    plt.show()
    
    return best_threshold


def prc_plot(predictions, labels, y_test):
    
    '''Precision-Recall curve.
       predictions: a list of predictions.
       labels: a list of names to differentiate each model in models.
       y_test: observed values.
       This function uses the Youden's J statistic to estimate the best threshold.
       This function uses F1-score to estimate the best threshold.'''
    
    plt.subplots(1, figsize=(7,5))
    plt.title('Precision-Recall Curve')
    best_threshold = []
    
    steps = list(zip(labels, predictions))
    
    for i,step in enumerate(steps):
        precision, recall, threshold = precision_recall_curve(y_test, step[1][:,1])
        fscore = 2*(precision * recall)/(precision + recall)
        ix = np.argmax(fscore)
        best_threshold.append((step[0], threshold[ix]))
        ap = average_precision_score(y_test, step[1][:,1])
        plt.step(recall, precision, where = 'post', label = step[0] + ' - AP: ' + str(np.round(ap,3)))
        
        # show thresholds in the plot
        #plt.scatter(recall[ix], precision[ix], marker='o', color='black', 
                   # label='Best threshold' if i==0 else "")

    no_skill = y_test.value_counts(normalize=True)[1]
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend()
    plt.show()
    
    return best_threshold