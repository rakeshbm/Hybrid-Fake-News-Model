# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union #to use FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def get_manual(df):
    return df.iloc[:, 5:]

def get_text(df):
    return df.text

def get_dense(df):
    return df.toarray()

def logistic_regression(union, train, test, y):
    
    logreg = LogisticRegression()
    pipe_logreg = make_pipeline(union, logreg)
    print(pipe_logreg.steps)
    print(cross_val_score(pipe_logreg, train, y, cv=5, scoring='accuracy').mean())
    
    # create a grid of parameters to search (and specify the pipeline step along with the parameter)
    param_grid_logreg = {}
    param_grid_logreg['logisticregression__C'] = [0.1, 1, 10]
    
    grid_logreg = GridSearchCV(pipe_logreg, param_grid_logreg, cv=5, scoring='accuracy') #pipe is a nested pipe
    grid_logreg.fit(train, y)
    print(grid_logreg.best_score_)
    print(grid_logreg.best_params_)
    
    # print the best logistic regression model found by GridSearchCV
    print(grid_logreg.best_estimator_) #output the pipeline with the best estimator
    
    # GridSearchCV automatically refit the best model with the entire dataset, and can be used to make predictions
    pred_class_logreg = grid_logreg.predict(test)
    print(pred_class_logreg)
    
    # calculate predicted probabilities of class membership for the new data
    pred_prob_logreg = grid_logreg.predict_proba(test)
    print(pred_prob_logreg)
    
    print(metrics.accuracy_score(test.label_num, pred_class_logreg))
    print(metrics.roc_auc_score(test.label_num, pred_prob_logreg[:,1]))
    
    return pred_class_logreg
    
def gradient_boosting(union, train, test, y):
    
    get_dense_ft = FunctionTransformer(get_dense, validate=False) #convert the method into a ft, function transformer
    gb = GradientBoostingClassifier(verbose=True)
    pipe_gb = make_pipeline(union, get_dense_ft, gb)
    
    print(cross_val_score(pipe_gb, train, y, cv=5, scoring='accuracy').mean())
    
    print(pipe_gb.steps)
    
    # create a grid of parameters to search (and specify the pipeline step along with the parameter)
    param_grid_gb = {}
    param_grid_gb['gradientboostingclassifier__n_estimators'] = [50,100]
    param_grid_gb
    
    grid_gb = GridSearchCV(pipe_gb, param_grid_gb, cv=5, scoring='accuracy') #pipe is a nested pipe
    print(grid_gb.fit(train, y))
    print(grid_gb.best_score_)
    print(grid_gb.best_params_)
    
    # print the best gradient boosting classifier model found by GridSearchCV
    print(grid_gb.best_estimator_) #output the pipeline with the best estimator
    # GridSearchCV automatically refit the best model with the entire dataset, and can be used to make predictions
    pred_class_gb = grid_gb.predict(test)
    print(pred_class_gb)
    
    # calculate predicted probabilities of class membership for the new data
    pred_prob_gb = grid_gb.predict_proba(test)
    print(pred_prob_gb)
    
    print(metrics.accuracy_score(test.label_num, pred_class_gb))
    print(metrics.roc_auc_score(test.label_num, pred_prob_gb[:,1]))

    return pred_class_gb

def mlp_classifier(union, train, test, y):

    nn = MLPClassifier()
    pipe_nn = make_pipeline(union, nn)
    
    print(cross_val_score(pipe_nn, train, y, cv=5, scoring='accuracy').mean())
    
    # create a grid of parameters to search (and specify the pipeline step along with the parameter)
    param_grid_nn = {}
    param_grid_nn['mlpclassifier__hidden_layer_sizes'] = [(100,),(200,),(300,)]
    
    grid_nn = GridSearchCV(pipe_nn, param_grid_nn, cv=5, scoring='accuracy') #pipe is a nested pipe
    print(grid_nn.fit(train, y))
    print(grid_nn.best_score_)
    print(grid_nn.best_params_)
    
    # print the neural network model found by GridSearchCV
    print(grid_nn.best_estimator_) #output the pipeline with the best estimator
    # GridSearchCV automatically refit the best model with the entire dataset, and can be used to make predictions
    pred_class_nn = grid_nn.predict(test)
    print(pred_class_nn)

    # calculate predicted probabilities of class membership for the new data
    pred_prob_nn = grid_nn.predict_proba(test)
    print(pred_prob_nn)
    print(metrics.accuracy_score(test.label_num, pred_class_nn))
    print(metrics.roc_auc_score(test.label_num, pred_prob_nn[:,1]))
    
    return pred_class_nn

def hybrid_model(logreg, gb, nn):
    
    # calculate the mean of the predicted probabilities for all rows
    pred_prob = pd.DataFrame((logreg + gb + nn) / 3)
    # for each row, find the column with the highest predicted probability
    pred_class = pred_prob.apply(np.argmax, axis=1)
    print(metrics.accuracy_score(test.label_num, pred_class))
    print(metrics.roc_auc_score(test.label_num, pred_prob.loc[:,1]))
    #store final predictions in a csv file
    pred_class.to_csv("predictions.csv", encoding="utf-8")
    
def model():
    
    #read processed data
    news = pd.read_csv('processed_data.csv', index_col=0)
    #split data for training and testing
    train, test = train_test_split(news, random_state=888)
    y = train.label_num
    # create a stateless transformer from the get_manual function
    get_manual_ft = FunctionTransformer(get_manual, validate=False) # a wrapper
    get_manual_ft.transform(train)
    # create and test another transformer
    get_text_ft = FunctionTransformer(get_text, validate=False) #convert the method into a ft, function transformer
    vect2 = CountVectorizer() #instantiate another default count vectorizer to handles the news text directly
    union = make_union(make_pipeline(get_text_ft, vect2), get_manual_ft) #first transformer make_pipeline(get_text_ft, vect) converts the text into dtm via vect2
    #run logistic regression
    pred_class_logreg = logistic_regression(union, train, test, y)
    #run gradietn boosting
    pred_class_gb = gradient_boosting(union, train, test, y)
    #run MLP classifier
    pred_class_nn = mlp_classifier(union, train, test, y)    
    #run hybrid model
    hybrid_model(pred_class_logreg, pred_class_gb, pred_class_nn)
    
model()