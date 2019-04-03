# Hybrid Fake News Prediction Model

A machine learning based hybrid model to classify Fake and Real news articles.

Th aim is to predict authenticity of a new article using its title and text. The predictive features are decided using several techniques such as vectorization of input data, feature engineering, sentiment analysis, latent dirichlet allocation, and stance features extraction.

The prediction models used are Logistic Regression, Gradient Boosting, and Multilayer Perceptron Neural Network.

The models are tuned further and are combined using a pipeline to form a hybrid classifcation model.

### File Descriptions:

- news_articles.csv: input data with title, text, and label of news articles. <br>
- process.py: a python script to load and extract features of the input data. *Specify an input data file as a runtime argument.* <br>
- processed_data.csv: a csv file contained processed data and its features. <br>
- model.py: a hybrid model implementation for training and testing the processed data. <br>

### Results on testing data:

Models with tuned parameters:

- Logistic regression <br>
Accuracy = 92.76%
AUC = 97.23%
- Gradient Boosting <br>
Accuracy = 91.74%
AUC = 97.28%
- Neural Network <br>
Accuracy = 93.08%
AUC = 97.60%
- The Hybrid Model <br>
Accuracy = 93.02%
AUC = 98.04%
