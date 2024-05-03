import pandas as pd
import ast
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, KFold
import fasttext
import numpy as np
from main import EXCEL_FOLDER, DATA_FOLDER
from MLFunctions.ml_preprocessing_functions import embed_text_with_aspects, create_x_y_matrix_training


fasttext_model = fasttext.load_model(DATA_FOLDER + "fasttext_model.bin")

test_train_data = pd.read_excel(EXCEL_FOLDER + '\\test_train.xlsx')
test_test_data = pd.read_excel(EXCEL_FOLDER + '\\test_tes.xlsx')
test_train_data = test_train_data.dropna(subset=['tf_idf_sum', 'sentiment'])
test_test_data = test_test_data.dropna(subset=['tf_idf_sum', 'sentiment'])

train_data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
train_data = train_data[['tf_idf_sum', 'aspect', 'sentiment']]

embedded_train_data = embed_text_with_aspects(train_data, fasttext_model, 300)

embedded_train_data['sentiment'] = embedded_train_data['sentiment'].apply(lambda x: [int(i) for i in x if i.isdigit()])
embedded_train_data['aspect'] = embedded_train_data['aspect'].apply(lambda x: [int(i) for i in x if i.isdigit()])

X_train, y_binary = create_x_y_matrix_training(embedded_train_data)

svm_classifier = SVC(kernel='linear')
multi_label_svm = MultiOutputClassifier(svm_classifier)
param_grid = {'estimator__C': [1], 'estimator__kernel': ['linear']}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(multi_label_svm, param_grid, cv=kfold, scoring='f1_micro')

print('Grid search started')
grid_search.fit(X_train, y_binary)
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print("Mean F1 score:", mean_score, "Parameters:", params)
# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy score found: ", grid_search.best_score_)


# # Predict
# y_pred_binary = classifier.predict(X_test)
#
# # Evaluate
# accuracy = accuracy_score(y_test_binary, y_pred_binary)
# print("Accuracy:", accuracy)


# svm = LinearSVC(random_state=42)
# multilabel_classifier = MultiOutputClassifier(svm, n_jobs=-1)
#
# param_grid = {'estimator__C': [1], 'estimator__gamma': [0.01], 'estimator__kernel': ['linear'],  'decision_function_shape':['ovr']}
#
# # Define k-fold cross-validation
# kfolds = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
# mlb = MultiLabelBinarizer()
#
# # Fit and transform the multi-label target variable
# y_train_binary = mlb.fit_transform(y_train['sentiment'])
# print()
# multilabel_classifier = multilabel_classifier.fit(X_train, y_train_binary)

