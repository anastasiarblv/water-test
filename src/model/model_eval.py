import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#test_data = pd.read_csv("./data/processed/test_processed.csv")
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")

# Берем то же самое что бы ранее писали в файле model_building.py, но меняем train на test уже
#X_test = test_data.iloc[:,0:-1].values # берем все строки и все столбцы (кроме столбца target, y_test) из test_data
#y_test = test_data.iloc[:,-1].values   # наш целевой (target, y_test) столбец из test_data
def prepare_data(data): # data = train_data
    try:
        X_test = data.drop(columns = ['Potability'], axis = 1)
        y_test = data['Potability']
        return X_test, y_test
    except Exception as e:
        raise Exception(f"Error Preparing data : {e}")


# Теперь заружаем ранее созданную в файле model_building.py модель, которуя представляет собойт отдельный файл
# model.pkl, который у нас повился после dvc repro на этапе model_building
#model = pickle.load(open("model.pkl", "rb"))
def load_model(model_name):
    try:
        with open(model_name, "rb") as file:
            model = pickle.load(file)
        return model 
    except Exception as e:
        raise Exception(f"Error loading model from {model_name} : {e}")

# Теперь сделаем прогнозирование на наших данных X_test
#y_pred = model.predict(X_test)
# И теперь найдем значение показателя Точности (Accuracy Score)
#acc = accuracy_score(y_test, y_pred) # y_test = наши фактические данные, y_pred = наши прогнозные данные, полученные на основе модели
#pre = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1score = f1_score(y_test, y_pred)

# Теперь сохраним эти данные (по acc, pre, recall, f1score) в формте JSON,
# для этого создадим словарь metrics_dict
#metrics_dict = {
#    'acc':acc,
#    'precision':pre,
#    'recall' : recall,
#    'f1_score': f1score}

def evaluation_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) 
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        metrics_dict = {
            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score}
        return metrics_dict 
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")



# Теперь создадим непосредственно сам файл JSON, и запишем туда наши данные по метрикам:
#with open("metrics.json", "w") as file:
#    json.dump(metrics_dict, file, indent=4)
# Мы создали данный фалй с метриками в формате JSON, чтобы мы могли после всех наших проделанных этапов
# написать в VSCODE команду dvc metrics show, и увидеть значения по всем метрикам для данной модели.
def save_metrics(metrics_dict, filepath):
    try:
        with open("metrics.json", "w") as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath} : {e}")
    
##############
def main():
    data_filepath = "./data/processed/test_processed.csv"
    metrics_filepath = "metrics.json"
    model_name = "model.pkl"
    try:
        test_data = load_data(data_filepath)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_name)
        metrics_dict = evaluation_model(model, X_test, y_test)
        save_metrics(metrics_dict, metrics_filepath)
    except Exception as e:
        raise Exception(f"An error occurred : {e}")
if __name__ == "__main__":
    main()