import sys
import pandas as pd

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelTrainer import NNModelTrainer
from NNModelTester import NNModelTester

df = pd.read_csv('../../DATA/HSC-unWISE-W03__FIELD.csv')
data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', fields_list= [])
data.data = df
data.prep()


features = ['all_features', 'all_features_bcg', 'all_features_sigmas']
for feat in features:
    
    data.features_txt = feat + '.txt'
    data.features_labels()

    model = NNModelTrainer(data= data, name = feat)
    model.load_model()

    tester = NNModelTester(model = model.model, data = data, name = feat)
    tester.predict(optimize_threshold= False)
    tester.write_report()