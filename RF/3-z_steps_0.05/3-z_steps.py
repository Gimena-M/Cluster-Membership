import sys
sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

model_params =  {
    "n_estimators" : 120,
    "criterion" : "entropy",
    "max_depth" : 30,
    "min_samples_split" : 2, 
    "min_samples_leaf" : 1,
    "max_features" : "log2"
}

data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', fields_list=['W01', 'W02', 'W03', 'W04'])
data.main()
z_lims = [x / 100 for x in range(20, 131, 5)] # z from 0.2 to 1.3 with 0.05 step

for i in range(len(z_lims) - 1):
    
    data.min_z = z_lims[i]
    data.max_z = z_lims[i + 1]
    name = f"z_{z_lims[i] :.2f}_to_{z_lims[i+1] :.2f}"

    cont = RFModelController(data = data.copy(), name = name)
    cont.main_model(model_params= model_params, prep_data= True, sort_importances= False)
