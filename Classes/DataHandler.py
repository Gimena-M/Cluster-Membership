"""
Class to handle data reading and preparing.

Arguments for initialization are:
    * validation_sample (bool): When splitting into training and testing samples, include a validation sample?
    * features_txt (str): txt file with a list of features (must be in the DATA directory)
    * fields_list (list): List of HSC fields. (default: ["W01", "W02", "W03", "W04"])
    * z_filtered (bool): Use tables where galaxies were removed if their redshift differed significantly from the redshift of the nearest BCG? (default: False)
    * min_n500 (int), max_n500 (int): Limits for the nearest cluster n500. (default: None)
    * min_z (float), max_z (float): Limits for the nearest cluster z. (default: None)
    * feat_max (dict): Limits for galaxy features (as a dictionary, e.g.: feat_max = {"i_cmodel_mag" : 19, "phot_z": 0.5}). (default: {})
    * random_state (int): For training-testing and training-testing-validation split. (default: 42)
    * balance (str or None): Balancing strategy for the training sample. Can be 'weights', 'smote', 'undersample', or None. (default: None)

The main() method reads and prepares the data to be used by most scripts.
The prep() method only prepares the data.
"""

import os
import numpy as np
import pandas as pd
import copy

class DataHandler:
    def __init__(self, validation_sample: bool, features_txt: str, labels: str = 'member',
                 fields_list: list = ['W01','W02','W03','W04'], z_filtered: bool = False,
                 min_n500: int|None = None, max_n500: int|None = None, min_z: int|float = None, max_z: int|float = None,
                 feat_max: dict = {}, feat_min: dict = {},
                 random_state: int = 42, balance: str|None = None):
        self.validation_sample = validation_sample
        self.features_txt = features_txt
        self.labels = labels
        self.fields_list = fields_list
        self.z_filtered = z_filtered
        self.min_n500 = min_n500
        self.max_n500 = max_n500
        self.min_z = min_z
        self.max_z = max_z
        self.feat_max = feat_max
        self.feat_min = feat_min
        self.random_state = random_state
        self.balance = balance

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, '../DATA/')

    def args(self):
        # return dictionary with attributes.
        dd = dict(
            features_txt = self.features_txt,
            fields_list = self.fields_list,
            z_filtered = self.z_filtered,
            min_n500 = self.min_n500,
            max_n500 = self.max_n500,
            min_z = self.min_z,
            max_z = self.max_z,
            feat_max = self.feat_max,
            feat_min = self.feat_min,
            random_state = self.random_state, 
            balance = self.balance
        )
        return dd

    def main(self, only_members: bool = False):
        # read, prepare and split data.
        self.read_data(only_members = only_members)
        self.z_n500_limits()
        self.features_labels()
        self.feature_limits()

        if self.validation_sample:
            self.split_3()
        else:
            self.split_2()

        self.balancing()

    def prep(self):
        # prepare and split data.
        self.z_n500_limits()
        self.features_labels()
        self.feature_limits()

        if self.validation_sample:
            self.split_3()
        else:
            self.split_2()

        self.balancing()
        

    def read_data(self, only_members: bool = False):
        # read and join HSC tables 
        li = []
        for fi in self.fields_list:
            if self.z_filtered:
                d = pd.read_csv(f'{self.data_dir}z-filtered_clean-HSC-unWISE-{fi}.csv')
            else:
                d = pd.read_csv(f'{self.data_dir}clean-HSC-unWISE-{fi}.csv')
            li.append(d)
        self.data = pd.concat(li, axis = 'rows')

        if only_members:
            self.data = self.data[self.data.member == 1]

    def features_labels(self):

        # read features list
        with open(f'{self.data_dir}features_lists/{self.features_txt}') as file:
            self.features = file.read().splitlines()

        # print number of members
        n_mem = self.data[self.data.member == 1].shape[0]
        n_no = self.data[self.data.member == 0].shape[0]
        n = self.data.shape[0]
        print ('Members: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
        print ('Non members: {} ({:.2f}%)'.format(n_no, n_no/n*100))
        print('-'*70)

    def z_n500_limits(self):

        # Select galaxies near clusters in a z and n500 range...
        # df has a column 'id_cl_near', with a list of ids of the nearests clusters. 

        if any(val != None for val in [self.min_n500, self.max_n500, self.min_z, self.max_z]):
            df_cl = pd.read_table(self.data_dir + 'clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])

            cond = True
            if self.min_z != None:
                cond = (cond) & (df_cl.phot_z_cl >= self.min_z)
            if self.max_z != None:
                cond = (cond) & (df_cl.phot_z_cl <= self.max_z)
            if self.min_n500 != None:
                cond = (cond) & (df_cl.n500_cl >= self.min_n500)
            if self.max_n500 != None:
                cond = (cond) & (df_cl.n500_cl <= self.max_n500)

            df_cl = df_cl[cond]
            try:
                self.data['id_cl_near'] = [eval(s) for s in self.data['id_cl_near']] # this column is saved as a string in the .csv, and needs to be evaluated
            except:
                pass # in case it has alredy been evaluated...
            self.data = self.data[[any([id in df_cl['id_cl'].values for id in gal.id_cl_near]) for _,gal in self.data.iterrows()]]

    def feature_limits(self):

        # remove galaxies with features outside of limits given by feat_max and feat_min

        if any([self.feat_max, self.feat_min]):
            conds = []
            for key in self.feat_max.keys():
                conds.append((self.data[key] <= float(self.feat_max[key])))
            for key in self.feat_min.keys():
                conds.append((self.data[key] >= float(self.feat_min[key])))
            conds = np.array(conds)
            
            # for c in conds:
            #     self.data = self.data[c]
            c = []
            for i in range(conds.shape[1]):   
                c.append((all(conds[:,i])))
            self.data = self.data[c]

            # print number of members
            n_mem = self.data[self.data.member == 1].shape[0]
            n_no = self.data[self.data.member == 0].shape[0]
            n = self.data.shape[0]
            print ('Members after feature limits: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
            print ('Non members after feature limits: {} ({:.2f}%)'.format(n_no, n_no/n*100))
            print('-'*70)

    def split_2(self):

        # split into training and testing samples. 
        from sklearn.model_selection import train_test_split
        self.training, self.testing = train_test_split(self.data, test_size = 0.3, stratify = self.data[self.labels], random_state = self.random_state)

        print ('Training: {} members, {} non members'.format(self.training[self.training.member == 1].shape[0], self.training[self.training.member == 0].shape[0]))
        print ('Testing: {} members, {} non members'.format(self.testing[self.testing.member == 1].shape[0], self.testing[self.testing.member == 0].shape[0]))
        print('-'*70)

    def split_3(self):

        # split into training, testing and validation samples. 
        from sklearn.model_selection import train_test_split
        if np.issubdtype(self.data_labels().dtype, float):
            # if labels are floats, do not stratify
            self.training, self.testing = train_test_split(self.data, test_size = 0.3, random_state = self.random_state)
            self.validation, self.testing = train_test_split(self.testing, test_size = 0.3, random_state = self.random_state)
        else:
            self.training, self.testing = train_test_split(self.data, test_size = 0.3, stratify = self.data_labels(), random_state = self.random_state)
            self.validation, self.testing = train_test_split(self.testing, test_size = 0.3, stratify = self.testing_labels(), random_state = self.random_state)

        print ('Training: {} members, {} non members'.format(self.training[self.training_labels() == 1].shape[0], self.training[self.training_labels() == 0].shape[0]))
        print ('Validation: {} members, {} non members'.format(self.validation[self.validation_labels() == 1].shape[0], self.validation[self.validation_labels() == 0].shape[0]))
        print ('Testing: {} members, {} non members'.format(self.testing[self.testing_labels() == 1].shape[0], self.testing[self.testing_labels() == 0].shape[0]))
        print('-'*70)

    def data_features(self):
        return self.data[self.features]
    def data_labels(self):
        return self.data[self.labels]
    def testing_features(self):
        return self.testing[self.features]
    def testing_labels(self):
        return self.testing[self.labels]
    def training_features(self):
        return self.training[self.features]
    def training_labels(self):
        return self.training[self.labels]
    def validation_features(self):
        return self.validation[self.features]
    def validation_labels(self):
        return self.validation[self.labels]

    def undersample(self, ratio: float = 1.):
        # Undersample majority class 
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(sampling_strategy= ratio, replacement= False)
        rus_feat, rus_lab = rus.fit_resample(self.training[self.features], self.training[self.labels])
        rus_feat[self.labels] = rus_lab

        n_mem = rus_feat[rus_feat.member == 1].shape[0]
        n_no = rus_feat[rus_feat.member == 0].shape[0]
        n = rus_feat.shape[0]
        print ('Training members after undersampling...: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
        print ('Training non members after undersampling...: {} ({:.2f}%)'.format(n_no, n_no/n*100))
        del n,n_mem,n_no

        self.training = rus_feat
        self.class_weights()

    def class_weights(self):
        # compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        wei =  compute_class_weight(class_weight = 'balanced', classes = np.unique(self.training_labels()), y = self.training_labels())
        weights = {}
        for w,l in zip(wei,np.unique(self.training_labels())):
            weights[l] = w
        self.weights = weights

        return weights

    def smote(self):
        # deal with imbalanced data using undersampling + smote
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        # undersampling majority class
        rus = RandomUnderSampler(sampling_strategy= 3./7, replacement= False)
        rus_feat, rus_lab = rus.fit_resample(self.training_features(), self.training_labels())
        # smote
        smote = SMOTE(sampling_strategy= 2./3)
        smote_feat, smote_lab = smote.fit_resample(rus_feat, rus_lab)

        n_mem = len(smote_lab[smote_lab == 1])
        n_no = len(smote_lab[smote_lab == 0])
        n = len(smote_lab)
        print ('Training members after SMOTE: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
        print ('Training non members after SMOTE: {} ({:.2f}%)'.format(n_no, n_no/n*100))

        smote_feat[self.labels] = smote_lab
        self.training = smote_feat
        self.class_weights()

    def balancing(self):
        self.class_weights()
        match self.balance:
            case None:
                pass
            case 'undersample':
                self.undersample()
            case 'smote':
                self.smote()
            case 'weights':
                return self.weights
            case _:
                raise ValueError("Invalid value for 'balance'")

    def copy(self):
            return copy.copy(self)