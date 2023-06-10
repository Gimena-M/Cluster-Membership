### Classes directory

Directory that contains classes that handle data, train models and test models.

-----------------------------------
### Scripts and tests directories

+ NN: neural networks scripts.
+ RF: random forest scripts.
+ SVM: support vector machine scripts.
+ smote: tests for data augmentation using smote + undersampling.

-----------------------------------
### DATA directory

Contains tables with data, used by NN, RF, SVM and smote. Also contains scripts:

+ sql_scripts: scripts to write and submit SQL queries, and join results with existing tables.
+ add_features: script to add a set of features to a table.
+ clean_data_script: script to remove outliers, NaN and inf from tables.
+ cut_z: script (and notebook) to remove galaxies whose redshift differs significantly from the redshift of the nearest BCG (output are the z_filtered tables)

Tables are also in https://drive.google.com/drive/folders/1z3me5bzOujSoPwNI5bzpuJ7LwGnnaSzX?usp=share_link
(in case the Git LFS bandwidth or storage quota is exceeded.) Make sure to save them to the DATA directory.