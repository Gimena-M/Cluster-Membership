### Classes directory

Directory that contains classes that handle data, train models and test models.

-----------------------------------
### Scripts and tests directories

+ NN: neural networks scripts.
+ RF: random forest scripts.
+ SVM: support vector machine tests.

-----------------------------------
### DATA directory

Contains tables with data, used by NN, RF, SVM and smote. 

The scripts directory contains:

+ sql_scripts: scripts to write and submit SQL queries, and join results with existing tables.
+ clean_data_script: script to remove outliers, NaN and inf from tables.
+ add_features: script to add a set of features to a table, and script to add absolute magnitudes to tables.
+ cut_z: script (and notebook) to remove galaxies whose redshift differs significantly from the redshift of the nearest BCG (output are the z_filtered tables)

The augmentation directory contains notebooks testing SMOTE and normalizing flows.

Tables are in https://drive.google.com/drive/folders/1z3me5bzOujSoPwNI5bzpuJ7LwGnnaSzX?usp=share_link
Make sure to save them to the DATA directory.
**Last Drive update: 23/07/23**