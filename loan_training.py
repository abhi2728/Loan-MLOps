#Testing Abhi
import os
import sys

import azureml as aml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.run import Run
import azureml._restclient.snapshots_client
import argparse
import json
import time
#import traceback
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import re
import math
import seaborn as sn
import matplotlib.pyplot as plt
#from sklearn.externals import joblib
import joblib

'''
LOAN Classification
'''
class LOANClassification():
    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Create directories
        '''
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)
        

    def get_files_from_datastore(self, container_name, file_name):
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        '''
        datastore_paths = [(self.datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name     
        if dataset_name not in self.workspace.datasets:
            data_ds = data_ds.register(workspace=self.workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds 

    def create_pipeline(self):
        '''
        LOAN Data training and Validation
        '''        
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        input_ds = self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df = input_ds.to_pandas_dataframe()

        # url = "https://loanmlwostorage00e497186.blob.core.windows.net/azureml-blobstore-c9fcaddc-b2c3-4674-9ec4-96c309832ac2/loan_data/loan_data.csv?sp=r&st=2022-04-24T12:20:37Z&se=2022-05-06T20:20:37Z&spr=https&sv=2020-08-04&sr=b&sig=WPXDJfiFdFtN5QWWqSfaOmaMU75z9e7tzhg8BFv3Pyk%3D"
        # final_df = pd.read_csv(url)
        print("Received datastore")
        
        print("Input DF Info",final_df.info())
        print("Input DF Head",final_df.head())

        X = final_df.drop(labels=['Bad Indicator'], axis=1)
        y = final_df[['Bad Indicator']]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1980)
        pipeline = Pipeline(steps=[
                    ('LoanData_Tranform', LoanDataTransformer()), 
                    ('RandomForestClassifier', RandomForestClassifier(n_estimators=500,
                                                                criterion='gini', 
                                                                max_features='auto',
                                                                random_state=43))
        ])

        model = pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Model Score : ", model.score(X_test,y_test))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

    def create_confusion_matrix(self, y_true, y_pred, name):
        '''
        Create confusion matrix
        '''
        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")
            self.run.log_image(name=name, plot=plt)
        except Exception as e:
            #traceback.print_exc()    
            logging.error("Create consufion matrix Exception")

    def create_outputs(self, y_true, y_pred, X_test, name):
        '''
        Create prediction results as a CSV
        '''
        pred_output = {"Actual Bad Indicator" : y_true['Bad Indicator'].values, "Predicted Bad Indicator": y_pred['Bad Indicator'].values}        
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

    def validate(self, y_true, y_pred, X_test):
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))

        # self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")

        y_pred_df = pd.DataFrame(y_pred, columns = ['Bad Indicator'])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag("LOANClassifierFinalRun") 

class LoanDataTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        '''
        Initialize LoanDataTransformer
        '''

    def fit(self, X, y = None):
        '''
        fit LoanDataTransformer
        '''
        return self

    def transform(self, X, y = None):
        '''
        transform LoanDataTransformer
        '''
        X = self.date_cols(X)
        X = self.data_imputation(X)
        X = self.date_to_days(X)
        X = self.data_frequency_encode(X)
        X = self.data_onehotendocde(X)
        X = self.data_feature_select(X)
        X = self.data_correlation(X)
        return X
    
    def data_imputation(self, X):
        '''
        Missing Data Imputation for feature columns
        '''        
        X['BUSINESS_TYPE'] = X['BUSINESS_TYPE'].fillna('MISSING')
        X['OCCUPATION'] = X['OCCUPATION'].fillna('MISSING')
        return X
    
    def date_to_days(self, X):
        '''
        Trandform Date to days for feature columns
        '''        
        X['RELATIONSHIP_START_DAYS'] = (pd.to_datetime('today') - X['Relationship_Start_Date']).dt.days
        X['AGE'] = (pd.to_datetime('today') - X['DATE_OF_BIRTH']).dt.days
        X['LOAN_APPLICATION_DAYS'] = (pd.to_datetime('today') - X['Loan Application Date']).dt.days
        X['LOAN_APPROVAL_DAYS'] = (pd.to_datetime('today') - X['Loan Approval Date']).dt.days
        X['LOAN_DISBURSEMENT_DAYS'] = (pd.to_datetime('today') - X['Loan Disbursement Date']).dt.days
        X['LOAN_MATURITY_DAYS'] = (pd.to_datetime('today') - X['Loan Maturity Date']).dt.days
        return X
    
    def data_frequency_encode(self, X):
        '''
        Frequency Encoding for feature columns
        '''        
        X_frq = X['Latest Known Status'].value_counts().to_dict()
        X['LATEST_KNOWN_STATUS_ENCODED'] = X['Latest Known Status'].map(X_frq)

        X_frq = X['OCCUPATION'].value_counts().to_dict()
        X['OCCUPATION_ENCODED'] = X['OCCUPATION'].map(X_frq)

        X_frq = X['BUSINESS_TYPE'].value_counts().to_dict()
        X['BUSINESS_TYPE_ENCODED'] = X['BUSINESS_TYPE'].map(X_frq)

        X_frq = X['STATE'].value_counts().to_dict()
        X['STATE_ENCODED'] = X['STATE'].map(X_frq)
        
        X_frq = X['MARITAL_STATUS'].value_counts().to_dict()
        X['MARITAL_STATUS_ENCODED'] = X['MARITAL_STATUS'].map(X_frq)

        X_frq = X['REGION'].value_counts().to_dict()
        X['REGION_ENCODED'] = X['REGION'].map(X_frq)

        X_frq = X['BASIC_CURRENT'].value_counts().to_dict()
        X['BASIC_CURRENT_ENCODED'] = X['BASIC_CURRENT'].map(X_frq)

        X_frq = X['ATMCARD'].value_counts().to_dict()
        X['ATMCARD_ENCODED'] = X['ATMCARD'].map(X_frq)
        return X

    def data_onehotendocde(self, X):
        '''
        One Hot Endoing Encoding for feature columns
        '''        
        X_frq = X['Latest Known Status'].value_counts().to_dict()
        X['LATEST_KNOWN_STATUS_ENCODED'] = X['Latest Known Status'].map(X_frq)

        X_dummies = pd.get_dummies(X.GENDER, drop_first=True, prefix='GENDER')
        X = pd.concat([X, X_dummies], axis=1)
        return X

    def data_feature_select(self, X):
        '''
        Select Numeric Column feature columns
        '''        
        X = X[['Loan Tenure', 'GENDER_M', 'GENDER_P', 'No_of_Mobile_No',
                        'RELATIONSHIP_START_DAYS', 'AGE', 'LOAN_APPLICATION_DAYS', 
                        'LOAN_APPROVAL_DAYS', 'LOAN_DISBURSEMENT_DAYS', 'LOAN_MATURITY_DAYS', 
                        'OCCUPATION_ENCODED', 'BUSINESS_TYPE_ENCODED', 'STATE_ENCODED', 
                        'MARITAL_STATUS_ENCODED', 'REGION_ENCODED', 'BASIC_CURRENT_ENCODED', 
                        'ATMCARD_ENCODED', 'TOTAL_PRODUCTS', 'AMOUNT_IN_NAIRA', 'TRANS_TYPE_Debit', 
                        'LATEST_KNOWN_STATUS_ENCODED','Loan Amount (Principal)', 'Ever 90dpd+', 
                        'Currently ≥ 60dpd']]
        return X
    
    def data_correlation(self, X):
        '''
        Remove corelated features
        '''        
        corr_features = self.correlation(X, 0.9)
        X = X.drop(corr_features, axis=1)
        return X
    
    def correlation(self, X, threshold):
        '''
        Remove corelated features
        '''
        col_corr = set() 
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: 
                    colname = corr_matrix.columns[i] 
                    col_corr.add(colname)
        return col_corr

    def data_onehotendocde(self, X):
        '''
        One Hot Endoing Encoding for feature columns
        '''        
        X_frq = X['Latest Known Status'].value_counts().to_dict()
        X['LATEST_KNOWN_STATUS_ENCODED'] = X['Latest Known Status'].map(X_frq)

        X_dummies = pd.get_dummies(X.GENDER, drop_first=True, prefix='GENDER')
        X = pd.concat([X, X_dummies], axis=1)
        return X

    def date_cols(self, X):
        '''
        Convert to date columns 
        '''        
        X['Relationship_Start_Date'] =  pd.to_datetime(X['Relationship_Start_Date'])
        X['DATE_OF_BIRTH'] =  pd.to_datetime(X['DATE_OF_BIRTH'])
        X['Loan Application Date'] =  pd.to_datetime(X['Loan Application Date'])
        X['Loan Approval Date'] =  pd.to_datetime(X['Loan Approval Date'])
        X['Loan Disbursement Date'] =  pd.to_datetime(X['Loan Disbursement Date'])
        X['Loan Maturity Date'] =  pd.to_datetime(X['Loan Maturity Date'])
        return X 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QA Code Indexing pipeline')
    parser.add_argument('--container_name', type=str, help='Path to default datastore container')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--model_path', type=str, help='Path to store the model')
    parser.add_argument('--artifact_loc', type=str, 
                        help='DevOps artifact location to store the model', default='')
    args = parser.parse_args()

    loan_classifier = LOANClassification(args)
    loan_classifier.create_pipeline()
    
