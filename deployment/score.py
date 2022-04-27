import os
import sys
import numpy as np
import joblib
#from sklearn.externals import joblib

import math
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import json
import re
import traceback
import logging
from sklearn.ensemble import RandomForestClassifier

'''
Inference script for Loan Classification:

'''

def init():
    '''
    Initialize required models:
        Get the LOAN Model from Model Registry and load
    '''
    global prediction_dc
    global model
    prediction_dc = ModelDataCollector("LOAN", designation="predictions", feature_names=['Person_ID', 'Loan Account Number', 'Relationship_Start_Date', 'OCCUPATION', 'DATE_OF_BIRTH', 'BUSINESS_TYPE', 'STATE', 'No_of_Mobile_No', 'CUSTOMER_EMAIL', 'GENDER', 'MARITAL_STATUS', 'REGION', 'BASIC_CURRENT', 'BASIC_SAVINGS', 'ATMCARD', 'TOTAL_PRODUCTS', 'AMOUNT_IN_NAIRA', 'TRANS_TYPE_Debit', 'Loan_ID', 'Loan Tenure', 'Payment Period', 'Loan Amount (Principal)', 'Loan Application Date', 'Loan Approval Date', 'Loan Disbursement Date', 'Loan Maturity Date', 'Latest Known Status', 'Ever 90dpd+', 'Currently \u2265 60dpd'])

    model_path = Model.get_model_path('LOAN')
    model = joblib.load(model_path+"/"+"loan_model.pkl")
    print('loan model loaded...')

def create_response(predicted_loan_default):
    '''
    Create the Response object
    Arguments :
        predicted_label : Predicted LOAN default
    Returns :
        Response JSON object
    '''
    resp_dict = {}
    print("Predicted Species : ",predicted_loan_default)
    resp_dict["loan_default"] = str(predicted_loan_default)
    return json.loads(json.dumps({"output" : resp_dict}))

def run(raw_data):
    '''
    Get the inputs and predict the LOAN default
    '''
    try:
        data = json.loads(raw_data)
        arg1 = data['Person_ID']
        arg2 = data['Loan Account Number']
        arg3 = data['Relationship_Start_Date']
        arg4 = data['OCCUPATION']
        arg5 = data['DATE_OF_BIRTH']
        arg6 = data['BUSINESS_TYPE']
        arg7 = data['STATE']
        arg8 = data['No_of_Mobile_No']
        arg9 = data['CUSTOMER_EMAIL']
        arg10 = data['GENDER']
        arg11 = data['MARITAL_STATUS']
        arg12 = data['REGION']
        arg13 = data['BASIC_CURRENT']
        arg14 = data['BASIC_SAVINGS']
        arg15 = data['ATMCARD']
        arg16 = data['TOTAL_PRODUCTS']
        arg17 = data['AMOUNT_IN_NAIRA']
        arg18 = data['TRANS_TYPE_Debit']
        arg19 = data['Loan_ID']
        arg20 = data['Loan Tenure']
        arg21 = data['Payment Period']
        arg22 = data['Loan Amount (Principal)']
        arg23 = data['Loan Application Date']
        arg24 = data['Loan Approval Date']
        arg25 = data['Loan Disbursement Date']
        arg26 = data['Loan Maturity Date']
        arg27 = data['Latest Known Status']
        arg28 = data['Ever 90dpd+']
        arg29 = data['Currently \u2265 60dpd']

        predictedion = model.predict([[arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29]])[0]
        prediction_dc.collect([arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29])
        return create_response(predictedion)
    except Exception as err:
        traceback.print_exc()