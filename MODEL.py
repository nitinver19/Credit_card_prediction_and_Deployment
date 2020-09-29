#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle


html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Default Detection </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

activities=['Random Forest','Logistic Regression','Decision Tree']
option=st.sidebar.selectbox('Which model would you like to use?',activities)

st.write("""
This app predicts the **Defaulters**!
""")
   
EDUCATION = st.selectbox('EDUCATION',('graduate school','university', 'high school'))
MARRIAGE = st.selectbox('MARRIAGE',('married','single','others'))
AGE = st.text_input('AGE','25')
LIMIT_BAL = st.text_input('LIMIT_BAL','40000')
PAY_1 = st.selectbox('PAY_1',('Account started that month with a zero balance or never used any credit','Account had a balance that was paid in full',"At least the minimum payment was made,but the entire balance wasn't paid",'Payment delay for 1 month','Payment delay for 2 month','Payment delay for 3 month','Payment delay for 4 month','Payment delay for 5 month','Payment delay for 6 month','Payment delay for 7 month','Payment delay for 8 month'))

st.subheader("""
 Bill Amount For Past 6 Months
""")
    
BILL_AMT1 = st.text_input('Last month bill amount','40000')
BILL_AMT2 = st.text_input('Last 2nd month bill amount','10000')
BILL_AMT3 = st.text_input('Last 3rd month bill amount','0')
BILL_AMT4 = st.text_input('Last 4th month bill amount','0')
BILL_AMT5 = st.text_input('Last 5th month bill amount','0')
BILL_AMT6 = st.text_input('Last 6th month bill amount','0')


st.subheader("""
 Paid Amount For Past 6 Months
""")
PAY_AMT1 = st.text_input('Amount paid in last month','20000')
PAY_AMT2 = st.text_input('Amount paid in last 2nd month','10000')
PAY_AMT3 = st.text_input('Amount paid in last 3rd month','20000')
PAY_AMT4 = st.text_input('Amount paid in last 4th month','40000')
PAY_AMT5 = st.text_input('Amount paid in last 5th month','10000')
PAY_AMT6 = st.text_input('Amount paid in last 6th month','0')


data = {'EDUCATION':EDUCATION,'MARRIAGE':MARRIAGE,'AGE':AGE,'LIMIT_BAL':LIMIT_BAL,'BILL_AMT1':BILL_AMT1,'BILL_AMT2':BILL_AMT2,'BILL_AMT3':BILL_AMT3,'BILL_AMT4':BILL_AMT4,'BILL_AMT5':BILL_AMT5,'BILL_AMT6':BILL_AMT6,'PAY_AMT1':PAY_AMT1,'PAY_AMT2':PAY_AMT2,'PAY_AMT3':PAY_AMT3,'PAY_AMT4':PAY_AMT4,'PAY_AMT5':PAY_AMT5,'PAY_AMT6':PAY_AMT6,'PAY_1':PAY_1}
features = pd.DataFrame(data, index=[0])

features['MARRIAGE'] = features['MARRIAGE'].map({'married':1,'single':2,'others':3})
features['EDUCATION'] = features['EDUCATION'].map({'graduate school':1, 'university':2,'high school':3})
features['PAY_1']=features['PAY_1'].map({'Account started that month with a zero balance or never used any credit':-2,'Account had a balance that was paid in full':-1,"At least the minimum payment was made,but the entire balance wasn't paid":0,'Payment delay for 1 month':1,'Payment delay for 2 month':2,'Payment delay for 3 month':3,'Payment delay for 4 month':4,'Payment delay for 5 month':5,'Payment delay for 6 month':6,'Payment delay for 7 month':7,'Payment delay for 8 month':8})

input =  features
st.subheader('User Input parameters')

st.write(features)

st.subheader('MODEL OPTED')
st.write(option)

load_rf = pickle.load(open('credit_model_rf.pkl','rb'))
load_log = pickle.load(open('credit_model_log.pkl','rb'))
load_tree = pickle.load(open('credit_model_tree.pkl','rb'))

    
if st.button('Predict'):
        if option=='Random Forest':
            y_predict = load_rf.predict(input)
            y_test_predict_proba = load_rf.predict_proba(input)
            a = []
            n = len(y_predict)
            for i in range(0,n):
                if(y_predict[i]== 0):
                    a.append('NOT_DEFAULTED')
                else:
                    a.append('DEFAULTED')

            y_test_predict_proba=pd.DataFrame(y_test_predict_proba,columns =['NOT_DEFAULTED','DEFAULTED'])

            st.subheader('Prediction Probability')
            st.write(y_test_predict_proba)
            st.success(' {}'.format(a))
        elif option=='Logistic Regression':
            y_predict1 = load_log.predict(input)
            y_test_predict_proba1 = load_log.predict_proba(input)
            b = []
            n1 = len(y_predict1)
            for p in range(0,n1):
                if(y_predict1[p]== 0):
                    b.append('NOT_DEFAULTED')
                else:
                    b.append('DEFAULTED')


            y_test_predict_proba1=pd.DataFrame(y_test_predict_proba1,columns =['NOT_DEFAULTED','DEFAULTED'])

            st.subheader('Prediction Probability')
            st.write(y_test_predict_proba1)
            st.success(' {}'.format(b))
        else:
            y_predict2 = load_tree.predict(input)
            y_test_predict_proba2 = load_tree.predict_proba(input)
            c = []
            n2 = len(y_predict2)
            for q in range(0,n2):
                if(y_predict2[q]== 0):
                    c.append('NOT_DEFAULTED')
                else:
                    c.append('DEFAULTED')


            y_test_predict_proba2=pd.DataFrame(y_test_predict_proba2,columns =['NOT_DEFAULTED','DEFAULTED'])

            st.subheader('Prediction Probability')
            st.write(y_test_predict_proba2)
            st.success(' {}'.format(c))
    
    
    

