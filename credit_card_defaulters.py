import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Default Detection </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.write("""
This app predicts the **Defaulters**!
""")

df = pd.read_csv('C:\\Users\\Lenovo\\cleaned_data.csv')

   
EDUCATION = st.selectbox('EDUCATION',('graduate school','university', 'high school'))
MARRIAGE = st.selectbox('MARRIAGE',('married','single','others'))
AGE = st.text_input('AGE','type here')
LIMIT_BAL = st.text_input('LIMIT_BAL','type here')
PAY_1 = st.selectbox('PAY_1',('Account started that month with a zero balance or never used any credit','Account had a balance that was paid in full',"At least the minimum payment was made,but the entire balance wasn't paid",'Payment delay for 1 month','Payment delay for 2 month','Payment delay for 3 month','Payment delay for 4 month','Payment delay for 5 month','Payment delay for 6 month','Payment delay for 7 month','Payment delay for 8 month'))

st.subheader("""
 Bill Amount For Past 6 Months
""")
    
BILL_AMT1 = st.text_input('','Last month bill amount')
BILL_AMT2 = st.text_input('','Last 2nd month bill amount')
BILL_AMT3 = st.text_input('','Last 3rd month bill amount')
BILL_AMT4 = st.text_input('','Last 4th month bill amount')
BILL_AMT5 = st.text_input('','Last 5th month bill amounte')
BILL_AMT6 = st.text_input('','Last 6th month bill amount')


st.subheader("""
 Paid Amount For Past 6 Months
""")
PAY_AMT1 = st.text_input('','Amount paid in last month')
PAY_AMT2 = st.text_input('','Amount paid in last 2nd month')
PAY_AMT3 = st.text_input('','Amount paid in last 3rd month')
PAY_AMT4 = st.text_input('','Amount paid in last 4th month')
PAY_AMT5 = st.text_input('','Amount paid in last 5th month')
PAY_AMT6 = st.text_input('','Amount paid in last 6th month')


df = pd.read_csv('C:\\Users\\Lenovo\\cleaned_data.csv')


data = {'EDUCATION':EDUCATION,'MARRIAGE':MARRIAGE,'AGE':AGE,'LIMIT_BAL':LIMIT_BAL,'BILL_AMT1':BILL_AMT1,'BILL_AMT2':BILL_AMT2,'BILL_AMT3':BILL_AMT3,'BILL_AMT4':BILL_AMT4,'BILL_AMT5':BILL_AMT5,'BILL_AMT6':BILL_AMT6,'PAY_AMT1':PAY_AMT1,'PAY_AMT2':PAY_AMT2,'PAY_AMT3':PAY_AMT3,'PAY_AMT4':PAY_AMT4,'PAY_AMT5':PAY_AMT5,'PAY_AMT6':PAY_AMT6,'PAY_1':PAY_1}
features = pd.DataFrame(data, index=[0])

features['MARRIAGE'] = features['MARRIAGE'].map({'married':1,'single':2,'others':3})
features['EDUCATION'] = features['EDUCATION'].map({'graduate school':1, 'university':2,'high school':3})
features['PAY_1']=features['PAY_1'].map({'Account started that month with a zero balance or never used any credit':-2,'Account had a balance that was paid in full':-1,"At least the minimum payment was made,but the entire balance wasn't paid":0,'Payment delay for 1 month':1,'Payment delay for 2 month':2,'Payment delay for 3 month':3,'Payment delay for 4 month':4,'Payment delay for 5 month':5,'Payment delay for 6 month':6,'Payment delay for 7 month':7,'Payment delay for 8 month':8})

input =  data
st.subheader('User Input parameters')
st.write(features)


features_response = df.columns.tolist()

items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university',]

features_response = [item for item in features_response if item not in items_to_remove]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[features_response[:-1]], df['default payment next month'],
test_size=0.2, random_state=24)

rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)




rf.fit(X_train, y_train)
y_predict = rf.predict(features)
y_test_predict_proba = rf.predict_proba(features)
a = []
n = len(y_predict)
for i in range(0,n):
    if(y_predict[i]== 0):
        a.append('NOT_DEFAULTED')
    else:
        a.append('DEFAULTED')
       
b = pd.DataFrame(a)

y_test_predict_proba=pd.DataFrame(y_test_predict_proba,columns =['NOT_DEFAULTED','DEFAULTED'])

st.subheader('Prediction Probability')
st.write(y_test_predict_proba)

if st.button("Predict"):
    st.success(' {}'.format(a))
    


