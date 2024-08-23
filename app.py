import pandas as pd
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


pickle_in = open('churnclassifier.pkl', 'rb')
churn_classifier = pickle.load(pickle_in)

# Define original min and max values of tenure
min_tenure = 0  # Replace with the actual minimum value from your dataset
max_tenure = 72  # Replace with the actual maximum value from your dataset

def scale_tenure(tenure, min_value, max_value):
    return (tenure - min_value) / (max_value - min_value)

def prediction(Partner, Dependents, scaled_tenure, OnlineSecurity, TechSupport, PaperlessBilling, InternetService_Fiber_optic, InternetService_No, Contract_One_year, Contract_Two_year, PaymentMethod_Credit_card, PaymentMethod_Electronic_check):

    if Partner == 'Yes':
        Partner = 1  
    else:
        Partner = 0

    if Dependents == 'Yes':
        Dependents = 1
    else:
        Dependents = 0
    
    if OnlineSecurity == 'Yes':
        OnlineSecurity = 1
    else:
        OnlineSecurity = 0
    
    if TechSupport == 'Yes':
        TechSupport = 1
    else:
        TechSupport = 0
    
    if PaperlessBilling == 'Yes':
        PaperlessBilling = 1
    else:
        PaperlessBilling = 0

    if InternetService_Fiber_optic == 'Yes':
        InternetService_Fiber_optic = 1
    else:
        InternetService_Fiber_optic = 0

    if InternetService_No == 'Yes':
        InternetService_No = 0
    else:
        InternetService_No = 1

    if Contract_One_year == 'Yes':
        Contract_One_year = 1
    else:
        Contract_One_year = 0

    if Contract_Two_year == 'Yes':
        Contract_Two_year = 1
    else:
        Contract_Two_year = 0  

    if PaymentMethod_Credit_card == 'Yes':
        PaymentMethod_Credit_card = 1
    else:
        PaymentMethod_Credit_card = 0

    if PaymentMethod_Electronic_check == 'Yes':
        PaymentMethod_Electronic_check = 1
    else:
        PaymentMethod_Electronic_check = 0

    # Making predictions and getting probability
    prediction = churn_classifier.predict([[Partner, Dependents, scaled_tenure, OnlineSecurity, TechSupport, PaperlessBilling, InternetService_Fiber_optic, InternetService_No, Contract_One_year, Contract_Two_year, PaymentMethod_Credit_card, PaymentMethod_Electronic_check]])
    probability = churn_classifier.predict_proba([[Partner, Dependents, scaled_tenure, OnlineSecurity, TechSupport, PaperlessBilling, InternetService_Fiber_optic, InternetService_No, Contract_One_year, Contract_Two_year, PaymentMethod_Credit_card, PaymentMethod_Electronic_check]])

    if prediction == 0:
        pred = 'Not Churn'
        prob = probability[0][0] * 100  # Probability of Not Churn
    else:
        pred = 'Churn'
        prob = probability[0][1] * 100  # Probability of Churn

    return pred, prob

def main():       
    # front end elements of the web page 

    st.title('Telecommunication Customer Churn Prediction ML App')
    st.write('**Our model can accurately identify customers who are likely to churn.**')
    # st.write('---')

    df = pd.read_csv('selected_features.csv')
    df1 = pd.read_csv('telco-customer-churn.csv')

    st.write("""### Data Characteristics""")
    st.write("""* The dataset consists of 7043 observations and 12 features.""")
    st.write("""* The variables include both demographic and service-specific information about the customers.""")
    st.write("""* The target variable is "Churn" which is a binary variable indicating whether the customer has churned or not.""")
    st.write(df)

    st.write("""### Visualization""")

    chart_select = st.selectbox(
        label = 'Type of chart',
        options = ['Histogram','Boxplot','Countplot']
    )

    numeric_columns = ['tenure']
    cat_columns = ['Partner', 'Dependents', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'InternetService', 'Contract', 'PaymentMethod']
    
    if chart_select == 'Histogram':
        st.subheader('Histogram Settings')
        try:
            x_values = st.selectbox('X_axis',options=numeric_columns)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.histplot(data=df1, x=x_values, ax=ax,color='lightblue')
            st.pyplot(fig)
        except Exception as e:
            print(e)
    if chart_select == 'Boxplot':
        st.subheader('Boxplot Settings')
        try:
            x_values = st.selectbox('X_axis',options=numeric_columns)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(data=df1, x=x_values, ax=ax,color='lightblue')
            st.pyplot(fig)
        except Exception as e:
            print(e)
    if chart_select == 'Countplot':
        st.subheader('Countplot Settings')
        try:
            x_values = st.selectbox('X axis', options=cat_columns)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.countplot(data=df1, x=x_values, hue='Churn', ax=ax, palette='pastel')
            st.pyplot(fig)
        except Exception as e:
            st.write(e)

    st.sidebar.title('Prediction')
    st.sidebar.write("""### Specify Input Parameters""")
    # User inputs
    Partner = st.sidebar.selectbox('Partner', ("Yes", "No"))
    Dependents = st.sidebar.selectbox('Dependents', ("Yes", "No")) 
    tenure = st.sidebar.number_input("Number of months", min_value=min_tenure)
    OnlineSecurity = st.sidebar.selectbox('Online Security', ("Yes", "No")) 
    TechSupport = st.sidebar.selectbox('Tech Support', ("Yes", "No"))  
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ("Yes", "No")) 
    InternetService_Fiber_optic = st.sidebar.selectbox('Fiber Optic Internet Service', ("Yes", "No"))
    InternetService_No = st.sidebar.selectbox('Having Internet Service', ("Yes", "No"))  
    Contract_One_year = st.sidebar.selectbox("One Year Contract", ("Yes", "No"))
    Contract_Two_year = st.sidebar.selectbox("Two Year Contract", ("Yes", "No"))
    PaymentMethod_Credit_card = st.sidebar.selectbox("Payment Method is Credit card (automatic)", ("Yes", "No"))
    PaymentMethod_Electronic_check = st.sidebar.selectbox("Payment Method is Electronic Check", ("Yes", "No"))

    result = ""
    probability = 0

    data = {
        'Partner':Partner, 
        'Dependents':Dependents, 
        'Tenure':tenure, 
        'OnlineSecurity':OnlineSecurity, 
        'TechSupport':TechSupport, 
        'PaperlessBilling':PaperlessBilling, 
        'InternetService_Fiber_optic':InternetService_Fiber_optic, 
        'InternetService_No':InternetService_No, 
        'Contract_One_year':Contract_One_year, 
        'Contract_Two_year':Contract_Two_year, 
        'PaymentMethod_Credit_card':PaymentMethod_Credit_card, 
        'PaymentMethod_Electronic_check':PaymentMethod_Electronic_check
    }

    features = pd.DataFrame(data,index=[0])

    st.header('Specified Input Parameters')
    st.write(features)
    
    # When 'Predict' is clicked, make the prediction and store it 
    if st.sidebar.button("Predict"): 
        # Scale tenure before prediction
        scaled_tenure = scale_tenure(tenure, min_tenure, max_tenure)
        result, probability = prediction(Partner, Dependents, scaled_tenure, OnlineSecurity, TechSupport, PaperlessBilling, InternetService_Fiber_optic, InternetService_No, Contract_One_year, Contract_Two_year, PaymentMethod_Credit_card, PaymentMethod_Electronic_check)
        st.sidebar.success('Your customer is {} with a probability of {:.2f}%'.format(result, probability))
     
    
if __name__=='__main__': 
    main()


