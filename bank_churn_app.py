import streamlit as st
import altair as alt
import base64
import json
import pickle
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn import set_config

set_config(transform_output="pandas") # Transform outputs of pipeline steps to pandas DataFrames

# Set the css styles from the local file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Redefining the custom transformer from the ML pipeline for the app
# to ensure that the pipeline can be executed
from sklearn.base import BaseEstimator, TransformerMixin

class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.column_names)

# Set the correct directory to the directory of the script
script_directory = os.path.dirname(__file__)
os.chdir(script_directory)

# Load data
@st.cache_data
def load_data():
    train_data = pd.read_csv('training_data.csv')
    test_data = pd.read_csv('test_data.csv')
    data = pd.concat([train_data, test_data])
    # Dropping unnecessary columns
    data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
               'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1, inplace=True)
    return data

df = load_data()

# Load model
@st.cache_resource
def load_model():
    return joblib.load('trained_model.joblib')

model = load_model()

# Load variable explanation data
@st.cache_data
def load_variable_explanation():
    return pd.read_csv('Variable_explanation.csv')

variable_explanation_df = load_variable_explanation()

# Define the streamlit sidebar
st.sidebar.title("Credit Card Churn Analysis")
st.sidebar.image("image_sidebar.png")
st.sidebar.title("Global Filters")

# Define the global filters
customer_age = st.sidebar.slider("Select Customer Age", 
                                 int(df['Customer_Age'].min()), 
                                 int(df['Customer_Age'].max()), 
                                 (int(df['Customer_Age'].min()), 
                                  int(df['Customer_Age'].max())))
gender_options = ['All'] + df['Gender'].unique().tolist()
gender_selection = st.sidebar.selectbox("Select Gender", options=gender_options)
churn_status_options = ['All'] + df['Attrition_Flag'].unique().tolist()
churn_status_selection = st.sidebar.selectbox("Select Churn Status", 
                                              options=churn_status_options)

# Apply global filters to the DataFrame 
filtered_df = df.loc[df['Customer_Age'].between(customer_age[0], customer_age[1])]
if gender_selection != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == gender_selection]
if churn_status_selection != 'All':
    filtered_df = filtered_df[filtered_df['Attrition_Flag'] == churn_status_selection]

# Set banner image for the app
st.image("image_banner.png", use_column_width=True)

# Define main panel with all tabs
tabs = ["Dataset Description", 
        "Raw Data", 
        "Demographic Visualizations", 
        "Financial Visualizations", 
        "ML Model Performance", 
        "ML Model Predictions", 
        "Interactive Churn Analysis Tool"]
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs)

# Define the content of the dataset description tab
with tab0:
    st.title("📚 Dataset Description")
    st.write("""
    This dataset contains information about credit card customers of a bank, including their demographics, 
    financial behaviors, and attrition status. The goal is to predict customer churn and understand the factors 
    that contribute to it. By analyzing this dataset, we can gain insights into customer behavior and 
    improve retention strategies.
    """)
    st.write("### Variable Explanation")
    st.dataframe(variable_explanation_df)

# Define the content of the raw data tab
with tab1:
    st.title("📄 Raw Data")
    raw_data_with_clientnum_as_string = filtered_df.copy()
    # Convert CLIENTNUM to string for better display
    raw_data_with_clientnum_as_string['CLIENTNUM'] = raw_data_with_clientnum_as_string['CLIENTNUM'].astype(str)
    st.dataframe(raw_data_with_clientnum_as_string)

    # Define the selection box for the data format
    selected_format = st.selectbox("Select data format for download", 
                                   ["CSV", "JSON", "PKL"])

    # Create download link based on selected format
    if st.button("Download Raw Data"):
        if selected_format == "CSV":
            csv_data = raw_data_with_clientnum_as_string.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="raw_data.csv">Download CSV</a>'
        elif selected_format == "JSON":
            json_data = raw_data_with_clientnum_as_string.to_dict(orient="records")
            json_str = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
            b64 = base64.b64encode(json_str).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="raw_data.json">Download JSON</a>'

        elif selected_format == "PKL":
            pkl_data = pickle.dumps(raw_data_with_clientnum_as_string)
            b64 = base64.b64encode(pkl_data).decode()
            href = f'<a href="data:file/pkl;base64,{b64}" download="raw_data.pkl">Download PKL</a>'
        
        # Display the download link
        st.markdown(href, unsafe_allow_html=True)

# Define the content of the demographic visualizations tab
with tab2:
    st.title("📊 Demographic Visualizations")
    
    # Define demographic Filters
    st.write("### Demographic Filters")
    education_categories = st.multiselect("Select Education Levels", 
                                          options=filtered_df['Education_Level'].unique())
    dependent_count_range = st.slider("Select Dependent Count Range", 
                                      min_value=int(filtered_df['Dependent_count'].min()), 
                                      max_value=int(filtered_df['Dependent_count'].max()), 
                                      value=(int(filtered_df['Dependent_count'].min()), 
                                             int(filtered_df['Dependent_count'].max())))

    # Filter DataFrame based on selected filters
    filtered_demographic_df = filtered_df[(filtered_df['Education_Level'].isin(education_categories) if education_categories else filtered_df['Education_Level'].notnull()) &
                                          (filtered_df['Dependent_count'].between(dependent_count_range[0], dependent_count_range[1]))]

    # Display visualization "Age Distribution" using seaborn and matplotlib
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_demographic_df['Customer_Age'], bins=10, kde=False, ax=ax)
    st.pyplot(fig)

    # Display visualization "Gender Distribution" using seaborn and matplotlib
    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_demographic_df, x='Gender', ax=ax)
    st.pyplot(fig)

    # Display visualization "Churn by Age Group" using seaborn and matplotlib
    st.write("### Churn by Age Group")
    # Reset index to prevent duplicate labels
    filtered_demographic_df.reset_index(drop=True, inplace=True)
    
    fig, ax = plt.subplots()
    # Group the ages into intervals of 5 years
    age_bins = pd.cut(filtered_demographic_df['Customer_Age'], 
                      bins=range(int(filtered_demographic_df['Customer_Age'].min()), 
                                 int(filtered_demographic_df['Customer_Age'].max()) + 1, 5), 
                                 right=False)
    sns.countplot(x=age_bins, 
                  hue='Attrition_Flag', 
                  data=filtered_demographic_df, 
                  ax=ax)
    # Rotate x-axis labels for better visibility
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Define the Bi-variate Analysis plot
    st.title("Bi-variate Analysis")

    # Display disclaimer about the filters
    st.write("### Disclaimer")
    st.write("Global and demographic filters are not applied in this area.")
    
    # Provide the selection boxes for the demographic variables
    st.write("### Select Demographic Variables")
    variable1 = st.selectbox("Select First Variable", 
                             options=['Gender', 
                                      'Education_Level', 
                                      'Marital_Status', 
                                      'Income_Category', 
                                      'Attrition_Flag'])
    variable2 = st.selectbox("Select Second Variable", 
                             options=['Gender', 
                                      'Education_Level', 
                                      'Marital_Status', 
                                      'Income_Category', 
                                      'Attrition_Flag'])

    # Check if both variables are the same
    if variable1 == variable2:
        # Display an error message if both variables are the same
        st.error("Please select two different variables.")
    else:
        # Replace missing values with a placeholder
        df_filled = df[[variable1, variable2]].fillna("Missing")

        # Generate contingency table
        contingency_table = pd.crosstab(df_filled[variable1], df_filled[variable2])

        # Display heatmap of contingency table using seaborn and matplotlib
        st.write("### Contingency Table with Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(contingency_table, 
                    annot=True, 
                    cmap="YlGnBu", 
                    fmt="d", 
                    cbar=False, 
                    ax=ax, 
                    annot_kws={"size": 8})  # Adjust font size here to fit the size of annotations in the cells
        st.pyplot(fig)

# Define the content of the financial visualizations tab
with tab3:
    st.title("💵 Financial Visualizations")

    # Define Financial Filters
    st.write("### Financial Filters")
    card_categories = st.multiselect("Select Card Categories", 
                                     options=filtered_df['Card_Category'].unique())
    months_inactive_range = st.slider("Select Months Inactive During Last Year", 
                                      min_value=int(filtered_df['Months_Inactive_12_mon'].min()), 
                                      max_value=12, 
                                      value=(int(filtered_df['Months_Inactive_12_mon'].min()), 12))
    avg_utilization_ratio_range = st.slider("Select Average Utilization Ratio Range", 
                                            min_value=float(filtered_df['Avg_Utilization_Ratio'].min()), 
                                            max_value=float(filtered_df['Avg_Utilization_Ratio'].max()), 
                                            value=(float(filtered_df['Avg_Utilization_Ratio'].min()), 
                                                   float(filtered_df['Avg_Utilization_Ratio'].max())))

    # Filter the DataFrame based on selected filters
    filtered_financial_df = filtered_df[(filtered_df['Card_Category'].isin(card_categories) if card_categories else filtered_df['Card_Category'].notnull()) &
                                        (filtered_df['Months_Inactive_12_mon'].between(months_inactive_range[0], months_inactive_range[1])) &
                                        (filtered_df['Avg_Utilization_Ratio'].between(avg_utilization_ratio_range[0], avg_utilization_ratio_range[1]))]

    # Display "Churn Rate by Card Category (Bar Chart)" using seaborn and matplotlib
    st.write("### Churn Rate by Card Category")
    churn_rate_by_card_category = filtered_financial_df.groupby('Card_Category')['Attrition_Flag'].apply(lambda x: (x == 'Attrited Customer').mean()).reset_index()
    churn_rate_by_card_category.columns = ['Card Category', 'Churn Rate']
    fig, ax = plt.subplots()
    sns.barplot(x='Card Category', 
                y='Churn Rate', 
                data=churn_rate_by_card_category, 
                ax=ax)
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Card Category")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)  # Adjust fontsize here to fit the bar
    st.pyplot(fig)

    # Display "Average Credit Limit by Income Category and Churn Status"
    st.write("### Average Credit Limit by Income Category and Churn Status")
    
    # Define the order of income categories which is different from the default alphabetical order
    income_category_order = ['Less than $40K', 
                             '$40K - $60K', 
                             '$60K - $80K', 
                             '$80K - $120K', 
                             '$120K +']
    avg_credit_limit_by_income = filtered_financial_df.groupby(['Income_Category', 
                                                                'Attrition_Flag'])['Credit_Limit'].mean().reset_index()
    avg_credit_limit_by_income.columns = ['Income Category', 
                                          'Churn Status', 
                                          'Average Credit Limit']
    avg_credit_limit_by_income['Income Category'] = pd.Categorical(avg_credit_limit_by_income['Income Category'], 
                                                                   categories=income_category_order, 
                                                                   ordered=True)
    fig, ax = plt.subplots()
    sns.barplot(x='Income Category', 
                y='Average Credit Limit', 
                hue='Churn Status', 
                data=avg_credit_limit_by_income, 
                ax=ax)
    ax.set_ylabel("Average Credit Limit")
    ax.set_xlabel("Income Category")
    for container in ax.containers:
        ax.bar_label(container, fontsize=6)  # Adjust fontsize here to fit the bar
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Display "Total Transaction Amount by Months on Book and Churn Status"
    st.write("### Total Transaction Amount by Months on Book and Churn Status")
    total_transaction_amount_by_months = filtered_financial_df.groupby(['Months_on_book', 
                                                                        'Attrition_Flag'])['Total_Trans_Amt'].sum().reset_index()
    total_transaction_amount_by_months.columns = ['Months on Book', 
                                                  'Churn Status', 
                                                  'Total Transaction Amount']
    fig, ax = plt.subplots()
    sns.lineplot(x='Months on Book', 
                 y='Total Transaction Amount', 
                 hue='Churn Status', 
                 data=total_transaction_amount_by_months, 
                 ax=ax)
    ax.set_ylabel("Total Transaction Amount (in millions)")
    ax.set_xlabel("Months on Book")
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x/1e6))))
    st.pyplot(fig)

    # Define the Bi-variate Analysis for Financial Variables
    st.title("Bi-variate Analysis for Financial Variables")

    # Display disclaimer about the filters
    st.write("### Disclaimer")
    st.write("Global and demographic filters are not applied in this area.")

    # Mapping of display names to contracted names used as dataframe columns
    variable_mapping = {
        'Credit Limit': 'Credit_Limit',
        'Total Revolving Balance': 'Total_Revolving_Bal',
        'Average Open to Buy': 'Avg_Open_To_Buy',
        'Total Amount Change Q4-Q1': 'Total_Amt_Chng_Q4_Q1',
        'Total Transaction Amount': 'Total_Trans_Amt',
        'Total Transaction Count': 'Total_Trans_Ct',
        'Total Count Change Q4-Q1': 'Total_Ct_Chng_Q4_Q1',
        'Average Utilization Ratio': 'Avg_Utilization_Ratio'
    }

    # Define the checkboxes for the financial variables with the mapped names
    variable1_display = st.selectbox("Select First Financial Variable", 
                                     options=list(variable_mapping.keys()))
    variable2_display = st.selectbox("Select Second Financial Variable", 
                                     options=list(variable_mapping.keys()))

    # Get corresponding dataframe columns using the mapping
    variable1 = variable_mapping[variable1_display]
    variable2 = variable_mapping[variable2_display]

    # Check if both variables are the same
    if variable1 == variable2:
        # Display an error message if both variables are the same
        st.error("Please select two different variables.")
    else:
        # Plot Scatterplot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, 
                        x=variable1, 
                        y=variable2, 
                        hue='Attrition_Flag', 
                        palette='viridis', 
                        ax=ax)
        ax.set_xlabel(variable1_display)
        ax.set_ylabel(variable2_display)
        ax.set_title(f"Scatterplot of {variable1_display} vs {variable2_display} (Colored by Attrition Status)")
        ax.legend(title='Attrition Flag')
        st.pyplot(fig)

        # Calculate and Display Correlation Coefficient
        correlation_coefficient = df[[variable1, variable2]].corr().iloc[0, 1]
        st.write(f"### Correlation Coefficient: {correlation_coefficient:.2f}")

# Define the content of the ML model performance tab
with tab4:
    st.title("🚀 ML Model Performance")

    # Display disclaimer about the filters
    st.write("### Disclaimer")
    st.write("The global filters do not apply to this tab.")

    # Load the test data
    test_data = pd.read_csv('test_data.csv')

    # Extract features and target variable
    X_test = test_data.drop(columns=['Attrition_Flag'])
    y_test = test_data['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

    # Predict probabilities using the loaded model
    y_prob = model.predict_proba(X_test)[:, 1]

    # Define the slider for user to input the probability threshold
    st.write("### Model Performance Metrics")
    threshold = st.slider("Select Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Calculate predicted labels based on the threshold provided by the user
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate performance metrics based on the selected threshold
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display the performance metrics in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"🎯 **Accuracy:** {accuracy:.2f}")
        st.write(f"🔬 **Precision:** {precision:.2f}")

    with col2:
        st.write(f"🔍 **Recall:** {recall:.2f}")
        st.write(f"📈 **F1 Score:** {f1:.2f}")

    # Calculate ROC (Receiver Operating Characteristic) curve and AUC (Area Under the Curve) 
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')

    # Highlight the point on the ROC curve corresponding to the selected threshold
    closest_threshold_index = np.argmin(np.abs(thresholds - threshold))
    ax.plot(fpr[closest_threshold_index], tpr[closest_threshold_index], 'ro')
    ax.legend(loc="lower right")

    st.pyplot(fig)

    # Define Confusion Matrix
    # Convert y_test and y_pred to pandas Series
    y_test_series = pd.Series(y_test)
    y_pred_series = pd.Series(y_pred)

    # Replace 0 with 'Stay' and 1 with 'Churn' due to the nature of the target variable
    y_test_labels = y_test_series.replace({0: 'Stay', 1: 'Churn'})
    y_pred_labels = y_pred_series.replace({0: 'Stay', 1: 'Churn'})

    # Create confusion matrix using crosstab function from pandas
    confusion_matrix = pd.crosstab(y_test_labels, y_pred_labels, rownames=['Actual'], colnames=['Predicted'])

    # Reset index and stack the dataframe to make it suitable for Altair
    # We use Altair to create a heatmap for the confusion matrix, as it is more interactive
    # Moreover the default display of confusion matrix was not working as expected
    confusion_matrix = confusion_matrix.reset_index().melt(id_vars='Actual')

    # Create the Altair heatmap
    heatmap = alt.Chart(confusion_matrix).mark_rect().encode(
        x=alt.X('Actual:O', title='Actual'),
        y=alt.Y('Predicted:O', title='Predicted'),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='lightgreyred'), title='Count'),
        tooltip=['Actual', 'Predicted', 'value']
    ).properties(
        title='Confusion Matrix Heatmap',
        width=500,  
        height=400  
    )

    # Add text annotations for count values
    text = heatmap.mark_text(baseline='middle').encode(
        text='value:Q',
        color=alt.condition(
            alt.datum.value > confusion_matrix['value'].max() / 2,
            alt.value('white'),
            alt.value('black')
        )
    )

    # Combine heatmap and text annotations
    confusion_matrix_chart = heatmap + text

    # Display the heatmap with text annotations
    st.altair_chart(confusion_matrix_chart)

    # Define the Feature Importance
    st.write("### Feature Importance")
    
    # First, extract the XGBoost model from the pipeline
    xgb_model = model.steps[-1][1]

    # Second, fet feature importances from that model
    feature_importances = xgb_model.feature_importances_

    # Third, get feature names from the same model
    feature_names = xgb_model.get_booster().feature_names

    # And create a dictionary to store feature names and importances
    feature_dict = dict(zip(feature_names, feature_importances))

    # Then, modify feature names to display only the part after double underscore
    # Otherwise, we would see the names thats the model uses internally, for example,
    # inner_pipe_num__Trans_Amt instead of Trans_Amt
    modified_feature_names = [name.split('__')[-1] for name in feature_names]

    # Zip modified feature names with importances
    sorted_features = sorted(zip(modified_feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    # Create a toggle button to choose between displaying top 5 and all features
    display_top5 = st.toggle("Display Top 5 Features", value=False)

    # Display the selected features (top 5 or all features)
    if display_top5:
        features_to_display = sorted_features[:5]
    else:
        features_to_display = sorted_features

    # Reverse the order of features (we need to display the most important feature first)
    features_to_display.reverse()

    # Plot the feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([x[0] for x in features_to_display], [x[1] for x in features_to_display])
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('Feature Name')
    
    # Set the title of the plot based on the selected number of features to be displayed
    if display_top5:
        ax.set_title('Top 5 Most Important Features')
    else:
        ax.set_title('All Features')
    st.pyplot(fig)

# Define the content of the ML model predictions tab
with tab5:
    st.title("🔮 ML Model Predictions")

    # Define selection boxes for categorical features used in the model
    gender_options = df['Gender'].unique().tolist()
    education_options = df['Education_Level'].unique().tolist()
    marital_options = df['Marital_Status'].unique().tolist()
    income_options = df['Income_Category'].unique().tolist()
    card_options = df['Card_Category'].unique().tolist()


    gender = st.selectbox('Gender', gender_options)
    education = st.selectbox('Education Level', education_options)
    marital = st.selectbox('Marital Status', marital_options)
    income = st.selectbox('Income Category', income_options)
    card = st.selectbox('Card Category', card_options)

    # Set clientnum as constant, as the model was trained using this feature
    # However, the clientnum is not important for the prediction, as it is essentially random
    clientnum = '111111111'

    # Define input sliders for numerical features
    customer_age = st.slider("Customer Age", int(df['Customer_Age'].min()), int(df['Customer_Age'].max()), int(df['Customer_Age'].mean()))
    dependent_count = st.slider("Dependent Count", int(df['Dependent_count'].min()), int(df['Dependent_count'].max()), int(df['Dependent_count'].mean()))
    months_on_book = st.slider("Months on Book", int(df['Months_on_book'].min()), int(df['Months_on_book'].max()), int(df['Months_on_book'].mean()))
    total_relationship_count = st.slider("Total Relationship Count", int(df['Total_Relationship_Count'].min()), int(df['Total_Relationship_Count'].max()), int(df['Total_Relationship_Count'].mean()))
    months_inactive_12_mon = st.slider("Months Inactive Last 12 Months", int(df['Months_Inactive_12_mon'].min()), 12, int(df['Months_Inactive_12_mon'].mean()))
    contacts_count_12_mon = st.slider("Contacts Count Last 12 Months", int(df['Contacts_Count_12_mon'].min()), int(df['Contacts_Count_12_mon'].max()), int(df['Contacts_Count_12_mon'].mean()))
    credit_limit = st.slider("Credit Limit", float(df['Credit_Limit'].min()), float(df['Credit_Limit'].max()), float(df['Credit_Limit'].mean()))
    total_revolving_bal = st.slider("Total Revolving Balance", int(df['Total_Revolving_Bal'].min()), int(df['Total_Revolving_Bal'].max()), int(df['Total_Revolving_Bal'].mean()))
    avg_open_to_buy = st.slider("Average Open to Buy", float(df['Avg_Open_To_Buy'].min()), float(df['Avg_Open_To_Buy'].max()), float(df['Avg_Open_To_Buy'].mean()))
    total_amt_chng_q4_q1 = st.slider("Total Amount Change Q4-Q1", float(df['Total_Amt_Chng_Q4_Q1'].min()), float(df['Total_Amt_Chng_Q4_Q1'].max()), float(df['Total_Amt_Chng_Q4_Q1'].mean()))
    total_trans_amt = st.slider("Total Transaction Amount", float(df['Total_Trans_Amt'].min()), float(df['Total_Trans_Amt'].max()), float(df['Total_Trans_Amt'].mean()))
    total_trans_ct = st.slider("Total Transaction Count", int(df['Total_Trans_Ct'].min()), int(df['Total_Trans_Ct'].max()), int(df['Total_Trans_Ct'].mean()))
    total_ct_chng_q4_q1 = st.slider("Total Count Change Q4-Q1", float(df['Total_Ct_Chng_Q4_Q1'].min()), float(df['Total_Ct_Chng_Q4_Q1'].max()), float(df['Total_Ct_Chng_Q4_Q1'].mean()))
    avg_utilization_ratio = st.slider("Average Utilization Ratio", float(df['Avg_Utilization_Ratio'].min()), float(df['Avg_Utilization_Ratio'].max()), float(df['Avg_Utilization_Ratio'].mean()))

    # Include the button to trigger prediction
    if st.button('Predict'):
        # Prepare the input features as a DataFrame
        input_data = pd.DataFrame({
            'CLIENTNUM': [clientnum],
            'Gender': [gender],
            'Education_Level': [education],
            'Marital_Status': [marital],
            'Income_Category': [income],
            'Card_Category': [card],
            'Customer_Age': [customer_age],
            'Dependent_count': [dependent_count],
            'Months_on_book': [months_on_book],
            'Total_Relationship_Count': [total_relationship_count],
            'Months_Inactive_12_mon': [months_inactive_12_mon],
            'Contacts_Count_12_mon': [contacts_count_12_mon],
            'Credit_Limit': [credit_limit],
            'Total_Revolving_Bal': [total_revolving_bal],
            'Avg_Open_To_Buy': [avg_open_to_buy],
            'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
            'Total_Trans_Amt': [total_trans_amt],
            'Total_Trans_Ct': [total_trans_ct],
            'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
            'Avg_Utilization_Ratio': [avg_utilization_ratio]
        })
    
        # Make prediction using the loaded model and the threshold of 0.5
        y_prob = model.predict_proba(input_data)[:, 1]
        prediction = (y_prob >= 0.5).astype(int)

        # Display predicted class
        if prediction[0] == 0:
            st.markdown("**Prediction:** Not Churned ✅")
        else:
            st.markdown("**Prediction:** Churned ❌")

        # Display predicted probability
        st.write(f"**Probability:** {y_prob[0]:.2f}")
        st.write("The predicted status is based on the threshold of 0.5.")

# Define the content of the interactive churn analysis tool tab   
with tab6:
    # Function to compute both churn probability and status simultaneously
    def get_churn_prediction(input_data):
        probability = model.predict_proba(input_data)[:, 1][0]
        status = "Churn" if probability > 0.5 else "Stay"
        return probability, status
    st.title("🔄 Interactive Churn Analysis Tool")
    
    # Display disclaimer about the filters
    st.write("### Disclaimer")
    st.write("The global filters do not apply to this tab.")

    # Define customer selector using a dropdown for CLIENTNUM
    customer_id = st.selectbox("Select Customer ID", df['CLIENTNUM'].unique())
    customer_data = df[df['CLIENTNUM'] == customer_id].iloc[0]

    # Display actual churn status
    actual_status = customer_data['Attrition_Flag']
    st.write(f"**Actual Churn Status:** {actual_status}")

    # Compute initial churn probability
    initial_input = pd.DataFrame(customer_data.drop('Attrition_Flag')).T
    initial_probability, _ = get_churn_prediction(initial_input)
    st.write(f"**Initial Churn Probability:** {initial_probability:.2f}")

    # Define feature adjusters. Only main features according to importance are included for simplicity
    age = st.slider("Customer Age", 18, 100, int(customer_data['Customer_Age']))
    revolving_balance = st.slider("Total Revolving Balance", 0, int(df['Total_Revolving_Bal'].max()), int(customer_data['Total_Revolving_Bal']))
    trans_amt = st.slider("Total Transaction Amount", 0, int(df['Total_Trans_Amt'].max()), int(customer_data['Total_Trans_Amt']))
    amt_change = st.slider("Total Amount Change (Q4-Q1)", 0.0, df['Total_Amt_Chng_Q4_Q1'].max(), float(customer_data['Total_Amt_Chng_Q4_Q1']))
    trans_ct = st.slider("Total Transaction Count", 0, int(df['Total_Trans_Ct'].max()), int(customer_data['Total_Trans_Ct']))
    ct_change = st.slider("Total Count Change (Q4-Q1)", 0.0, df['Total_Ct_Chng_Q4_Q1'].max(), float(customer_data['Total_Ct_Chng_Q4_Q1']))

    # Create a copy of customer data and update with slider values
    adjusted_data = customer_data.copy()
    adjusted_data['Customer_Age'] = age
    adjusted_data['Total_Revolving_Bal'] = revolving_balance
    adjusted_data['Total_Trans_Amt'] = trans_amt
    adjusted_data['Total_Amt_Chng_Q4_Q1'] = amt_change
    adjusted_data['Total_Trans_Ct'] = trans_ct
    adjusted_data['Total_Ct_Chng_Q4_Q1'] = ct_change

    # Compute adjusted churn probability
    adjusted_input = pd.DataFrame(adjusted_data.drop('Attrition_Flag')).T
    adjusted_probability, adjusted_status = get_churn_prediction(adjusted_input)

    # Display predicted churn status and probability
    st.write(f"**Predicted Churn Status based on adjustments:** {adjusted_status}")
    st.write(f"**Predicted Churn Probability based on adjustments:** {adjusted_probability:.2f}")

    # Compute and display probability delta
    probability_delta = adjusted_probability - initial_probability
    st.write(f"**Delta of the Probability:** {probability_delta:.2f}")

    # Display the highlighted message based on probability change
    if probability_delta > 0:
        st.markdown("<div class='increase'>Such change would <strong>INCREASE</strong> the probability of the churn for this client</div>", unsafe_allow_html=True)
    elif probability_delta < 0:
        st.markdown("<div class='decrease'>Such change would <strong>DECREASE</strong> the probability of the churn for this client</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='same'>The probability of the churn would remain the same</div>", unsafe_allow_html=True)
