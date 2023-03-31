import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import os ,csv
if not os.path.exists("database.csv"):
    with open('database.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["face_data", "name","id_number","branch_name","designation"])
        
st.session_state.setdefault('visibility', True)
label_visibility = st.session_state.visibility

# Load the dataset
df = pd.read_csv('./database.csv')
store_df=df
df["face_data"] = df["face_data"].apply(eval)
df = df.sample(frac=1).reset_index(drop=True)

# Display the dataset
st.markdown("## This is the dataset of the registered faces")
st.dataframe(df.head())



# Extract the unique values in the 'name' column
unique_names = df['name'].unique()

# Create a new DataFrame that counts the number of occurrences of each unique name
name_counts = pd.DataFrame({'name': unique_names, 'count': df['name'].value_counts()})

# Create a histogram of the name counts
fig = go.Figure(data=[go.Bar(x=name_counts['name'], y=name_counts['count'])])
fig.update_layout(title='Distribution of Unique Names', xaxis_title='Name', yaxis_title='Count')
st.plotly_chart(fig)

# Display an overview of the dataset
st.markdown("## Overview of the dataset")
grouped_data = df.groupby(['id_number', 'name']).count()
st.dataframe(grouped_data)


#give a choice you want to delete or not
your_choice = st.selectbox("Clean Data", ("No","Yes"))
if your_choice=="Yes":
    id_numbers = df['id_number'].unique()
    selected_id = st.text_input("Enter your id_number")
    if selected_id in id_numbers:
        df = df.drop(df[df['id_number'] == selected_id].index)
        st.success("Rows with id_number {} have been deleted.".format(selected_id))
        grouped_data = df.groupby(['id_number', 'name']).count()
        st.dataframe(grouped_data)    
else:
    st.warning("No rows have been deleted.")




from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import pickle

def run_model():
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df["face_data"], df["id_number"], test_size=0.2, random_state=42)

    # Create the SVM model
    clf = svm.SVC(kernel='linear')

    # Train the model on the training set
    clf.fit(list(X_train), list(y_train))

    # Test the model on the testing set
    y_pred = clf.predict(list(X_test))

    # Calculate the accuracy of the model
    acc = accuracy_score(list(y_test), y_pred)
    #display accuracy
    st.write("Accuracy: ",acc)
    pickle.dump(clf, open('svm_model.pkl', 'wb'))
    st.caption('Model saved as :blue[model.pkl] file')
if st.button('Run Model'):
    run_model()

    
