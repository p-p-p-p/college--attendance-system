import streamlit as st
import os 
import csv
st.set_page_config(page_title = "This is a Multipage WebApp")
st.title("This is the Home Page Geeks.")
st.sidebar.success("Select Any Page from here")

st.sidebar.title("Navigation")
#check database.csv file is esist or not
if not os.path.exists("database.csv"):
    with open('database.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["face_data", "name","id_number","branch_name","designation"])
#check tmp folder exist or not
if not os.path.exists("temp"):
    os.mkdir("temp")


import os

if not os.path.exists("database.json"):
    with open("database.json", "w") as f:
        pass  # empty file

    
    


    