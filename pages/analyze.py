import streamlit as st
import json
import pandas as pd
import plotly.express as px

# Load attendance data from JSON file
with open('attandance.json') as f:
    data = json.load(f)

# Convert attendance data to Pandas DataFrame
rows = []
for date, courses in data.items():
    for course, attendees in courses.items():
        for id, attendee in attendees.items():
            time = pd.to_datetime(attendee['time'], format='%I:%M:%S %p').strftime('%H:%M:%S')
            rows.append([date, course, id, time, attendee['name'], attendee['branch_name'], attendee['designation']])
df = pd.DataFrame(rows, columns=['Date', 'Course', 'ID', 'Time', 'Name', 'Branch', 'Designation'])

# Create bar chart of attendance data by date and course using Plotly
group_df = df.groupby(['Date', 'Course']).size().reset_index(name='Count')
fig1 = px.bar(group_df, x='Date', y='Count', color='Course', barmode='group', title='Attendance Count by Date and Course')
st.plotly_chart(fig1)

# Create stacked bar chart of attendance data by student and course using Plotly
group_df2 = df.groupby(['Name', 'Course']).size().reset_index(name='Count')
fig2 = px.bar(group_df2, x='Name', y='Count', color='Course', title='Attendance Count by Student and Course', barmode='stack')
st.plotly_chart(fig2)
# Create line chart of attendance data over time using Plotly
group_df3 = df.groupby(['Date']).size().reset_index(name='Count')
fig3 = px.line(group_df3, x='Date', y='Count', title='Attendance Count over Time')
st.plotly_chart(fig3)

# Create histogram of attendance data by time of day using Plotly
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
fig4 = px.histogram(df, x='Hour', nbins=24, title='Attendance by Time of Day')
st.plotly_chart(fig4)

# Create bar chart of attendance data by branch using Plotly
group_df5 = df.groupby(['Branch']).size().reset_index(name='Count')
fig5 = px.bar(group_df5, x='Branch', y='Count', title='Attendance by Branch')
st.plotly_chart(fig5)
