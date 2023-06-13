import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns
import hydralit_components as hc 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset into a DataFrame
df = pd.read_csv('heart-good.csv')  # Replace 'your_dataset.csv' with your actual dataset file path
df_long= pd.read_csv('New_clean.csv')
df2= pd.read_csv('map2_hosp.csv')

############################################################################################################################

#Creating Navigation bar
menu_data = [{"label":"Overview"},{'label':'Risk Factors'},{'label':'Are you at Risk?'},{'label':'Recommendations'}]
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky')

if menu_id =="Overview":
  # st.image("https://i.pinimg.com/originals/47/e5/8f/47e58ff1ac31eeabb21afa856d4576dc.gif", width=300)
  st.header("Cardiovascular Disease: A global problem")
  st.subheader("Global Overview")
  col1,col2 = st.columns([6,3])

  # Create the choropleth map
  fig = px.choropleth(df_long,  # DataFrame containing the data in long format
                    locations='Country Code',  # Column specifying the country codes
                    color='Value',  # Column specifying the values for coloring
                    animation_frame='Year',
                    hover_name='Country Name',   # Column specifying the country names for hover labels
                    labels={'Value': 'Mortality'},  # Labels for the colorbar
                    range_color=[df_long['Value'].min(), df_long['Value'].max()],  # Set the color rang
                    color_continuous_scale=px.colors.sequential.PuBu,       
                    )

  # Update the layout of the map
  fig.update_layout(geo=dict(showframe=False,  # Hide the frame around the map
                           showcoastlines=False,  # Hide the coastlines
                           projection_type='equirectangular',  # Use equirectangular projection
                           ), height=500, width=1500, margin=dict(l=0, r=0, t=20, b=0),
                           title='Mortality from CVD by Country'
                           )

  # Add animation settings
  fig.update_traces(hovertemplate='Mortality: %{z:.2f}',  # Template for the hover labels
                  selector=dict(type='choropleth'))  # Select the choropleth trace for animation

  fig.update_geos(
    showcountries=True, countrycolor="darkgray", showcoastlines=True, coastlinecolor="darkgray"
  )

  # Display the map in Streamlit
  map_chart= st.plotly_chart(fig)

  # Create a selection box to choose a country
  selected_country = st.selectbox('Select a Country', df2['Country Name'])

  # Filter the data for the selected country
  filtered_data2 = df2[df2['Country Name'] == selected_country]
  # Get the relevant information for the selected country
  er_beds = round(filtered_data2['Average'].values[0] * 10)
  population = round(filtered_data2['Population'].values[0])
  death_rate = round(filtered_data2['Death of non-comm'].values[0])
# Set the style for the boxes
  box_style = '''
    div.stButton > button:first-child {
        background-color: white;
        color: black;
        font-size: 50px;
    }
  '''
# Apply the style to Streamlit
  st.markdown(f'<style>{box_style}</style>', unsafe_allow_html=True)
  # Display the information in three boxes
  col0, col1, col2 = st.columns(3)
  with col0:
     st.image('https://i.pinimg.com/564x/8a/b6/56/8ab656faa2989706ecfdd926c030e757.jpg')
  with col1:
    st.markdown(f'<p style="font-size:50px;"><b>{er_beds}</b></p>', unsafe_allow_html=True)
    st.write('The Number of Er Beds per 1000')

    st.markdown(f'<p style="font-size:50px;"><b>{population}</b></p>', unsafe_allow_html=True)
    st.write('The Population under consideration')

    st.markdown(f'<p style="font-size:50px;"><b>{death_rate}%</b></p>', unsafe_allow_html=True)
    st.write('% Of Non-Communicable Deaths')
  #ER vs. Mortality:
  er= df2['Average']
  mort= df2['mortality']
  countries = df2['Country Name']

# Create a categorical colormap with a unique color for each country
  unique_countries = np.unique(countries)
  num_countries = len(unique_countries)
  cmap = plt.get_cmap('tab20')

# Plot the scatter plot with color-coded dots for each country
  fig, ax = plt.subplots(figsize=(5, 3))
  for i, country in enumerate(unique_countries):
    mask = (countries == country)
    ax.scatter(er[mask], mort[mask], s=10, color=cmap(i % num_countries))

# Set the chart title and axis labels
  ax.set_title('ER Beds vs Mortality', fontsize=15)
  ax.set_xlabel('Number of ER Beds', fontsize=15)
  ax.set_ylabel('Mortality', fontsize=10)
  # ax.set_frame_on(False)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(True)
  ax.spines['left'].set_visible(True)
# Adjust the font size of the x and y-axis tick labels
  ax.tick_params(axis='both', which='both', labelsize=8)

# Display the scatter plot using Streamlit
  with col2:
    st.pyplot(fig)

  st.subheader("Demographic Overview")
  # Group the filtered data by race and calculate the count of heart disease cases
  gender = st.multiselect('Sex', ['Male', 'Female'])
  filtered_df = df[df['Sex'].isin(gender)]
  grouped_df = filtered_df[filtered_df['HeartDisease'] == 'Yes'].groupby('Race').size().reset_index(name='Count')
  grouped_df2 = filtered_df.groupby('HeartDisease').size().reset_index(name='Count')

    # Plot the pie chart
  fig_pie = go.Figure(data=go.Pie(
    labels=grouped_df['Race'],
    values=grouped_df['Count'],
    hole=0.5,
    textinfo='percent',
    insidetextorientation='radial',
  ))
  fig_pie.update_traces(marker_colors=['Blue', 'Red'])

    # Plot the bar chart
  fig_bar = px.bar(grouped_df2, x='HeartDisease', y='Count',
                 labels={'Count': 'Count of Cases'},
                 title='Heart Disease Cases by Sex',
                 color='HeartDisease',
                 color_discrete_map={'Yes': 'red', 'No': 'blue'})

# Set the x-axis and y-axis labels
  fig_bar.update_xaxes(title='Heart Disease')
  fig_bar.update_yaxes(title='Count')
  fig_bar.update_traces(opacity= 0.6)
    # Show the charts side by side using Streamlit
  col1, col2 = st.columns(2)
  with col1:
        st.plotly_chart(fig_pie)

  with col2:
        st.plotly_chart(fig_bar)

###################################################################################################################################
if menu_id == "Risk Factors":
    # col1, col2, col3= st.columns(3)
    # with col1:
    #   st.header("Health")
    #   st.write("- Obesity")
    #   st.write("- Physical Activity")
    #   st.write("- Mental Health")
    #   st.write("- Smoking")
    # with col2:
    #    st.image("https://i.pinimg.com/564x/0f/89/4e/0f894e134532d46fff1bb094a183b97f.jpg", width=700)
    # with col3:
    #   st.header("Diseases")
    #   st.write("- Lungs")
    #   st.write("- Kidney")
    #   st.write("- Skin")
    #   st.write("- Diabetes")
    disease = ['Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer']
    col1, col2 = st.columns(2)
    with col1:
       gender = st.selectbox('Sex', ['Male', 'Female'])
    with col2:
      selected_disease = st.selectbox('Select Disease', disease)


    # Create a filter for gender selection
    filtered_df1 = df[df['Sex'] == gender]

    # Filter the data for individuals with and without heart disease
    with_heart_disease = filtered_df1[filtered_df1['HeartDisease'] == 'Yes']
    without_heart_disease = filtered_df1[filtered_df1['HeartDisease'] == 'No']

# Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(4, 3))

# Plot a histogram of BMI distribution for individuals with heart disease
    sns.histplot(data=with_heart_disease, x='BMI', kde=True, color='red', label='With Heart Disease', ax=ax)
    sns.despine()
# Plot a histogram of BMI distribution for individuals without heart disease
    sns.histplot(data=without_heart_disease, x='BMI', kde=True, color='blue', label='Without Heart Disease', ax=ax)
    sns.despine()
    # Set the font size of tick labels
# Set the plot title and axis labels
    plt.title('BMI Distribution', fontsize=8)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('BMI', fontsize= 5)
    plt.ylabel('Heart Disease Cases', fontsize=5)

# Add a legend
    plt.legend(fontsize= 5)
    # DISEASES
    
# Create a layout with two columns
# Place the radio button in the first column

# Filter the data based on the selected disease
    grouped_df1 = df.groupby([selected_disease, 'HeartDisease']).size().reset_index(name='Count1')
# Create the bar chart
    fig2 = px.bar(grouped_df1, x=selected_disease, y='Count1', color='HeartDisease',
             labels={'Count1': 'Count of Heart Disease'},
             title=f'Count of Heart Disease Cases by Disease ({selected_disease})')
    fig2.update_layout(
    height=600,  # Set the height of the figure
    width=800,  # Set the width of the figure
    )
# Set the x-axis label
    fig2.update_xaxes(title='Disease Status')
# Set the y-axis label
    fig2.update_yaxes(title='Heart Disease Cases')

# Place the chart in the second column
    col1, col2 = st.columns(2)
    with col1:
      st.pyplot(fig)
    with col2:
      st.plotly_chart(fig2)
##########################################################################################################3
        # Select the numerical variables
# Create a subset of the dataframe with the numerical variables
    numeric_vars = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    numeric_df = filtered_df1[numeric_vars]

# Create a correlation matrix
    correlation_matrix = numeric_df.corr()

# Create a heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(5, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"fontsize": 8})

# Adjust tick label font size
    ax_heatmap.tick_params(axis='both', labelsize=8)
# Set the title and axis labels
    plt.title('Correlation Matrix', fontsize=10)
    col1, col2 = st.columns(2)
    with col1:
      st.pyplot(fig_heatmap)
    with col2:
# Create a selectbox for choosing the factor
      factor = st.selectbox('Select Factor', ['Smoking', 'AlcoholDrinking'])
      
# Filter the data based on the selected factor
    filtered_df5 = df[df[factor] == 'Yes']

# Calculate the count of heart disease cases
    heart_disease_count = filtered_df5['HeartDisease'].value_counts()

# Create a pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(3, 2))
    colors = ['#0000FF', '#FF0000']  # Red and Blue colors
    patches, texts, autotexts = plt.pie(heart_disease_count, labels=heart_disease_count.index, autopct='%1.1f%%', colors=colors)
    plt.title(f'Heart Disease Cases by {factor.capitalize()}', fontsize= 5)
    plt.axis('equal')
    for text in texts:
      text.set_fontsize(3)
    for autotext in autotexts:
      autotext.set_fontsize(3)

# Set the alpha (opacity) for each pie slice
    for patch in patches:
      patch.set_alpha(0.8)
      
# Show the plots side by side
    with col2:
       st.pyplot(fig_pie)
###########################################################################################################################
  
  # Import necessary libraries
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
df.replace("Yes",1,inplace=True)
df.replace("No",0,inplace=True)

target=df["HeartDisease"]
df.drop(["HeartDisease"], axis=1, inplace=True)
df.AgeCategory.unique()
df.replace("18-24",0,inplace=True)
df.replace("25-29",1,inplace=True)
df.replace("30-34",2,inplace=True)
df.replace("35-39",3,inplace=True)
df.replace("40-44",4,inplace=True)
df.replace("45-49",5,inplace=True)
df.replace("50-54",6,inplace=True)
df.replace("55-59",7,inplace=True)
df.replace("60-64",8,inplace=True)
df.replace("65-69",9,inplace=True)
df.replace("70-74",10,inplace=True)
df.replace("75-79",11,inplace=True)
df.replace("80 or older",13,inplace=True)

df.Diabetic.unique()
df.replace("No, borderline diabetes",2,inplace=True)
df.replace("Yes (during pregnancy)",3,inplace=True)

df.GenHealth.unique()
df.replace("Excellent",0,inplace=True)
df.replace("Good",1,inplace=True)
df.replace("Fair",2,inplace=True)
df.replace("Very good",3,inplace=True)
df.replace("Poor",4,inplace=True)

df.Race.unique()
df.replace("White",0,inplace=True)
df.replace("Other",1,inplace=True)
df.replace("Black",2,inplace=True)
df.replace("Hispanic",3,inplace=True)
df.replace("Asian",4,inplace=True)
df.replace("American Indian/Alaskan Native",4,inplace=True)

df.Sex.unique()
df.replace("Female",0,inplace=True)
df.replace("Male",1,inplace=True)

df['BMI'].mask(df['BMI']  < 18.5, 0, inplace=True)
df['BMI'].mask(df['BMI'].between(18.5,25), 1, inplace=True)
df['BMI'].mask(df['BMI'].between(25,30), 2, inplace=True)
df['BMI'].mask(df['BMI']  > 30, 3, inplace=True)

# Split the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(df,target,test_size=50,random_state=2)

# Train a logistic regression model on the training set
LogRegModel=LogisticRegression()
LogRegModel.fit(X_train, y_train)

# Save the model using pickle
with open('LogRegModel.pkl', 'wb') as f:
    pickle.dump(LogRegModel, f)



if menu_id== 'Are you at Risk?':
  import streamlit as st
  import numpy as np
  import pandas  as pd
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  import pickle
  from sklearn.linear_model import LogisticRegression
  st.image("https://media.istockphoto.com/id/1314491013/vector/heart-cardiogram-line-icon-vector-illustration-isolated-on-white-background.jpg?s=612x612&w=0&k=20&c=TyWfzdG5SykehwrxRvBy9deP1zdbqKBoMntz_quSfpI=", width=400)

  # create streamlit interface, asome info about the app
  st.write("""
         ## Here you can calculate the probability of having Heart Disease based on multiple risk factors""")
  st.write("""
         ### Follow the steps below:
         ###### 1- Fill in the information.
         ###### 2- Click on the "Predict" button below.
         """)



  # st.write(BMIdata)

  #  Thesidebar func tion from streamlit is used to create a sidebar for users 
  # to input their information.
  # -------------------------------------------------------------------------
  st.sidebar.title('Please select the most suitable option')



  BMI=st.sidebar.selectbox("What's your BMI level", ("Normal weight BMI  (18.5-25)", 
                             "Underweight BMI (< 18.5)" ,
                             "Overweight BMI (25-30)",
                             "Obese BMI (> 30)"))
  Age=st.sidebar.selectbox("What's your age", 
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

  Race=st.sidebar.selectbox("What's your Race", ("Asian", 
                             "Black" ,
                             "Hispanic",
                             "American Indian/Alaskan Native",
                             "White",
                             "Other"
                             ))

  Sex=st.sidebar.selectbox("What's your gender", ("Female", 
                             "Male" ))
  Smoking = st.sidebar.selectbox("Do you consider yourself a smoker? (more than 100 cigarettes))",
                          options=("No", "Yes"))
  alcoholDink = st.sidebar.selectbox("Do you usually drink alcohol?", options=("No", "Yes"))
  stroke = st.sidebar.selectbox("Did you ever have a stroke?", options=("No", "Yes"))

  sleepTime = st.sidebar.number_input("How many hours do you sleep on average per day", 0, 24, 7) 

  genHealth = st.sidebar.selectbox("Describe your health",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))

  physHealth = st.sidebar.number_input("Physical health in the past month (Good to bad: 0-30)"
                                 , 0, 30, 0)
  mentHealth = st.sidebar.number_input("Mental health in the past month (Good to Bad: 0-30)"
                                 , 0, 30, 0)
  physAct = st.sidebar.selectbox('Were you physically active last 30 days?'
                           , options=("No", "Yes"))



  diffWalk = st.sidebar.selectbox("Do you have serious difficulty walking"
                            " or climbing stairs?", options=("No", "Yes"))
  diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                           options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
  asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
  kidneyDisease= st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
  skinCancer = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

  dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke": [stroke],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Sex": [Sex],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
  })

  #   Mapping the data as explained in the script above
  dataToPredic.replace("Underweight BMI (< 18.5)",0,inplace=True)
  dataToPredic.replace("Normal weight BMI  (18.5-25)",1,inplace=True)
  dataToPredic.replace("Overweight BMI (25-30)",2,inplace=True)
  dataToPredic.replace("Obese BMI (> 30)",3,inplace=True)

  dataToPredic.replace("Yes",1,inplace=True)
  dataToPredic.replace("No",0,inplace=True)
  dataToPredic.replace("18-24",0,inplace=True)
  dataToPredic.replace("25-29",1,inplace=True)
  dataToPredic.replace("30-34",2,inplace=True)
  dataToPredic.replace("35-39",3,inplace=True)
  dataToPredic.replace("40-44",4,inplace=True)
  dataToPredic.replace("45-49",5,inplace=True)
  dataToPredic.replace("50-54",6,inplace=True)
  dataToPredic.replace("55-59",7,inplace=True)
  dataToPredic.replace("60-64",8,inplace=True)
  dataToPredic.replace("65-69",9,inplace=True)
  dataToPredic.replace("70-74",10,inplace=True)
  dataToPredic.replace("75-79",11,inplace=True)
  dataToPredic.replace("80 or older",13,inplace=True)


  dataToPredic.replace("No, borderline diabetes",2,inplace=True)
  dataToPredic.replace("Yes (during pregnancy)",3,inplace=True)


  dataToPredic.replace("Excellent",0,inplace=True)
  dataToPredic.replace("Good",1,inplace=True)
  dataToPredic.replace("Fair",2,inplace=True)
  dataToPredic.replace("Very good",3,inplace=True)
  dataToPredic.replace("Poor",4,inplace=True)


  dataToPredic.replace("White",0,inplace=True)
  dataToPredic.replace("Other",1,inplace=True)
  dataToPredic.replace("Black",2,inplace=True)
  dataToPredic.replace("Hispanic",3,inplace=True)
  dataToPredic.replace("Asian",4,inplace=True)
  dataToPredic.replace("American Indian/Alaskan Native",4,inplace=True)


  dataToPredic.replace("Female",0,inplace=True)
  dataToPredic.replace("Male",1,inplace=True)

  # Load the previously saved machine learning model
  filename='LogRegModel.pkl'
  loaded_model= pickle.load(open(filename, 'rb'))
  Result=loaded_model.predict(dataToPredic)
  ResultProb= loaded_model.predict_proba(dataToPredic)
  ResultProb1=round(ResultProb[0][1] * 100, 2)

  # Calculate the probability of getting heart disease
  if st.button('PREDICT'):
  # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
    if (ResultProb1>30):
      st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
  else:
    st.write('You have a', ResultProb1, '% chance of getting a heart disease' )

if menu_id== "Recommendations":
   st.header("What can you do to protect your heart?")
   col1, col2, col3 = st.columns(3)
   with col1:
      st.markdown("<h3 style='text-align: center; font-size: 40px;'> Eat HEALTHY </h3>", unsafe_allow_html=True)
      st.image('https://i.pinimg.com/originals/db/35/a0/db35a0f611e023c1bd2224772b761ba3.gif', width= 500)
   with col2:
      st.markdown("<h3 style='text-align: center; font-size: 40px;'> Execrice Regularly </h3>", unsafe_allow_html=True)
      st.image("https://i.pinimg.com/originals/02/b6/2b/02b62b7ee1484dcb9331297658803a9f.gif", width=500)
   with col3:
      st.markdown("<h3 style='text-align: center; font-size: 40px;'> Sleep WELL </h3>", unsafe_allow_html=True)
      st.image("https://i.pinimg.com/originals/84/4d/ca/844dca6a465b339b1952c4892539ce9e.gif", width=500)
   col1, col2, col3 = st.columns(3)
  #  with col1:
      # st.markdown("<h3 style='text-align: center; font-size: 40px;'> Take care of your Mental Health</h3>", unsafe_allow_html=True)
      # st.write("Reach out to therapists and mental health organizations")
      # st.write("https://www.idraac.org/idraac/homepage")
   with col2:
      st.markdown("<h3 style='text-align: center; font-size: 50px;'> HAVE YOUR OWN IMPACT! </h3>", unsafe_allow_html=True)
      # st.write("Check your heart's health regularly")
  #     st.image("")
  #  with col3:
  #     st.markdown("<h3 style='text-align: center; font-size: 40px;'> Sleep WELL </h3>", unsafe_allow_html=True)
  #     st.image("https://i.pinimg.com/originals/84/4d/ca/844dca6a465b339b1952c4892539ce9e.gif", width=500)
  #  st.components.v1.iframe("https://globalhearthub.org/?gclid=CjwKCAjwp6CkBhB_EiwAlQVyxaXrseJYMOjcynFCyAgxJNClbVy3AFM8JzCD_IrS2wj_GmwmjwJMhhoCgj0QAvD_BwE", width=1000, height=1000)
# Define the URL of the website

# Define the URL of the website
   url = 'https://globalhearthub.org/?gclid=CjwKCAjwp6CkBhB_EiwAlQVyxaXrseJYMOjcynFCyAgxJNClbVy3AFM8JzCD_IrS2wj_GmwmjwJMhhoCgj0QAvD_BwE'

# Define the HTML code to extract the specific element
   html_code = '''
<div class="ums_MapPreview leaflet-container leaflet-touch leaflet-fade-anim leaflet-grab leaflet-touch-drag leaflet-touch-zoom" id="ultimate_maps_1_60703" style="position: relative; outline: none;" tabindex="0"></div>
'''

# Create an iframe component and set its content to the desired HTML code
   iframe_component = st.components.v1.html(
    f'<iframe src="{url}" width="100%" height="500" frameborder="0"></iframe>',
    height=550
)

# Use CSS styling to display only the specific element from the website
  #  iframe_component.html = iframe_component.html('</body>', html_code + '</body>')

