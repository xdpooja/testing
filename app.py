import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("Virat_Kohli_ODI.csv")
df.drop(['Dismissal', 'Opposition', 'Ground', 'Start Date'], axis=1, inplace=True)
model = RandomForestRegressor()

x = df.iloc[:,1:8]
y= df['Runs']



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

model.fit(x_train, y_train)

st.sidebar.write("# USER INPUTS")
def user_input():

    Min = st.sidebar.number_input("Minutes")
    bf = st.sidebar.number_input("BF")
    fours=st.sidebar.slider("Fours", 0,25)
    sixes= st.sidebar.slider("Sixes", 0,25)
    sr= st.sidebar.number_input("SR")
    pos= st.sidebar.slider("Position", 1,9, 3)
    inn = st.sidebar.slider("Innings", 1,3)
    data = {
        'Mins' : Min,
        'BF': bf,
        'Fours': fours,
        'Sixes': sixes,
        'SR': sr,
        'Pos': pos,
        'Inns': inn
    }
    features = pd.DataFrame(data, index=[0])
    return features

dataframe = user_input()
st.write("# INPUT PARAMETERS")
st.write(dataframe)

prediction =model.predict(dataframe)
st.write("## Predicted Runs \n")
st.write("## ", prediction[0])