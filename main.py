import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sweetviz as sv
import plotly.graph_objs as pg
import plotly.figure_factory as ffc
from tensorflow import keras
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | df[column] > upper_bound]


def process_data():

    train_data = pd.read_csv("weather_data.csv")
    test_data = pd.read_csv("weather_data.csv")

    #-----------TRENOVACIE DATA--------------#

    # najskor pridam do irradiance vsade 0 aby sa neodstranilo prilis vela zaznamov
    train_data["Irradiance"] = train_data["Irradiance"].fillna(0.0)

    # odstranenie prazdnych zaznamov
    train_data = train_data.dropna()
    train_data = train_data.drop_duplicates(keep='first')
    train_data = train_data.fillna(0)

    train_data = train_data.drop('Irradiance', axis=1)
    

    # zistenie kolko roznych druhov Cloud Cover sa nachadza v DB a ich zakodovnie
    # Cloud Cover sme si vybrali druh kodovanie: Label Encoding 

    unique_values_cloud = train_data['Cloud Cover'].unique()
    print(f"Unique words: {unique_values_cloud}")
    print(f"Number of unique words cloud: {len(unique_values_cloud)}")

    train_data['Cloud Cover Encoded'] = pd.factorize(train_data['Cloud Cover'])[0]

    unique_values_weather = train_data['Weather Type'].unique()
    print(f"Unique words: {unique_values_weather}")
    print(f"Number of unique words weather: {len(unique_values_weather)}")

    train_data['Weather Type Encoded'] = pd.factorize(train_data['Weather Type'])[0]
    
    # na stlpec season sme vybrali druh zakódovania One-Hot Encoding

    train_data_one_hot = pd.get_dummies(train_data['Season'], prefix='Season')
    train_data = pd.concat([train_data, train_data_one_hot], axis = 1)
    print(train_data.head(10))

    # Vyhľadanie duplicitných záznamov

    duplicate = train_data[train_data.duplicated()]
    train_data = train_data.drop_duplicates()
    print("Počet záznamov v datasete: "+str(len(train_data)))
    print("V datasete sa nachádza: " + str(len(duplicate)) + " duplikátov.")

    train_data.isnull().sum()


    train_data['Atmospheric Pressure'].plot(kind='box')
    plt.show()
    

process_data()