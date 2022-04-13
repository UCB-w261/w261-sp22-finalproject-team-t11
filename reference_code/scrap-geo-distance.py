# Databricks notebook source
import pandas as pd
import numpy as np

from sklearn.neighbors import BallTree
from io import StringIO

def example():
    # Create DataFrame from you lat/lon dataset
    data = """NAME,Latitude,Longitude
    B,50.94029883,7.019146728
    C,50.92073002,6.975268711
    D,50.99807758,6.980865543
    E,50.98074288,7.035060206
    F,51.00696972,7.035993783
    G,50.97369889,6.928538763
    H,50.94133859,6.927878587
    A,50.96712502,6.977825322"""

    # Use StringIO to allow reading of string as CSV
    df = pd.read_csv(StringIO(data), sep = ',')
    

    # Setup Balltree using df as reference dataset
    # Use Haversine calculate distance between points on the earth from lat/long
    # haversine - https://pypi.org/project/haversine/ 
    tree = BallTree(np.deg2rad(df[['Latitude', 'Longitude']].values), metric='haversine')

    # Setup distance queries (points for which we want to find nearest neighbors)
    other_data = """NAME,Latitude,Longitude
    B_alt,50.94029883,7.019146728
    C_alt,50.92073002,6.975268711"""

    df_other = pd.read_csv(StringIO(other_data), sep = ',')

    query_lats = df_other['Latitude']
    query_lons = df_other['Longitude']

    # Find closest city in reference dataset for each in df_other
    # use k = 3 for 3 closest neighbors
    distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k = 3)

    r_km = 6371 # multiplier to convert to km (from unit distance)
    for name, d, ind in zip(df_other['NAME'], distances, indices):
        print(f"NAME {name} closest matches {ind}:")
        for i, index in enumerate(ind):
            print(f"\t{df['NAME'][index]} with distance {d[i]*r_km:.4f} km")
        
example()

# COMMAND ----------


