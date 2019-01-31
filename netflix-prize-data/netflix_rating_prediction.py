import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# Load Movie Mapping Data
movie_titles = pd.read_csv('../netflix-prize-data/movie_titles.csv',
                           header = None,
                           encoding = 'ISO-8859-1',
                           names = ['Movie_ID', 'Release_yr', 'Title'])
movie_titles.set_index('Movie_ID', inplace = True)
print ('glancing at the movie_titles')
print (movie_titles.head(10))
print ('Shape of movie_titles: ', movie_titles.shape)

# Load the training dataset
df_1 = pd.read_csv('../netflix-prize-data/combined_data_1.txt',
                   header = None,
                   names = ['User_ID', 'Rating', 'Date'])

df_1['Rating'] = df_1['Rating'].astype(float)
print ('glancing at the training data 1')
print (df_1.head(10))
print ('Shape of training data 1: ', df_1.shape)

# Exploratory Data Analysis

# Movies by Release Year
data = movie_titles['Release_yr'].value_counts().sort_index()
plt.plot(data.index, data.values)
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.title('Movies by Release Year')
plt.show()

# Rating Histogram
data = df_1['Rating'].value_counts().sort_index(ascending=False)
plt.bar(data.index, data.values)
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.title('Ratings Histogram')
plt.show()

# Rating per User Histogram
data = df_1.groupby('User_ID')['Rating'].count().clip(upper=199)
plt.hist(data.values, bins=100)
plt.title('Ratings Per Movie')
plt.show()