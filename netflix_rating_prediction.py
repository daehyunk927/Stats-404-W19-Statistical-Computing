import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# The movie rating files contain over 100 million ratings from 480 thousand
# randomly-chosen, anonymous Netflix customers over 17 thousand movie titles.  The
# data were collected between October, 1998 and December, 2005 and reflect the
# distribution of all ratings received during this period.  The ratings are on a
# scale from 1 to 5 (integral) stars. To protect customer privacy, each customer
# id has been replaced with a randomly-assigned id.  The date of each rating and
# the title and year of release for each movie id are also provided.

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
data_percent = round(df_1['Rating'].value_counts(normalize=True).sort_index(ascending=False) * 100,2)
print(data_percent)
plt.bar(data.index, data.values)
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.title('Ratings Histogram')
plt.show()

# Rating per User Histogram
data = df_1.groupby('User_ID')['Rating'].count().clip(upper=199)
plt.hist(data.values, bins=100)
plt.title('Ratings Per User')
plt.show()