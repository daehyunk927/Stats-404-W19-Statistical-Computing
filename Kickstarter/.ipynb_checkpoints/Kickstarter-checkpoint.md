

```python
!pip install matplotlib
!pip install plotly
!pip install sklearn
!pip install joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import inspect
from joblib import dump, load
import random
import os
```

    Requirement already satisfied: matplotlib in c:\python37\lib\site-packages (3.0.2)
    Requirement already satisfied: python-dateutil>=2.1 in c:\python37\lib\site-packages (from matplotlib) (2.7.5)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\python37\lib\site-packages (from matplotlib) (1.0.1)
    Requirement already satisfied: cycler>=0.10 in c:\python37\lib\site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\python37\lib\site-packages (from matplotlib) (2.3.1)
    Requirement already satisfied: numpy>=1.10.0 in c:\python37\lib\site-packages (from matplotlib) (1.16.1)
    Requirement already satisfied: six>=1.5 in c:\python37\lib\site-packages (from python-dateutil>=2.1->matplotlib) (1.12.0)
    Requirement already satisfied: setuptools in c:\python37\lib\site-packages (from kiwisolver>=1.0.1->matplotlib) (40.6.2)
    

    You are using pip version 18.1, however version 19.0.2 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    

    Requirement already satisfied: plotly in c:\python37\lib\site-packages (3.6.1)
    Requirement already satisfied: requests in c:\python37\lib\site-packages (from plotly) (2.21.0)
    Requirement already satisfied: pytz in c:\python37\lib\site-packages (from plotly) (2018.9)
    Requirement already satisfied: retrying>=1.3.3 in c:\python37\lib\site-packages (from plotly) (1.3.3)
    Requirement already satisfied: decorator>=4.0.6 in c:\python37\lib\site-packages (from plotly) (4.3.0)
    Requirement already satisfied: six in c:\python37\lib\site-packages (from plotly) (1.12.0)
    Requirement already satisfied: nbformat>=4.2 in c:\python37\lib\site-packages (from plotly) (4.4.0)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in c:\python37\lib\site-packages (from requests->plotly) (1.24.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\python37\lib\site-packages (from requests->plotly) (2018.11.29)
    Requirement already satisfied: idna<2.9,>=2.5 in c:\python37\lib\site-packages (from requests->plotly) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in c:\python37\lib\site-packages (from requests->plotly) (3.0.4)
    Requirement already satisfied: jupyter-core in c:\python37\lib\site-packages (from nbformat>=4.2->plotly) (4.4.0)
    Requirement already satisfied: traitlets>=4.1 in c:\python37\lib\site-packages (from nbformat>=4.2->plotly) (4.3.2)
    Requirement already satisfied: ipython-genutils in c:\python37\lib\site-packages (from nbformat>=4.2->plotly) (0.2.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\python37\lib\site-packages (from nbformat>=4.2->plotly) (2.6.0)
    

    You are using pip version 18.1, however version 19.0.2 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    

    Requirement already satisfied: sklearn in c:\python37\lib\site-packages (0.0)
    Requirement already satisfied: scikit-learn in c:\python37\lib\site-packages (from sklearn) (0.20.2)
    Requirement already satisfied: scipy>=0.13.3 in c:\python37\lib\site-packages (from scikit-learn->sklearn) (1.2.1)
    Requirement already satisfied: numpy>=1.8.2 in c:\python37\lib\site-packages (from scikit-learn->sklearn) (1.16.1)
    

    You are using pip version 18.1, however version 19.0.2 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    

    Requirement already satisfied: joblib in c:\python37\lib\site-packages (0.13.2)
    

    You are using pip version 18.1, however version 19.0.2 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    


```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, \
                            mean_squared_error, roc_auc_score

from sklearn import svm
```


```python
import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>



```python
np.random.seed(2000)
```


```python
notebook_dir = os.getcwd()
notebook_dir
```




    'C:\\Users\\luxur\\Documents\\Stats-404-W19-Statistical-Computing\\Kickstarter'




```python
df = pd.read_csv("../Kickstarter/ks-projects-201801.csv",
                 encoding='ISO-8859-1',
                 )
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>620302213</td>
      <td>LOVELAND Round 6: A Force More Powerful</td>
      <td>Conceptual Art</td>
      <td>Art</td>
      <td>USD</td>
      <td>12/4/2009</td>
      <td>0.01</td>
      <td>11/25/2009 7:54</td>
      <td>100.0</td>
      <td>successful</td>
      <td>6</td>
      <td>US</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>688564643</td>
      <td>Word-of-mouth publishing: get "Corruptions" ou...</td>
      <td>Fiction</td>
      <td>Publishing</td>
      <td>USD</td>
      <td>12/13/2011</td>
      <td>0.01</td>
      <td>11/7/2011 16:46</td>
      <td>0.0</td>
      <td>canceled</td>
      <td>0</td>
      <td>US</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9572984</td>
      <td>Nana</td>
      <td>Shorts</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>3/16/2012</td>
      <td>0.15</td>
      <td>1/25/2012 7:23</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>US</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1379346088</td>
      <td>Bug's Big Adventure: Mexico Edition</td>
      <td>Art</td>
      <td>Art</td>
      <td>MXN</td>
      <td>11/12/2016</td>
      <td>10.00</td>
      <td>11/11/2016 16:30</td>
      <td>335.0</td>
      <td>successful</td>
      <td>7</td>
      <td>MX</td>
      <td>18.05</td>
      <td>16.41</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>219760504</td>
      <td>RocknRoll NoisePollution</td>
      <td>Documentary</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>7/19/2011</td>
      <td>0.50</td>
      <td>7/12/2011 15:59</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>US</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>69101025</td>
      <td>Most percentage funded ever for a Guinness Record</td>
      <td>Publishing</td>
      <td>Publishing</td>
      <td>MXN</td>
      <td>6/26/2017</td>
      <td>10.00</td>
      <td>5/25/2017 19:01</td>
      <td>9430.0</td>
      <td>successful</td>
      <td>2</td>
      <td>MX</td>
      <td>506.87</td>
      <td>522.81</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>6</th>
      <td>843112170</td>
      <td>Nothing (Suspended)</td>
      <td>Comedy</td>
      <td>Film &amp; Video</td>
      <td>NOK</td>
      <td>11/14/2015</td>
      <td>5.00</td>
      <td>9/30/2015 10:14</td>
      <td>0.0</td>
      <td>suspended</td>
      <td>0</td>
      <td>NO</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>952551201</td>
      <td>Xtreme Champion Tournament Issue #3</td>
      <td>Comics</td>
      <td>Comics</td>
      <td>AUD</td>
      <td>11/27/2015</td>
      <td>1.00</td>
      <td>10/30/2015 2:02</td>
      <td>1297.0</td>
      <td>successful</td>
      <td>59</td>
      <td>AU</td>
      <td>923.80</td>
      <td>934.17</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1793062138</td>
      <td>flying cars (Suspended)</td>
      <td>Flight</td>
      <td>Technology</td>
      <td>CAD</td>
      <td>2/13/2016</td>
      <td>1.00</td>
      <td>1/14/2016 17:38</td>
      <td>0.0</td>
      <td>suspended</td>
      <td>0</td>
      <td>CA</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>9</th>
      <td>741010120</td>
      <td>New Soundtracks for your Space Walk</td>
      <td>Rock</td>
      <td>Music</td>
      <td>DKK</td>
      <td>10/18/2015</td>
      <td>5.00</td>
      <td>9/18/2015 12:00</td>
      <td>145.0</td>
      <td>successful</td>
      <td>3</td>
      <td>DK</td>
      <td>21.96</td>
      <td>21.54</td>
      <td>0.74</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 378302 entries, 0 to 378301
    Data columns (total 15 columns):
    ID                  378302 non-null int64
    name                378298 non-null object
    category            378302 non-null object
    main_category       378302 non-null object
    currency            378302 non-null object
    deadline            378302 non-null object
    goal                378302 non-null float64
    launched            378302 non-null object
    pledged             378302 non-null float64
    state               378302 non-null object
    backers             378302 non-null int64
    country             378302 non-null object
    usd pledged         374509 non-null float64
    usd_pledged_real    378302 non-null float64
    usd_goal_real       378302 non-null float64
    dtypes: float64(5), int64(2), object(8)
    memory usage: 43.3+ MB
    


```python
df.isna().sum()
```




    ID                     0
    name                   4
    category               0
    main_category          0
    currency               0
    deadline               0
    goal                   0
    launched               0
    pledged                0
    state                  0
    backers                0
    country                0
    usd pledged         3793
    usd_pledged_real       0
    usd_goal_real          0
    dtype: int64




```python
### exclude NA columns
df = df.iloc[:,0:15]
```


```python
df['project_length'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched'])).dt.days + 1
```


```python
df.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>project_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>620302213</td>
      <td>LOVELAND Round 6: A Force More Powerful</td>
      <td>Conceptual Art</td>
      <td>Art</td>
      <td>USD</td>
      <td>12/4/2009</td>
      <td>0.01</td>
      <td>11/25/2009 7:54</td>
      <td>100.0</td>
      <td>successful</td>
      <td>6</td>
      <td>US</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>0.01</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>688564643</td>
      <td>Word-of-mouth publishing: get "Corruptions" ou...</td>
      <td>Fiction</td>
      <td>Publishing</td>
      <td>USD</td>
      <td>12/13/2011</td>
      <td>0.01</td>
      <td>11/7/2011 16:46</td>
      <td>0.0</td>
      <td>canceled</td>
      <td>0</td>
      <td>US</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9572984</td>
      <td>Nana</td>
      <td>Shorts</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>3/16/2012</td>
      <td>0.15</td>
      <td>1/25/2012 7:23</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>US</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.15</td>
      <td>51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1379346088</td>
      <td>Bug's Big Adventure: Mexico Edition</td>
      <td>Art</td>
      <td>Art</td>
      <td>MXN</td>
      <td>11/12/2016</td>
      <td>10.00</td>
      <td>11/11/2016 16:30</td>
      <td>335.0</td>
      <td>successful</td>
      <td>7</td>
      <td>MX</td>
      <td>18.05</td>
      <td>16.41</td>
      <td>0.49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>219760504</td>
      <td>RocknRoll NoisePollution</td>
      <td>Documentary</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>7/19/2011</td>
      <td>0.50</td>
      <td>7/12/2011 15:59</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>US</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (378302, 16)




```python
df['state'].value_counts()
```




    failed        197522
    successful    133833
    canceled       38747
    undefined       3558
    live            2797
    suspended       1845
    Name: state, dtype: int64




```python
df['main_category'].value_counts()
```




    Film & Video    63533
    Music           51864
    Publishing      39837
    Games           35203
    Technology      32542
    Design          30038
    Art             28124
    Food            24577
    Fashion         22791
    Theater         10898
    Comics          10811
    Photography     10772
    Crafts           8796
    Journalism       4748
    Dance            3768
    Name: main_category, dtype: int64




```python
# Only consider failed or successful projects
# Exclude suspended, cancelled, and other misc states
df = df[(df['state'] == "failed") | (df['state'] == "successful")]

```


```python
# Select proper features for modeling
# Exclude id, name, dates, other features that are not needed in modeling
df_features = df[['main_category', 'currency', 'state', 'backers', 'country', 'usd_pledged_real', 'usd_goal_real', 'project_length']]
```


```python
df_features['state'].value_counts()
```




    failed        197522
    successful    133833
    Name: state, dtype: int64




```python
df_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 331355 entries, 0 to 378301
    Data columns (total 8 columns):
    main_category       331355 non-null object
    currency            331355 non-null object
    state               331355 non-null object
    backers             331355 non-null int64
    country             331355 non-null object
    usd_pledged_real    331355 non-null float64
    usd_goal_real       331355 non-null float64
    project_length      331355 non-null int64
    dtypes: float64(2), int64(2), object(4)
    memory usage: 22.8+ MB
    


```python
# Modify our dependent variable to 0 or 1
mapping = {'failed': 0, 'successful': 1}
df_features = df_features.replace({'state': mapping})

```


```python
df_features['state'] = pd.to_numeric(df_features['state'], errors='coerce')
```


```python
# Categorial columns to numerical using dummy variables
df_features = pd.get_dummies(df_features)
```


```python
df_features.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>backers</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>project_length</th>
      <th>main_category_Art</th>
      <th>main_category_Comics</th>
      <th>main_category_Crafts</th>
      <th>main_category_Dance</th>
      <th>main_category_Design</th>
      <th>...</th>
      <th>country_JP</th>
      <th>country_LU</th>
      <th>country_MX</th>
      <th>country_N,0"</th>
      <th>country_NL</th>
      <th>country_NO</th>
      <th>country_NZ</th>
      <th>country_SE</th>
      <th>country_SG</th>
      <th>country_US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6</td>
      <td>100.00</td>
      <td>0.01</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.15</td>
      <td>51</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>7</td>
      <td>16.41</td>
      <td>0.49</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2</td>
      <td>522.81</td>
      <td>0.55</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>




```python
# Split the data to train and test
df_train, df_valid = train_test_split(df_features,
                                      test_size = 0.25,
                                      random_state=2018)
```


```python
df_train['state'].value_counts()
```




    0    148174
    1    100342
    Name: state, dtype: int64




```python
df_valid['state'].value_counts()
```




    0    49348
    1    33491
    Name: state, dtype: int64




```python
y = df_train['state']
X = df_train.drop(columns=['state'])
```


```python
X.shape
```




    (248516, 56)




```python
y.shape
```




    (248516,)




```python
inspect.signature(RandomForestClassifier)
```




    <Signature (n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)>




```python
### --- Step 1: Specify different number of trees in forest, to determine
###             how many to use based on leveling-off of OOB error:
n_trees = [50, 100, 250, 500, 1000]
```


```python
### --- Step 2: Create dictionary to save-off each estimated RF model:
rf_dict = dict.fromkeys(n_trees)
```


```python
for num in n_trees:
    print(num)
    ### --- Step 3: Specify RF model to estimate:
    rf = RandomForestClassifier(n_estimators=num,
                                min_samples_leaf=30,
                                oob_score=True,
                                random_state=2018,
                                class_weight='balanced',
                                verbose=1)
    ### --- Step 4: Estimate RF model and save estimated model:
    rf.fit(X, y)
    rf_dict[num] = rf
```

    50
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:   51.0s finished
    

    100
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:  1.5min finished
    

    250
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 250 out of 250 | elapsed:  3.7min finished
    

    500
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 500 out of 500 | elapsed:  8.0min finished
    

    1000
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed: 15.4min finished
    


```python

### --- Save-off model:
# Specify location and name of object to contain estimated model:
model_object_path = os.path.join(notebook_dir, 'rf.joblib')
# Save estimated model to specified location:
dump(rf_dict, model_object_path) 

```




    ['C:\\Users\\luxur\\Documents\\Stats-404-W19-Statistical-Computing\\Kickstarter\\rf.joblib']




```python
# Compute OOB error per
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
oob_error_list = [None] * len(n_trees)

# Find OOB error for each forest size:
for i in range(len(n_trees)):
    oob_error_list[i] = 1 - rf_dict[n_trees[i]].oob_score_
else:
    # Visulaize result:
    plt.plot(n_trees, oob_error_list, 'bo',
             n_trees, oob_error_list, 'k')
```


![png](output_34_0.png)



```python
# Feature importance plot
top_num = 20
forest = rf_dict[1000]
importances = forest.feature_importances_

# Sort in decreasing order:
indices = np.argsort(importances)[::-1]    

```


```python
len(importances)
```




    56




```python
np.array(list(X))[indices[0:top_num]]
```




    array(['backers', 'usd_pledged_real', 'usd_goal_real', 'project_length',
           'main_category_Technology', 'main_category_Music',
           'main_category_Theater', 'main_category_Design',
           'main_category_Food', 'main_category_Games',
           'main_category_Fashion', 'main_category_Comics',
           'main_category_Art', 'main_category_Dance',
           'main_category_Film & Video', 'currency_EUR', 'currency_USD',
           'main_category_Publishing', 'country_US', 'main_category_Crafts'],
          dtype='<U26')




```python
# Plot the feature importances of the forest
ax = plt.gca()
plt.title(f"Top {top_num} feature importances")
plt.bar(range(top_num), importances[indices[0:top_num]])
plt.xticks(range(top_num))
ax.set_xticklabels(np.array(list(X))[indices[0:top_num]], rotation = 90)
ax.set_xlabel("Features")
ax.set_ylabel("Feature Importance")
plt.show()
```


![png](output_38_0.png)



```python
y_pred_train = forest.predict(X)
y_pred_train[0:5]
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:  2.8min finished
    




    array([0, 1, 0, 0, 1], dtype=int64)




```python
conf_mat = confusion_matrix(y_true=y,
                            y_pred=y_pred_train)
conf_mat
```




    array([[144068,   4106],
           [   635,  99707]], dtype=int64)




```python
class_names = ['failed','successful']
```


```python
conf_df = pd.DataFrame(conf_mat, class_names, class_names)
conf_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>failed</th>
      <th>successful</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>failed</th>
      <td>144068</td>
      <td>4106</td>
    </tr>
    <tr>
      <th>successful</th>
      <td>635</td>
      <td>99707</td>
    </tr>
  </tbody>
</table>
</div>




```python
conf_df_pct = conf_df/conf_df.sum(axis=1)
round(conf_df_pct, 2)
# Very Successful results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>failed</th>
      <th>successful</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>failed</th>
      <td>0.97</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>successful</th>
      <td>0.00</td>
      <td>0.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Class-level performance: 0.9803
f1_score(y_true=y,
         y_pred=y_pred_train,
         average='macro')
```




    0.9802948828116604




```python
# Overall performance across all classes: 0.9809
f1_score(y_true=y,
         y_pred=y_pred_train,
         average='micro')
```


```python
# Alternate model: Support Vector Machine
C = 1.0
#svm_dict = dict.fromkeys(['linear', 'LinearSVC', 'rbf', 'poly'])
svm_linear = svm.SVC(kernel='linear', C=C)
svm_LinearSVC = svm.LinearSVC(C=C)
svm_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
svm_poly = svm.SVC(kernel='poly', degree=3, C=C)
svm_linear.fit(X,y)
svm_LinearSVC.fit(X,y)
svm_rbf.fit(X,y)
svm_poly.fit(X,y)

svm = [svm_linear, svm_LinearSVC, svm_rbf, svm_poly]
#svm_dict['linear'] = svm_linear
#svm_dict['LinearSVC'] = svm_LinearSVC
#svm_dict['rbf'] = svm_rbf
#svm_dict['poly'] = svm_poly
```


```python
# Compute OOB error per
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
oob_error_list = [None] * 4

# Find OOB error for each forest size:
for i in range(4):
    oob_error_list[i] = 1 - svm[i].oob_score_
else:
    # Visulaize result:
    plt.plot(np.array(list(svm)), oob_error_list, 'bo',
             np.array(list(svm)), oob_error_list, 'k')
```




    <generator object <genexpr> at 0x000001B8DA7E4408>




```python
y_pred_train_svm = svm_linear.predict(X)
y_pred_train_svm[0:5]
```


```python
conf_mat_svm = confusion_matrix(y_true=y,
                            y_pred=y_pred_train_svm)
conf_mat_svm
```


```python
conf_df_svm = pd.DataFrame(conf_mat_svm, class_names, class_names)
conf_df_svm
```


```python
conf_df_pct_svm = conf_df_svm/conf_df_svm.sum(axis=1)
round(conf_df_pct_svm, 2)
# Very Successful results
```


```python
# Class-level performance: 0.9803
f1_score(y_true=y,
         y_pred=y_pred_train_svm,
         average='macro')
```


```python
# Overall performance across all classes: 0.9809
f1_score(y_true=y,
         y_pred=y_pred_train_svm,
         average='micro')
```
