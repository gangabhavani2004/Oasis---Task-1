# Task 1

# Imporing Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


```


```python
import warnings
warnings.filterwarnings('ignore')
```

# 1. Data Loading and Cleaning: Load the retail sales dataset.


```python
df = pd.read_csv("C:/Users/GANGA/Desktop/Oyasis/retail_sales_dataset-Task1.csv");
```


```python
# Display the first few rows of the dataframe
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
      <th>Transaction ID</th>
      <th>Date</th>
      <th>Customer ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Product Category</th>
      <th>Quantity</th>
      <th>Price per Unit</th>
      <th>Total Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-11-24</td>
      <td>CUST001</td>
      <td>Male</td>
      <td>34</td>
      <td>Beauty</td>
      <td>3</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2023-02-27</td>
      <td>CUST002</td>
      <td>Female</td>
      <td>26</td>
      <td>Clothing</td>
      <td>2</td>
      <td>500</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2023-01-13</td>
      <td>CUST003</td>
      <td>Male</td>
      <td>50</td>
      <td>Electronics</td>
      <td>1</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2023-05-21</td>
      <td>CUST004</td>
      <td>Male</td>
      <td>37</td>
      <td>Clothing</td>
      <td>1</td>
      <td>500</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2023-05-06</td>
      <td>CUST005</td>
      <td>Male</td>
      <td>30</td>
      <td>Beauty</td>
      <td>2</td>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2023-04-25</td>
      <td>CUST006</td>
      <td>Female</td>
      <td>45</td>
      <td>Beauty</td>
      <td>1</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2023-03-13</td>
      <td>CUST007</td>
      <td>Male</td>
      <td>46</td>
      <td>Clothing</td>
      <td>2</td>
      <td>25</td>
      <td>50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>2023-02-22</td>
      <td>CUST008</td>
      <td>Male</td>
      <td>30</td>
      <td>Electronics</td>
      <td>4</td>
      <td>25</td>
      <td>100</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>2023-12-13</td>
      <td>CUST009</td>
      <td>Male</td>
      <td>63</td>
      <td>Electronics</td>
      <td>2</td>
      <td>300</td>
      <td>600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>2023-10-07</td>
      <td>CUST010</td>
      <td>Female</td>
      <td>52</td>
      <td>Clothing</td>
      <td>4</td>
      <td>50</td>
      <td>200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (1000, 9)




```python
df.columns
```




    Index(['Transaction ID', 'Date', 'Customer ID', 'Gender', 'Age',
           'Product Category', 'Quantity', 'Price per Unit', 'Total Amount'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 9 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   Transaction ID    1000 non-null   int64 
     1   Date              1000 non-null   object
     2   Customer ID       1000 non-null   object
     3   Gender            1000 non-null   object
     4   Age               1000 non-null   int64 
     5   Product Category  1000 non-null   object
     6   Quantity          1000 non-null   int64 
     7   Price per Unit    1000 non-null   int64 
     8   Total Amount      1000 non-null   int64 
    dtypes: int64(5), object(4)
    memory usage: 70.4+ KB
    


```python
# Check for missing values
df.isnull().sum()
```




    Transaction ID      0
    Date                0
    Customer ID         0
    Gender              0
    Age                 0
    Product Category    0
    Quantity            0
    Price per Unit      0
    Total Amount        0
    dtype: int64




```python
df = df.dropna()
df
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
      <th>Transaction ID</th>
      <th>Date</th>
      <th>Customer ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Product Category</th>
      <th>Quantity</th>
      <th>Price per Unit</th>
      <th>Total Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2023-11-24</td>
      <td>CUST001</td>
      <td>Male</td>
      <td>34</td>
      <td>Beauty</td>
      <td>3</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2023-02-27</td>
      <td>CUST002</td>
      <td>Female</td>
      <td>26</td>
      <td>Clothing</td>
      <td>2</td>
      <td>500</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2023-01-13</td>
      <td>CUST003</td>
      <td>Male</td>
      <td>50</td>
      <td>Electronics</td>
      <td>1</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2023-05-21</td>
      <td>CUST004</td>
      <td>Male</td>
      <td>37</td>
      <td>Clothing</td>
      <td>1</td>
      <td>500</td>
      <td>500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2023-05-06</td>
      <td>CUST005</td>
      <td>Male</td>
      <td>30</td>
      <td>Beauty</td>
      <td>2</td>
      <td>50</td>
      <td>100</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>996</td>
      <td>2023-05-16</td>
      <td>CUST996</td>
      <td>Male</td>
      <td>62</td>
      <td>Clothing</td>
      <td>1</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>996</th>
      <td>997</td>
      <td>2023-11-17</td>
      <td>CUST997</td>
      <td>Male</td>
      <td>52</td>
      <td>Beauty</td>
      <td>3</td>
      <td>30</td>
      <td>90</td>
    </tr>
    <tr>
      <th>997</th>
      <td>998</td>
      <td>2023-10-29</td>
      <td>CUST998</td>
      <td>Female</td>
      <td>23</td>
      <td>Beauty</td>
      <td>4</td>
      <td>25</td>
      <td>100</td>
    </tr>
    <tr>
      <th>998</th>
      <td>999</td>
      <td>2023-12-05</td>
      <td>CUST999</td>
      <td>Female</td>
      <td>36</td>
      <td>Electronics</td>
      <td>3</td>
      <td>50</td>
      <td>150</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1000</td>
      <td>2023-04-12</td>
      <td>CUST1000</td>
      <td>Male</td>
      <td>47</td>
      <td>Electronics</td>
      <td>4</td>
      <td>30</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 9 columns</p>
</div>



# 2. Descriptive Statistics: Calculate basic statistics (mean, median, mode, standard deviation).


```python
# Get summary statistics of the dataframe
df.describe()
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
      <th>Transaction ID</th>
      <th>Age</th>
      <th>Quantity</th>
      <th>Price per Unit</th>
      <th>Total Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>500.500000</td>
      <td>41.39200</td>
      <td>2.514000</td>
      <td>179.890000</td>
      <td>456.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>288.819436</td>
      <td>13.68143</td>
      <td>1.132734</td>
      <td>189.681356</td>
      <td>559.997632</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>18.00000</td>
      <td>1.000000</td>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>250.750000</td>
      <td>29.00000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>500.500000</td>
      <td>42.00000</td>
      <td>3.000000</td>
      <td>50.000000</td>
      <td>135.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>750.250000</td>
      <td>53.00000</td>
      <td>4.000000</td>
      <td>300.000000</td>
      <td>900.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1000.000000</td>
      <td>64.00000</td>
      <td>4.000000</td>
      <td>500.000000</td>
      <td>2000.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Time Series Analysis: Analyze sales trends over time using time series techniques.


```python
# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
```


```python
# Set the date column as the index
df.set_index('Date', inplace=True)
```


```python
# Resample the data by month and sum the total amount
monthly_sales = df['Total Amount'].resample('M').sum()
```


```python
# Plot the time series
plt.figure(figsize=(8, 4))
plt.plot(monthly_sales, label='Monthly Sales')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.legend()
plt.show()
```


    
![png](output_18_0.png)
    



```python
# Plot sales data
df.plot()
plt.show()

```


    
![png](output_19_0.png)
    



```python
# Decompose sales data into trend, seasonality, and residuals
decomposition = seasonal_decompose(df['Total Amount'], model='additive', period=12)
trend = decomposition.trend
seasonality = decomposition.seasonal
residuals = decomposition.resid


```


```python
# Plot decomposition components
plt.figure(figsize=(10,6))
plt.subplot(411)
plt.plot(df['Total Amount'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



```


    
![png](output_21_0.png)
    



```python
# Fit ARIMA model
model = ARIMA(df['Total Amount'], order=(5,1,0))
model_fit = model.fit()

```


```python
# Plot forecast
forecast = model_fit.forecast(steps=30)
plt.plot(df['Total Amount'])
plt.plot(forecast)
plt.show()

```


    
![png](output_23_0.png)
    



```python
# Resample the data to quarterly frequency
quarterly_sales = monthly_sales.resample('Q').sum()

```


```python
plt.plot(monthly_sales, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)

```




    <Axes: >




    
![png](output_25_1.png)
    


# 4. Customer and Product Analysis: Analyze customer demographics and purchasing behavior
Customer Demographics Analysis

```python
#Age Distribution:
df['Age'].describe() #summary statistics

```




    count    1000.00000
    mean       41.39200
    std        13.68143
    min        18.00000
    25%        29.00000
    50%        42.00000
    75%        53.00000
    max        64.00000
    Name: Age, dtype: float64




```python
#histogram
df['Age'].plot.hist()
plt.figure(figsize=(3, 1))
```




    <Figure size 300x100 with 0 Axes>




    
![png](output_29_1.png)
    



    <Figure size 300x100 with 0 Axes>


Gender Distribution


```python
# Pie chart
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()


```


    
![png](output_31_0.png)
    


Purchasing Behavior Analysis



```python
#Average Order Value (AOV):
df['Total Amount'].mean() #average order value

```




    456.0




```python

# Count plot
sns.countplot(x='Customer ID', data=df)
plt.title('Purchase Frequency')
plt.figure(figsize=(4, 2))  # Set a larger figure size
plt.xlabel('Customer ID')
plt.ylabel('Number of Purchases')
plt.show()


```


    
![png](output_34_0.png)
    



    
![png](output_34_1.png)
    



```python
# Product Category Distribution:
df['Product Category'].value_counts().plot.bar()
plt.show()

```


    
![png](output_35_0.png)
    



```python

#Customer Segmentation:
df.groupby('Customer ID')['Total Amount'].sum() #total spend by customer
df.groupby('Customer ID')['Total Amount'].sum().plot.hist() #histogram

```




    <Axes: ylabel='Frequency'>




    
![png](output_36_1.png)
    


# 5. Visualization: Present insights through bar charts, line plots, and heatmaps


```python
# Bar chart: Top 10 products by total amount
plt.figure(figsize=(6,4))
sns.barplot(x='Product Category', y='Total Amount', data=df.nlargest(10, 'Total Amount'))
plt.title('Top 10 Products by Total Amount')
plt.show()

```


    
![png](output_38_0.png)
    



```python
# Line plot: Total amount trend over time
plt.figure(figsize=(6,4))
plt.plot(df.index, df['Total Amount'])
plt.title('Total Amount Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.show()

```


    
![png](output_39_0.png)
    



```python
# Heatmap: Total amount by product category and gender
pivot_table = pd.pivot_table(df, values='Total Amount', index='Product Category', columns='Gender', aggfunc='sum')
plt.figure(figsize=(10,6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
plt.title('Total Amount by Product Category and Gender')
plt.show()

```


    
![png](output_40_0.png)
    


# 6. Recommendations: Provide actionable recommendations based on the EDA.


```python
# Target high-value customers
high_value_customers = df[df['Total Amount'] > 1000]
print(high_value_customers)

```

                Transaction ID Customer ID  Gender  Age Product Category  \
    Date                                                                   
    2023-08-05              13     CUST013    Male   22      Electronics   
    2023-01-16              15     CUST015  Female   42      Electronics   
    2023-02-17              16     CUST016    Male   19         Clothing   
    2023-05-23              31     CUST031    Male   44      Electronics   
    2023-06-26              46     CUST046  Female   20      Electronics   
    ...                    ...         ...     ...  ...              ...   
    2023-03-18             942     CUST942    Male   51         Clothing   
    2023-10-16             943     CUST943  Female   57         Clothing   
    2023-05-08             946     CUST946    Male   62      Electronics   
    2023-08-19             956     CUST956    Male   30         Clothing   
    2023-05-16             970     CUST970    Male   59      Electronics   
    
                Quantity  Price per Unit  Total Amount  
    Date                                                
    2023-08-05         3             500          1500  
    2023-01-16         4             500          2000  
    2023-02-17         3             500          1500  
    2023-05-23         4             300          1200  
    2023-06-26         4             300          1200  
    ...              ...             ...           ...  
    2023-03-18         3             500          1500  
    2023-10-16         4             300          1200  
    2023-05-08         4             500          2000  
    2023-08-19         3             500          1500  
    2023-05-16         4             500          2000  
    
    [153 rows x 8 columns]
    


```python
# Product category optimization
top_categories = df['Product Category'].value_counts().nlargest(5)
print(top_categories)

```

    Product Category
    Clothing       351
    Electronics    342
    Beauty         307
    Name: count, dtype: int64
    


```python
# Gender-based marketing
male_customers = df[df['Gender'] == 'Male']
female_customers = df[df['Gender'] == 'Female']

```


```python
# Age-based promotions
young_customers = df[(df['Age'] >= 18) & (df['Age'] <= 35)]
older_customers = df[(df['Age'] >= 45) & (df['Age'] <= 60)]

```


```python
# Seasonal sales strategies
seasonal_sales = df.groupby(df.index.quarter)['Total Amount'].sum()
print(seasonal_sales)

```

    Date
    1    110030
    2    123735
    3     96045
    4    126190
    Name: Total Amount, dtype: int64
    


```python
# Customer retention
loyal_customers = df[df['Customer ID'].isin(df['Customer ID'].value_counts().nlargest(10).index)]
print(loyal_customers)

```

                Transaction ID Customer ID  Gender  Age Product Category  \
    Date                                                                   
    2023-11-24               1     CUST001    Male   34           Beauty   
    2023-03-19             659     CUST659  Female   39      Electronics   
    2023-04-29             660     CUST660  Female   38           Beauty   
    2023-07-16             661     CUST661  Female   44         Clothing   
    2023-12-22             662     CUST662    Male   48           Beauty   
    2023-03-20             663     CUST663    Male   23         Clothing   
    2023-12-28             664     CUST664  Female   44         Clothing   
    2023-04-20             665     CUST665    Male   57         Clothing   
    2023-02-02             666     CUST666    Male   51      Electronics   
    2023-08-01             672     CUST672  Female   34           Beauty   
    
                Quantity  Price per Unit  Total Amount  
    Date                                                
    2023-11-24         3              50           150  
    2023-03-19         1              30            30  
    2023-04-29         2             500          1000  
    2023-07-16         4              25           100  
    2023-12-22         2             500          1000  
    2023-03-20         4             300          1200  
    2023-12-28         4             500          2000  
    2023-04-20         1              50            50  
    2023-02-02         3              50           150  
    2023-08-01         2              50           100  
    


```python
# Product bundling
df['Product ID'] = df['Product Category'].astype('category').cat.codes
product_bundles = df.groupby('Product Category')['Product ID'].apply(list)
print(product_bundles)

```

    Product Category
    Beauty         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    Clothing       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
    Electronics    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ...
    Name: Product ID, dtype: object
    


```python
# Price optimization
price_sensitivity = df.groupby('Price per Unit')['Total Amount'].sum()
print(price_sensitivity)

```

    Price per Unit
    25      13050
    30      13350
    50      26700
    300    155400
    500    247500
    Name: Total Amount, dtype: int64
    


```python

```


```python

```


```python

```
