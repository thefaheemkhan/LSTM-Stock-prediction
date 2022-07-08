# LSTM-Stock-prediction

# What we've created :
### A Machine Learning Model. which will help to predict future price of stocks, with the help of predefined neural network algorithms.
In this model, we proposed the financial data which is collected from stock market and applying algorithms in order to make some useful insights from provided financial data. This project aims to build a Deep learning model that can predict the future asset values of provided financial data with the help of Recurrent Neural Network (RNN) , especially (LSTM) Long Short Term Memory to prediction of stock market data.

# Why we've created :
Investors are familiar with the saying, “buy low, sell high” but this does not provide enough context to make proper investment decisions. Before an investor invests in any stock, he needs to be aware how the stock market behaves. Investing in a good stock but at a bad time can have disastrous results, while investment in a mediocre stock at the right time can bear profits. Financial investors of today are facing this problem of trading as they do not properly understand as to which stocks to buy or which stocks to sell in order to get optimum profits. Predicting long term value of the stock is relatively easy than predicting on day-to-day basis as the stocks fluctuate rapidly every hour based on world events.

## Now the Interesting part is How we've created :
Lets say 10 days data that we are taking and we have to predict the data for 11th day. For example ( x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 ) we have taken the 10 days data . Now we are going to predict the value of 11th day. So we should know that the value of the 11th day is going to be dependent on these previous 10 days values that we are taken. It will not just become 100 or 200 on 11th day ,It will be just dependent on these previous values only. But in this case we are talking about stock market, which fluctuate on the basis of external news, reports and performance of companies. We have to take care of external outliers that going to occurs. We worked on same Methodology that we have described above.

## Lets have brief look over the building process.
Libraries and Dependencies used in Project.
- PANDAS
- NUMPY
- MTPLOTLIB
- YAHOO FINANCE 
- PANDAS DATAREADER
- STREAMLIT
- TENSORFLOW
- SKLEARN

### Algorithm Used:
## Recurrent Neural Network (RNN)
A recurrent neural network (RNN) is a special type of an artificial neural network adapted to work for time series data or data that involves sequences. Ordinary feed forward neural networks are only meant for data points, which are independent of each other. However, if we have data in a sequence such that one data point depends upon the previous data point, we need to modify the neural network to incorporate the dependencies between these data points. RNNs have the concept of ‘memory’ that helps them store the states or information of previous inputs to generate the next output of the sequence.

## Long Short Term Memory (LSTM)
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.


# Proposed Steps:
1. 1. As a first step, we are importing the data from yahoo finance website and defining the start and the end point of dataset that we are going to use in our machine learning model.
2. After importing we will reset the index and dropping the columns from dataset, which are not useful for our model analysis. And selecting the CLOSE price column on that we are going to train our machine learning model.
3. DATA SPLITTING - Now coming to the important part of any machine learning model which Data Splitting. We have split the data into Training and Testing part that we usually do for predictions. So we have split a data in such a manner that Training part is 70% of the data and the rest 30% is for Testing part.
4. DATA SCALING - For this we are importing MinMax Scaler from Sklearn data preprocessing. And defining the features from 0 to 1. That means all the values from the closing price column will scaled down between 0 to 1. That’s the way we provide data to our LSTM model.
5. In next step we have to split our data into x train and y train , So that’s why we have use a Time Series Analogy That value for a particular day or we can say the closing price of a particular day will be dependent on a previous days values. In this model we have defined steps as 100, that means the value for the 101th day will be dependent on a previous 100 days. That’s why this previous 100 days values become my x train and the 101th day value will be my y train.
6. Then we have to convert our x train and y train into a numpy arrays.
7. Then we have defined a simple LSTM model, So in this model we Defined 4 layers in LSTM model and at last a Dense layer which connects the all layers together.
8. Now heading towards to finalize our machine learning model , we compiled the model with ADAM Optimizer (Zhang, 2018)and kept the losses as Mean Squared error. Which is used in Time Series Analysis. At last compiled our model for 50 epochs.


# Prediction:
<img src="#"/>
