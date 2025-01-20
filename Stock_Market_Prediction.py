import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date,timedelta,datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM # type: ignore
import yfinance as yf
import streamlit as st
from newsapi import NewsApiClient
from langchain_community.llms import HuggingFaceHub
import warnings
warnings.filterwarnings('ignore')

# Fetch stock data
@st.cache_data
def fetch_stock_data(stock, start, end):
    return yf.download(stock, start=start, end=end)


# Train the LSTM model
@st.cache_data
def train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=16, epochs=50)
    return model

# Forecast prices
def forecast_prices(model, scaler, test_data):
    x_input = test_data[-100:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
            
    date_range = pd.date_range(date.today() - timedelta(days=1), periods=30)
    inv=scaler.inverse_transform(lst_output)
    pred=pd.DataFrame({'predictions':inv.reshape(-1)},index = date_range)
    return pred

@st.cache_data
def fetch_and_show(stock,api_key):
    api_key = api_key
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}"
    response = requests.get(url).json()
    
    if response.get("articles"):
        articles = response["articles"][:10]  
        news = []
        for article in articles:
            content = article.get("content", "")
            if content:
                news.append(content[:200] + "...")  
        return news
    return ["No news articles found."]

# Fetch news
@st.cache_data
def fetch_news(query, from_date, to_date, language='en', sort_by='relevancy', page_size=30, api_key='YOUR_API_KEY'):
            newsapi = NewsApiClient(api_key=api_key)
            query = query.replace(' ','&')
            all_articles = newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by=sort_by,
                page_size=page_size
            )

            articles = all_articles.get('articles', [])

            if articles:
                df = pd.DataFrame(articles)
                return df
            else:
                return pd.DataFrame()

#Preprocess the news into desired form
@st.cache_data
def preprocess_news_data(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df = df[~df['author'].isna()]
    df = df[['author', 'title']]
    return df

#Building the prompyt for the LLM
@st.cache_data
def build_prompt(news_df):
    prompt = "You are a financial analyst tasked with providing insights into recent news articles related to the financial industry. Here are some recent news articles:\n\n"

    for index,row in news_df.iterrows():
        title = row['title']
        prompt += f"   **News:** {title}\n\n"

    prompt += f"In 100 words analyze these articles and provide summary on the {stock} company."

    return prompt

# Streamlit UI
st.title("Stock Market Price Prediction and News Summarization")

stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):")
start_date = date.today() - timedelta(days=4 * 365)
end_date = date.today()

if st.button("Predict and Summarize"):
    try:
        # Fetch stock data
        data = fetch_stock_data(stock, start_date, end_date)
        Adj_close_price = data['Adj Close']
        
        
        st.write("### Stock Data:")
        st.write(data.tail())
        
        #Visualize the historical stock price
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Adj Close'], label=f'Adj Close {stock}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Closing Price (USD)')
        ax.legend()
        ax.grid(True)
        st.subheader(f'Adjusted Closing Price of {stock} over Time')
        st.pyplot(fig)

        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data["Adj Close"].values.reshape(-1, 1))
        
        train_data = scaled_data[:int(len(scaled_data)*0.8)]
        test_data = scaled_data[int(len(scaled_data)*0.8):]
        
        x_train, y_train = [], []
        for i in range(100, len(train_data)):
            x_train.append(train_data[i-100:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        x_test, y_test = [], []
        for i in range(100, len(test_data)):
            x_test.append(test_data[i-100:i, 0])
            y_test.append(test_data[i, 0])
        
        
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Train model
        model = train_model(x_train, y_train)   
         
         
        #Visuazlizing model preformance
        train_predictions = model.predict(x_train)
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

        # Plot the training data vs predictions
        st.subheader("Actual Stock price Vs Predicted Stock Price")
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_train_actual)), y_train_actual, label="Actual Training Data")
        plt.plot(range(len(train_predictions)), train_predictions, label="Predicted Training Data")
        plt.title("Model Performance on Training Data")
        plt.xlabel("Index")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        

        #Calculate RMSE    
        predictions = model.predict(x_test)
        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        st.write(f"Root Mean Squared Error (MSE): {rmse}")

        # Forecast prices
        pred = forecast_prices(model, scaler, test_data)


        # Visualize the data
        st.write("### Stock Price Forecast with Historical Data:")

        plt.figure(figsize=(12, 6))
        plt.plot(Adj_close_price.index, Adj_close_price, label='Actual')
        plt.plot(pred.index, pred['predictions'], label='Predicted')
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Stock Price Prediction and Forecast for {stock}")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        

        # Fetch and summarize news
        current_time = datetime.now()
        time_10_days_ago = current_time - timedelta(days=10)
        api_key = 'x'
        
        df = fetch_news(stock, time_10_days_ago, current_time, api_key=api_key)        
        df_news = df.drop("source", axis=1)
        preprocessed_news_df = preprocess_news_data(df_news)
        prompt = build_prompt(preprocessed_news_df)
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "x"
        llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            model_kwargs={"temperature": 0.1},
        )

        response=llm.invoke(prompt)
        summary = response.split("company.")[1].strip() 
        st.subheader("Summary of recent News")
        st.write(summary)
        
        #Shows latest news
        latest_news = fetch_and_show(stock,api_key)
        st.write(f"### Latest {stock} News:")
        for news in latest_news:
            st.write("- ", news)

    except Exception as e:
        st.error(f"An error occurred: {e}")
