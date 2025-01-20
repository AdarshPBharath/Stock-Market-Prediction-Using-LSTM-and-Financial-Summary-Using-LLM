Stock Market Prediction and News Summarization
This project leverages machine learning and natural language processing to predict stock prices and summarize recent financial news related to a selected stock. The application provides insights into the stock's historical and forecasted prices, as well as recent news summaries, through an interactive Streamlit-based web interface.

Features
1. Stock Price Prediction:
  a.Historical stock price visualization.
  b.LSTM-based predictive modeling to forecast stock prices.
  c.Root Mean Squared Error (RMSE) calculation to evaluate model performance.

2. News Summarization:
  Fetches recent financial news using News API.
  Summarizes news articles using a language model from Hugging Face Hub.
  Displays the latest articles and their summaries.

3.Interactive Dashboard:
  Simple and user-friendly interface powered by Streamlit.
  Real-time data retrieval and visualization.
  
Tech Stack
  Frontend: Streamlit
  Backend: Python, Keras, LangChain, Hugging Face Hub
  APIs: Yahoo Finance, News API
Libraries:
  NumPy, Pandas: Data manipulation and analysis.
  Matplotlib: Data visualization.
  Scikit-learn: Data preprocessing and performance evaluation.
  Keras: Building and training the LSTM model.
  yfinance: Fetching stock market data.
  NewsApiClient: Retrieving news articles.
  HuggingFace : Prompting the LLM
