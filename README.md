Stock Market Prediction and News Summarization
This project leverages machine learning and natural language processing to predict stock prices and summarize recent financial news related to a selected stock. The application provides insights into the stock's historical and forecasted prices, as well as recent news summaries, through an interactive Streamlit-based web interface.

How to use:
Firstly you need the API keys from HuggingFace and NewAPI and you also need a Streamlit account to host this web application. You need to add you APIs to the code and run the code using "streamlit run {name of your file}". After that you will get a UI that asks for the stock ticker, type any ticker and see your predicition results visualized using graph and a financial summary of recent news of the stock.
Features

Stock Price Prediction:
  a.Historical stock price visualization.
  b.LSTM-based predictive modeling to forecast stock prices.
  c.Root Mean Squared Error (RMSE) calculation to evaluate model performance.

News Summarization:
  a.Fetches recent financial news using News API.
  b.Summarizes news articles using a language model from Hugging Face Hub.
  c.Displays the latest articles and their summaries.

Interactive Dashboard:
  a.Simple and user-friendly interface powered by Streamlit.
  b.Real-time data retrieval and visualization.
  
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
