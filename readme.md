# Bitcoin Price Prediction Using Sentiment Analysis

This project aims to predict Bitcoin price using a combination of natural language processing (NLP) techniques, specifically sentiment analysis on news headlines, and numerical financial indicators. 

## License

This project is licensed under the MIT License.

## Dataset

The primary component of our dataset is news headlines related to Bitcoin, cryptocurrency, or economics in general. These headlines were scraped from **Coindesk** and paired with numerical attributes including:

1. Price
2. Opening price
3. Minimum and maximum price
4. Volume
5. Price Change

## Steps 

The following steps were accomplished in this project:

1. **Data Collection**: We crawled news headlines from the internet to create our dataset. This data was supplemented with numerical attributes like closing price, minimum and maximum price, etc. 
2. **Data Preprocessing**
   1. **Sentiment Analysis**: This section focuses on analyzing the emotional undertones present in news headlines. It utilizes a specialized SentimentAnalysis class to read a dataset of headlines, preprocess the text by removing noise and cleaning it, and then evaluate the sentiment of each headline. The processed data, which now includes sentiment information, is stored in a new CSV file.
   2. **Dataset Modification**: This section aims to clean and standardize the dataset further. It includes two key functions. The first one is "double_quotation_remover," which removes any double quotation marks from the CSV files to ensure data integrity and readability. The second function, "modify_date_format," is used to normalize the date format across the entire dataset, ensuring consistency and facilitating data analysis.
   3. **Data Combination**: In this section, the different data streams are combined using the "combine_data" function. The function merges the Bitcoin price data, obtained from an external financial database, with the sentiment data derived from the headlines. It also cleanses the numerical data by removing special characters and converting them into appropriate data types for further analysis. The merging process is based on the "Date" column. The resulting combined data is saved into a CSV file.
3. **Feature Engineering**: The next stage in our methodology involves the engineering of features, a process that occurs within the process_data function. Here, we take several strategic steps to optimize the data for machine learning models:
   1. The date column is reformatted into a datetime format, ensuring that the time-stamped nature of the data is correctly represented and interpreted by subsequent models.
   2. The volume column, which initially contains textual data, is transformed into a numerical representation, making it compatible with computational operations and analysis.
   3. We create lagged features and rolling window features such as the mean, min, and max. These features are derived from previous data points and capture the temporal patterns in the data, providing our machine-learning models with crucial information about trends and fluctuations over time.
   4. As a part of data cleaning, any rows with NaN values are identified and systematically eliminated. These NaN values may have been introduced during the process of creating lagged and rolling window features and could lead to inaccuracies if not dealt with.
   5. The features are then normalized using one of three possible methods: StandardScaler, RobustScaler, or MinMaxScaler. This choice depends on the specific distribution of the dataset and the requirements of the machine learning model being used. But in our case, we decided to use MinMaxScaler as our scaler.
   6. The target column, which is the Bitcoin Price, is also scaled using the same scaler as the feature data. This ensures that the range of the target data is in line with that of the input features, which can significantly enhance the performance of certain machine learning algorithms.

### Models
#### Traditional Machine-Learning Models
AdaBoost Regressor, Bagging Regressor, Bayesian Ridge Regressor, Decision Tree Regressor, Elastic Net Regressor, Extra Trees Regressor, Gaussian Process Regressor, Gradient Boosting Regressor, KNN Regressor (k-Nearest Neighbors'), Lasso Regressor, LGBM Regressor (Light Gradient-Boosting Machine), NuSVR Regressor Model (Nu Support Vector Regression), Passive Aggressive Regressor, Random Forest Regressor, Ridge Regressor, Support Vector Regressor, Tweedie Regressor and XGB Regressor (eXtreme Gradient Boosting).

#### Neural Network Models
CNN Regressor (Convolutional Neural Networks), LSTM Regressor (Long Short-Term Memory), and Basic Neural Network Regressor.

## Performance Metrics

The success rate of our model was evaluated using several metrics including MSE and RMSE.

## Installation

To run this project, you will need Python 3.6 (or later) and the packages listed in `requirements.txt`. You can install them using:

`pip install -r requirements.txt`

## Contributing

Contributions to this project are welcome. Please open an issue to discuss your ideas or submit a pull request with your changes.

## Contact

If you have any questions or feedback, please feel free to contact me.

## Known Issues & Bugs
Some unused functions are still present, `weighted_predict_best_models` and `vote_predict_best_models` are not working as expected.