# LSTM Stock Trend Prediction

This project utilizes LSTM (Long Short-Term Memory) neural networks to predict stock price trends based on historical data.

## Description

This project aims to forecast stock price trends by leveraging Deep Learning techniques. It includes a frontend interface that allows users to input ticker symbols of interest. Upon submission, the backend retrieves the historical stock data for each ticker from Yahoo Finance, preprocesses it, generates training, validation, and test datasets, trains an LSTM model, and generates predictions. The predictions are then plotted to visualize the model's performance on the test set and to showcase the predicted trend for the next quarter.

## Getting Started

### Using Mamba

1. Install [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Navigate to the directory where your project resides.
3. Create a new Mamba environment and activate it
```
mamba create -n lstm-stock-prediction python=3.11.8
mamba activate lstm-stock-prediction
```

### Installation
1. Clone the repository from GitHub:
```
git clone https://github.com/nikkiDEEE/lstm-stock-price-prediction.git
```
2. To get all the required dependencies, run the following command:
```
mamba install -r requirements.txt
```

### Executing program

1. Navigate to the directory which contains the ```start.sh``` file (it should the current directory you are in).
2. On your terminal run the following command to enable permissions:
```
chmod +x start.sh
```
3. Then run start.sh:
```
./start.sh
```

## Help

Contact me on LinkedIn (below).

## Authors

Contributors names and contact info

* Nikhil Shashidhar
* [LinkedIn](https://www.linkedin.com/in/nikhil-will-work-wonders/)
