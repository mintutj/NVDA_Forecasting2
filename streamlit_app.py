import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, TCN, FEDformer, NHITS, TimesNet, Autoformer, PatchTST
from neuralforecast.losses.numpy import mae,mse
import streamlit as st

def load_and_prepare_data(file_path, ticker):
    """Load and prepare the dataset for inference."""
    df = pd.read_csv(file_path)
    df['unique_id'] = ticker
    df.rename(columns={'time': 'ds','close': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    return df[['ds', 'y', 'unique_id']].sort_values('ds').reset_index(drop=True)

def split_data(Y_df, val_ratio=0.2, test_ratio=0.2):
    """Split data into training, validation, and test sets."""
    n_time = len(Y_df['ds'].unique())
    val_size = int(val_ratio * n_time)
    test_size = int(test_ratio * n_time)
    train_size = n_time - val_size - test_size
    
    train_eval_data = Y_df.iloc[:train_size+val_size]
    test_data = Y_df.iloc[train_size:train_size + test_size]
    
    return train_eval_data, test_data, val_size


def load_model(model_path):
    """Load the saved model checkpoint."""
    nf_model = NeuralForecast.load(path=model_path)
    return nf_model

def testing(nf_model, test_data, horizon, selected_cutoff, actual_column='y'):
    """Test models and calculate metrics."""

    # Split test data into context and forecast
    #cutt_off = "2024-08-14 11:00:00-04:00"
    cutt_off = selected_cutoff
    test_context = test_data[test_data['ds'] <= cutt_off]  # Context: All data before cutoff
    test_future = test_data[test_data['ds'] > cutt_off].iloc[:horizon] 

    y_test = nf_model.predict(df=test_context)
    # Merge predictions with actual data
    Y_hat_test = pd.merge(
        test_future[['unique_id', 'ds', actual_column]],
        y_test,
        on=['unique_id', 'ds'],
        how='left'
    )
    return Y_hat_test,test_context

def calculate_metrics(Y_hat_test, actual_column='y'):
    """Calculate MAE and MSE for each model."""
    metrics = {}
    for model in Y_hat_test.columns[3:]:
        mae_val = mae(Y_hat_test[actual_column], Y_hat_test[model])
        mse_val = mse(Y_hat_test[actual_column], Y_hat_test[model])
        metrics[model] = {'MAE': mae_val, 'MSE': mse_val}

    for model, vals in metrics.items():
        print(f"{model} - MAE: {vals['MAE']}, MSE: {vals['MSE']}")
    return metrics

def plot_predictions(Y_hat_test,test_context):
    """Plot the predictions of each model alongside the true values."""
    plt.figure()#figsize=(20, 6))

    # Plot the actual values
    plt.plot(Y_hat_test['ds'], Y_hat_test['y'], label='True', color='green')
    plt.plot(Y_hat_test['ds'], Y_hat_test['Autoformer'], label='predicted', color='red')
    plt.plot(test_context['ds'].tail(12), test_context['y'].tail(12), label='Test Context', color='blue')#, linestyle='--')

    # Extract model names
    # model_names = [model.__class__.__name__ for model in models]

    # Generate a list of colors
    # colors = itertools.cycle(['orange', 'green', 'red', 'purple', 'brown', 'cyan', 'magenta'])

    # Plot predictions from each model
    # for model, color in zip(model_names, colors):
        # plt.plot(Y_hat_test['ds'], Y_hat_test[model], label=model, color=color)


    # Add labels and title
    plt.xlabel('Date & Time')
    plt.ylabel('Close')
    plt.title('Comparison of True Values and Model Predictions')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend(loc='upper right')

    if not os.path.exists('./graphs'):
        os.makedirs('./graphs')
        print('Created graphs folder')
    plt.savefig('./graphs/predictions_comparison_NVDA1.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to execute the forecasting pipeline."""

    # Define the mapping of tickers to their corresponding files and model paths
    ticker_to_paths = {
        'NVDA': {'csv': './data/NASDAQ_NVDA_5min_with_indicator.csv', 'model': './models/checkpoints0'},
        'AAPL': {'csv': 'path_to_aapl.csv', 'model': './models/checkpoints_aapl'},
        'SPY': {'csv': 'path_to_spy.csv', 'model': './models/checkpoints_spy'},
    }
        
    st.title("Forecasting Pipeline with Streamlit")

    ticker = st.selectbox("Select Ticker", options=list(ticker_to_paths.keys()))

    cutoff_times = [
        "2024-08-23 08:00:00-04:00",
        "2024-08-23 09:00:00-04:00",
        "2024-08-23 10:00:00-04:00",
        "2024-08-23 11:00:00-04:00",
        "2024-08-23 12:00:00-04:00",
        "2024-08-23 13:00:00-04:00",
        "2024-08-23 14:00:00-04:00",
        "2024-08-23 15:00:00-04:00",
        "2024-08-23 16:00:00-04:00"
    ]

    selected_cutoff = st.selectbox("Select Cutoff Time", options=cutoff_times)

    horizon = 12
    
    if ticker and selected_cutoff:
        file_path = ticker_to_paths[ticker]['csv']
        model_path = ticker_to_paths[ticker]['model']
        
        Y_df = load_and_prepare_data(file_path, ticker=ticker)
        train_data, test_data, val_size = split_data(Y_df)
        # print("test_data", test_data)
        
        nf = load_model(model_path=model_path)
        
        Y_hat_test, test_context = testing(nf, test_data, horizon, selected_cutoff)
        
        plot_predictions(Y_hat_test, test_context)
        st.image('./graphs/predictions_comparison_NVDA1.png')

# Execute the main function
if __name__ == "__main__":
    main()