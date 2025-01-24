# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.11
#     language: python
#     name: py311
# ---

# %%
import pandas as pd
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# %%
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error


# %%
def preprocessing(df, remove_date=False):
    
    ## preprocessing
    
    df = df.dropna()
    # Convert 'date' to datetime and extract useful features (e.g., year, month, day)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Feature engineering
    df["month_sin"] = np.sin(df['month'] * (2 * np.pi / 12))
    df["month_cos"] = np.cos(df['month'] * (2 * np.pi / 12))
    
    df['day'] = df['date'].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df['date'].apply(
        lambda x: x.timetuple().tm_yday if not (x.is_leap_year and x.month > 2) else x.timetuple().tm_yday - 1
    )
    important_dates = [1,2,3,4,5,6,7,8,9,10,99, 100, 101, 125,126,355,256,357,358,359,360,361,362,363,364,365]
    df["important_dates"] = df["day_of_year"].apply(lambda x: x if x in important_dates else 0)
    

    # Drop unnecessary columns
    if remove_date:
        df = df.drop(columns=['id', 'date'])
    else:
        df = df.drop(columns=['id'])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['country', 'store', 'product'], drop_first=True)

    return df


# %%
def test_train_split(df, test_size=0.2):

    # Calculate the cutoff date
    cutoff_date = df['date'].iloc[int((1 - test_size) * len(df))]
    
    # Split the data based on the cutoff date
    train = df[df['date'] < cutoff_date]  # All rows before the cutoff date
    test = df[df['date'] >= cutoff_date]  # All rows on or after the cutoff date 

    # Separate features (X) and target (y)
    X_train = train.drop(columns=['num_sold', 'date'])
    y_train = train['num_sold']

    X_test = test.drop(columns=['num_sold', 'date'])
    y_test = test['num_sold']

    print(f"Train length: {len(train)}")
    print(f"Test length: {len(test)}")
    print("Training set date range:", train['date'].min(), "to", train['date'].max())
    print("Testing set date range:", test['date'].min(), "to", test['date'].max())

    return X_train, X_test, y_train, y_test


# %%
def fit_model(X_train, X_test, y_train):
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# %%
def fit_model(X_train, y_train):
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# %%
def model_predict(model, X_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Convert predictions to integers
    y_pred = np.round(y_pred).astype(int)  # Round to nearest integer and convert to int

    # Clip negative values to zero
    y_pred = np.clip(y_pred, 0, None)  # Set all values less than 0 to 0

    return y_pred


# %%
def eval_model(y_test, y_pred):

    # Calculate evaluation metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")


# %%
df_pre = preprocessing(train_data)
X_train, X_test, y_train, y_test = test_train_split(df_pre)
model = fit_model(X_train, y_train)
y_pred = model_predict(model, X_test)
eval_model(y_test, y_pred)

# %%
# Plot actual vs. predicted values
y = pd.concat([y_train, y_test])

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual num_sold')
plt.ylabel('Predicted num_sold')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()
plt.show()


# %%
def run_model_on_test(train_data, test_data, save_path="submission.csv"):
    train_data_pp =  preprocessing(train_data, remove_date=False)
    test_data_pp = preprocessing(test_data, remove_date=True)

    X = train_data_pp.drop(columns=['num_sold', 'date'])
    y = train_data_pp['num_sold']
    model = fit_model(X, y)
    y_pred = model_predict(model, test_data_pp)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYYMMDD_HHMMSS
    # Append the timestamp to the save_path
    save_path_with_timestamp = f"{save_path}_{timestamp}.csv"

    submission = pd.DataFrame(data={'num_sold': y_pred}, index=test_data["id"])
    if save_path:
        submission.to_csv(save_path_with_timestamp)
    return submission
    

    

# %%
submission = run_model_on_test(train_data, test_data)

# %%
submission

# %%
