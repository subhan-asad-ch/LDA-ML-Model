import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Constants for solar power calculation
SOLAR_PANEL_AREA = 1.5  # m^2
SOLAR_PANEL_EFFICIENCY = 0.18

# Constants for wind power calculation
RADIUS = 20  # m
AIR_DENSITY = 1.225  # kg/m^3
CP = 0.35

# Read data from Excel file, ignoring the first 2 rows
file_path = 'Weather Dataset.xlsx'
data = pd.read_excel(file_path, skiprows=2)

# Print column names to verify
print("Columns in the dataset:")
print(data.columns)

# Extract the hour, month, and day from the columns
data['Hour'] = data['Hour']
data['Month'] = data['Month']
data['Day'] = data['Day']

# Create a datetime-like column for better plotting
data['Datetime'] = pd.to_datetime(data[['Month', 'Day']].assign(year=2014)) + pd.to_timedelta(data['Hour'], unit='h')


# Function to calculate power outputs
def calculate_power_outputs(data):
    data = data.copy()  # Make a copy to avoid SettingWithCopyWarning

    data.loc[:, 'Solar Power Output'] = np.where(data['DNI'] > 200,
                                                 data['DNI'] * SOLAR_PANEL_AREA * SOLAR_PANEL_EFFICIENCY, 0)
    data.loc[:, 'Wind Power Output'] = np.where((data['Wind Speed'] > 2.5) & (data['Wind Speed'] <= 10),
                                                0.5 * AIR_DENSITY * np.pi * (RADIUS ** 2) * CP * (
                                                        data['Wind Speed'] ** 3), 0)
    data.loc[:, 'Total Energy Output'] = data['Solar Power Output'] + data['Wind Power Output']

    return data


# Filter data for the hours 9 to 17
data_9_to_17 = data[(data['Hour'] >= 9) & (data['Hour'] <= 17)]
data_9_to_17 = calculate_power_outputs(data_9_to_17)

# Filter data for the hours 18 to 23
data_18_to_23 = data[(data['Hour'] >= 18) & (data['Hour'] <= 23)]
data_18_to_23 = calculate_power_outputs(data_18_to_23)

# Print column names to verify
print("Columns in data_9_to_17:")
print(data_9_to_17.columns)

print("Columns in data_18_to_23:")
print(data_18_to_23.columns)


# Create a categorical column for classification purposes
def create_energy_category(data):
    data['Energy Category'] = pd.cut(data['Total Energy Output'],
                                     bins=[0, 50, 100, 150, 200, np.inf],
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    return data


data_9_to_17 = create_energy_category(data_9_to_17)
data_18_to_23 = create_energy_category(data_18_to_23)

# Remove rows with missing values in target column
data_9_to_17 = data_9_to_17.dropna(subset=['Energy Category'])
data_18_to_23 = data_18_to_23.dropna(subset=['Energy Category'])

# Check the size of data after dropping NaNs
print("Size of data_9_to_17 after dropping NaNs:", data_9_to_17.shape)
print("Size of data_18_to_23 after dropping NaNs:", data_18_to_23.shape)

# Ensure that the data has enough samples
if data_9_to_17.shape[0] == 0 or data_18_to_23.shape[0] == 0:
    raise ValueError("Insufficient data after dropping NaN values. Please check your data.")


# Function to prepare and train LDA model
from sklearn.preprocessing import LabelEncoder


def train_lda_model(data):
    features = data[['Solar Power Output', 'Wind Power Output']]
    target = data['Energy Category']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)

    y_pred = lda_model.predict(X_test)

    # Convert categorical labels to numerical codes
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_pred_encoded = le.transform(y_pred)

    lda_accuracy = accuracy_score(y_test, y_pred)
    lda_report = classification_report(y_test, y_pred)

    # Calculate Mean Squared Error, Mean Absolute Error, and R Squared Error
    mse = mean_squared_error(y_test_encoded, y_pred_encoded)
    mae = mean_absolute_error(y_test_encoded, y_pred_encoded)
    r2 = r2_score(y_test_encoded, y_pred_encoded)

    print(f"LDA Accuracy: {lda_accuracy}")
    print(f"LDA Classification Report:\n{lda_report}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R Squared Error: {r2}")

    return lda_model, X_test, y_test, y_pred


# Train LDA model for 9 to 17 hours
lda_model_9_to_17, X_test_9_to_17, y_test_9_to_17, y_pred_9_to_17 = train_lda_model(data_9_to_17)

# Train LDA model for 18 to 23 hours
lda_model_18_to_23, X_test_18_to_23, y_test_18_to_23, y_pred_18_to_23 = train_lda_model(data_18_to_23)


# Function to create results DataFrame for plotting
def create_results_df(data, X_test, y_test, y_pred, lda_model):
    results_df = pd.DataFrame({
        'Actual Solar': y_test.index.map(lambda idx: data.loc[idx, 'Solar Power Output']),
        'Predicted Solar': X_test.index.map(lambda idx: lda_model.predict([X_test.loc[idx]])[0]),
        'Actual Wind': y_test.index.map(lambda idx: data.loc[idx, 'Wind Power Output']),
        'Predicted Wind': X_test.index.map(lambda idx: lda_model.predict([X_test.loc[idx]])[0]),
        'Actual Total': y_test.index.map(lambda idx: data.loc[idx, 'Total Energy Output']),
        'Predicted Total': y_pred,
        'Month': y_test.index.map(lambda idx: data.loc[idx, 'Month']),
        'Day': y_test.index.map(lambda idx: data.loc[idx, 'Day']),
        'Datetime': y_test.index.map(lambda idx: data.loc[idx, 'Datetime'])
    })
    return results_df


# Create results DataFrame for 9 to 17 hours
results_df_9_to_17 = create_results_df(data_9_to_17, X_test_9_to_17, y_test_9_to_17, y_pred_9_to_17, lda_model_9_to_17)
results_df_9_to_17.to_csv('results_9_to_17.csv', index=False)

# Create results DataFrame for 18 to 23 hours
results_df_18_to_23 = create_results_df(data_18_to_23, X_test_18_to_23, y_test_18_to_23, y_pred_18_to_23,
                                        lda_model_18_to_23)
results_df_18_to_23.to_csv('results_18_to_23.csv', index=False)


# Function to create scatter plot
def scatter_plot(data, x_col, y_col, title, ylabel, label1, label2, winter_months, summer_months, color1, color2,
                 file_name):
    plt.figure(figsize=(12, 8))
    for month in winter_months:
        plt.scatter(data[data['Month'] == month][x_col],
                    data.loc[data[data['Month'] == month].index, y_col],
                    color=color1, label=label1 if month == winter_months[0] else "", alpha=0.5)
    for month in summer_months:
        plt.scatter(data[data['Month'] == month][x_col],
                    data.loc[data[data['Month'] == month].index, y_col],
                    color=color2, label=label2 if month == summer_months[0] else "", alpha=0.5)
    plt.xlabel('Datetime')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()


# Define winter and summer months
winter_months = [1, 2, 3, 10, 11, 12]
summer_months = [4, 5, 6, 7, 8, 9]

# Scatter plot for 9 to 17 hours
scatter_plot(results_df_9_to_17, 'Datetime', 'Actual Solar',
             'Actual Solar Power Output for 0900-1700 hrs',
             'Solar Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_actual_solar_9_to_17.png')

scatter_plot(results_df_9_to_17, 'Datetime', 'Predicted Solar',
             'Predicted Solar Power Output for 0900-1700 hrs',
             'Solar Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_predicted_solar_9_to_17.png')

scatter_plot(results_df_9_to_17, 'Datetime', 'Actual Wind',
             'Actual Wind Power Output for 0900-1700 hrs',
             'Wind Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_actual_wind_9_to_17.png')

scatter_plot(results_df_9_to_17, 'Datetime', 'Predicted Wind',
             'Predicted Wind Power Output for 0900-1700 hrs',
             'Wind Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_predicted_wind_9_to_17.png')

scatter_plot(results_df_9_to_17, 'Datetime', 'Actual Total',
             'Actual Total Energy Output for 0900-1700 hrs',
             'Total Energy Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_actual_total_9_to_17.png')

scatter_plot(results_df_9_to_17, 'Datetime', 'Predicted Total',
             'Predicted Total Energy Output for 0900-1700 hrs',
             'Total Energy Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_predicted_total_9_to_17.png')

# Scatter plot for 18 to 23 hours
scatter_plot(results_df_18_to_23, 'Datetime', 'Actual Solar',
             'Actual Solar Power Output for 1800-2300 hrs',
             'Solar Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_actual_solar_18_to_23.png')

scatter_plot(results_df_18_to_23, 'Datetime', 'Predicted Solar',
             'Predicted Solar Power Output for 1800-2300 hrs',
             'Solar Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_predicted_solar_18_to_23.png')

scatter_plot(results_df_18_to_23, 'Datetime', 'Actual Wind',
             'Actual Wind Power Output for 1800-2300 hrs',
             'Wind Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_actual_wind_18_to_23.png')

scatter_plot(results_df_18_to_23, 'Datetime', 'Predicted Wind',
             'Predicted Wind Power Output for 1800-2300 hrs',
             'Wind Power Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_predicted_wind_18_to_23.png')

scatter_plot(results_df_18_to_23, 'Datetime', 'Actual Total',
             'Actual Total Energy Output for 1800-2300 hrs',
             'Total Energy Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_actual_total_18_to_23.png')

scatter_plot(results_df_18_to_23, 'Datetime', 'Predicted Total',
             'Predicted Total Energy Output for 1800-2300 hrs',
             'Total Energy Output (Watts)', 'Winter', 'Summer',
             winter_months, summer_months, 'blue', 'red', 'scatter_predicted_total_18_to_23.png')
