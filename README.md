# Air Quality Index Prediction using TensorFlow

A machine learning project that predicts the **Air Quality Index (AQI)** using air pollution measurements from different cities in India.

The project uses a fully connected neural network built with **TensorFlow/Keras** to learn the relationship between pollutant concentrations and AQI values.

The workflow includes data preprocessing, missing-value handling, exploratory data analysis, feature scaling, neural network training, model evaluation, and AQI prediction for new pollutant measurements.

## Project Overview

The objective of this project is to predict the AQI based on various air pollutant measurements.

The model uses the following pollutants as input features:

* PM2.5
* PM10
* NO
* NO2
* NOx
* NH3
* CO
* SO2
* O3
* Benzene
* Toluene
* Xylene

The target variable is:

```text
AQI
```

Since AQI is a continuous numerical value, this project treats the problem as a **regression problem**.

## Technologies Used

* Python
* TensorFlow
* Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly

## Dataset

The project uses the `city_day.csv` dataset.

The dataset contains air pollution measurements recorded for different cities over time.

Important columns include:

| Column     | Description                                                |
| ---------- | ---------------------------------------------------------- |
| City       | Name of the city                                           |
| Date       | Date of measurement                                        |
| PM2.5      | Particulate matter with diameter less than 2.5 micrometers |
| PM10       | Particulate matter with diameter less than 10 micrometers  |
| NO         | Nitric oxide                                               |
| NO2        | Nitrogen dioxide                                           |
| NOx        | Nitrogen oxides                                            |
| NH3        | Ammonia                                                    |
| CO         | Carbon monoxide                                            |
| SO2        | Sulfur dioxide                                             |
| O3         | Ozone                                                      |
| Benzene    | Benzene concentration                                      |
| Toluene    | Toluene concentration                                      |
| Xylene     | Xylene concentration                                       |
| AQI        | Air Quality Index                                          |
| AQI_Bucket | AQI category                                               |

## Data Preprocessing

The dataset is loaded using Pandas:

```python
import pandas as pd

data = pd.read_csv("city_day.csv")
```

Missing values are checked using:

```python
data.isnull().sum()
```

The current implementation removes rows containing missing values:

```python
data.dropna(axis=0, inplace=True)
```

The `Date` column is then converted into datetime format:

```python
data['Date'] = pd.to_datetime(data['Date'])
```

### Note

Dropping every row containing a missing value is simple but can discard a substantial portion of the dataset. A more advanced implementation could use techniques such as median imputation or city-specific interpolation.

## Exploratory Data Analysis

Plotly is used to visualize relationships and trends in the dataset.

### AQI Trend Over Time

The AQI trend for different cities is visualized using a line chart:

```python
fig1 = px.line(
    data,
    x='Date',
    y='AQI',
    color='City',
    title='AQI Trend Over Time'
)

fig1.show()
```

### AQI Distribution by City

A box plot is used to compare AQI distributions across cities:

```python
fig2 = px.box(
    data,
    x='City',
    y='AQI',
    title='AQI Distribution by City'
)

fig2.update_layout(
    xaxis={'categoryorder':'total descending'}
)

fig2.show()
```

### Scatter Plot Matrix

Relationships between selected pollutants and AQI are explored using a scatter plot matrix:

```python
selected_features = [
    'PM2.5',
    'NO2',
    'CO',
    'O3',
    'AQI'
]

fig3 = px.scatter_matrix(
    data[selected_features],
    title='Scatter Plot Matrix'
)

fig3.show()
```

## Feature Selection

The following pollutant measurements are used as input features:

```python
feature_columns = [
    'PM2.5',
    'PM10',
    'NO',
    'NO2',
    'NOx',
    'NH3',
    'CO',
    'SO2',
    'O3',
    'Benzene',
    'Toluene',
    'Xylene'
]
```

The input and target variables are created as:

```python
X = data[feature_columns]
y = data['AQI']
```

## Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

80% of the data is used for training and 20% is reserved for testing.

## Feature Scaling

Standardization is performed using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardization transforms the features so that they have approximately zero mean and unit variance.

The scaler is fitted only on the training data and then applied to the test data to avoid data leakage.

## Neural Network Architecture

The project uses a simple feed-forward neural network:

```text
Input Layer
12 Features

Dense Layer
64 Neurons
ReLU Activation

Dense Layer
32 Neurons
ReLU Activation

Output Layer
1 Neuron
Linear Activation

Output
Predicted AQI
```

The model is defined using TensorFlow/Keras:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

The final layer contains one neuron because the model predicts a single continuous AQI value.

## Model Compilation

The model uses the Adam optimizer and Mean Squared Error loss:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='mean_squared_error'
)
```

### Loss Function

Mean Squared Error (MSE) is appropriate for this regression problem.

The model minimizes the squared difference between the actual AQI and predicted AQI values.

## Training Configuration

The model is configured with:

* Maximum epochs: 200
* Batch size: 32
* Initial learning rate: 0.001
* Validation split: 20%
* Optimizer: Adam
* Loss: Mean Squared Error

The model is trained using:

```python
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)
```

## Training Callbacks

Two callbacks are used to improve the training process.

### ReduceLROnPlateau

The learning rate is reduced when the validation loss stops improving:

```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)
```

During training, the learning rate was reduced from:

```text
0.001
```

to:

```text
0.0005
```

then:

```text
0.00025
```

and finally:

```text
0.000125
```

### EarlyStopping

Training is stopped when validation loss stops improving:

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
```

Training stopped at **epoch 165**, and the weights from the best-performing epoch were restored.

The best epoch was **epoch 150** based on the validation loss.

## Training Results

The validation loss continued to improve substantially during training.

The best validation loss observed was approximately:

```text
425.57
```

at epoch 150.

The training process was automatically stopped at epoch 165 because further training did not provide sufficient improvement.

## Model Evaluation

The trained model is evaluated on the test dataset:

```python
loss = model.evaluate(
    X_test_scaled,
    y_test
)

print(
    "Mean Squared Error on Test Data:",
    loss
)
```

The resulting test Mean Squared Error was approximately:

```text
455.88
```

### Important Evaluation Metric

Since the model uses Mean Squared Error, the reported value is:

```text
Test MSE = 455.88
```

A more interpretable metric is the Root Mean Squared Error (RMSE):

```python
import numpy as np

rmse = np.sqrt(455.8786)

print("RMSE:", rmse)
```

This gives an RMSE of approximately:

```text
21.35 AQI points
```

This means that the model's typical prediction error is around 21 AQI points, although RMSE alone does not provide a complete picture of model performance.

## Training Loss Visualization

The training and validation losses are plotted using Matplotlib:

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(
    ['Train', 'Validation'],
    loc='upper left'
)

plt.show()
```

This plot helps visualize how the model learns over time and whether the model is overfitting.

## Saving the Model

The trained model is saved using:

```python
model.save('model.h5')
```

The saved model can later be loaded for prediction without retraining.

### Note

The HDF5 `.h5` format is considered a legacy Keras format. The recommended modern format is:

```python
model.save('model.keras')
```

## Making AQI Predictions

New pollutant measurements can be provided as a Pandas DataFrame:

```python
user_input = pd.DataFrame({
    'PM2.5': [81],
    'PM10': [124],
    'NO': [1.44],
    'NO2': [20],
    'NOx': [12],
    'NH3': [10],
    'CO': [0.1],
    'SO2': [15],
    'O3': [127],
    'Benzene': [0.20],
    'Toluene': [6],
    'Xylene': [0.06]
})
```

The input must be scaled using the same scaler that was fitted on the training data:

```python
user_input_scaled = scaler.transform(user_input)
```

The trained model is then used to generate the prediction:

```python
user_pred = model.predict(user_input_scaled)

print(f"Predicted AQI: {user_pred[0][0]}")
```

For the example input, the model produced:

```text
Predicted AQI: 194.05
```

## Project Workflow

```text
Load Dataset
    |
    v
Handle Missing Values
    |
    v
Convert Date Column
    |
    v
Exploratory Data Analysis
    |
    v
Select Pollution Features
    |
    v
Train-Test Split
    |
    v
Feature Standardization
    |
    v
Build Neural Network
    |
    v
Train Model
    |
    v
Reduce Learning Rate
    |
    v
Early Stopping
    |
    v
Evaluate on Test Data
    |
    v
Save Model
    |
    v
Predict AQI for New Data
```

## Recommended Project Structure

```text
Air-Quality-Index-Prediction/
│
├── city_day.csv
├── air_quality_prediction.ipynb
├── model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset and trained model do not necessarily need to be committed to GitHub, especially if they are large.

A suitable `.gitignore` can include:

```gitignore
# Virtual environment
venv/
.venv/

# Dataset
*.csv

# Python cache
__pycache__/
*.py[cod]

# Jupyter
.ipynb_checkpoints/

# Model files
*.h5
*.keras

# Environment variables
.env
```

## Limitations

The current implementation has several limitations:

* Rows with missing values are completely removed, which can significantly reduce the available data.
* The model does not use `City` as an input feature.
* The `Date` feature is not used for prediction.
* Only a simple feed-forward neural network is used.
* Model performance is evaluated primarily using MSE.
* No cross-validation is performed.
* No hyperparameter optimization is performed.
* The model predicts AQI directly rather than AQI categories.

## Possible Improvements

### Better Missing-Value Handling

Instead of removing all incomplete rows, missing pollutant values could be handled using:

* Median imputation
* Mean imputation
* Forward filling
* Interpolation
* City-specific imputation

### Additional Evaluation Metrics

The model could be evaluated using:

* MAE
* MSE
* RMSE
* R² Score

For example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

predictions = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
```

### Use Time-Based Splitting

Because AQI data is time-dependent, a random train-test split may not accurately represent how the model would perform when predicting future AQI values.

A better approach for a forecasting-oriented project would be to train on earlier dates and test on later dates.

### Include City Information

Different cities can have substantially different pollution patterns. Encoding the `City` column could potentially improve predictions.

### Try Other Machine Learning Models

The neural network could be compared against:

* Linear Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Regression

This would help determine whether a neural network is actually the best model for this tabular regression problem.

## Project Objectives

* Analyze air pollution data
* Explore AQI trends across cities
* Understand relationships between pollutants and AQI
* Build a neural network regression model
* Apply feature standardization
* Use learning-rate scheduling
* Use early stopping
* Evaluate AQI prediction performance
* Predict AQI for new pollution measurements

## Future Improvements

* Improve missing-value handling
* Add city and date-based features
* Compare multiple machine learning algorithms
* Perform hyperparameter tuning
* Add MAE, RMSE, and R² evaluation
* Use time-series train/test splitting
* Build an AQI prediction dashboard
* Deploy the trained model as a web application
* Add real-time AQI prediction using live air-quality data
