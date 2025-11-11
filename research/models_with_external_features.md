# Models That Handle External Features (Weather, Outages) for Electricity Trading

Based on the comprehensive research for MISO/PJM/SPP virtual trading strategies, here are the models that **explicitly handle external features like weather and outages**:

## Models That Accept External Features

### 1. **Temporal Fusion Transformer (TFT)** ⭐ BEST FOR YOUR USE CASE

**Why it's ideal:**
- **Explicitly designed** for time-varying covariates (weather, load forecasts)
- **Static covariates** (node characteristics, seasonal indicators)
- **Built-in feature selection** through gating mechanisms
- **Interpretable attention weights** showing which features matter

**Example usage:**
```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Create dataset with multiple covariates
training = TimeSeriesDataSet(
    data=df,
    time_idx="hour",
    target="lmp_spread",

    # Static features (don't change over time)
    static_categoricals=["node_id", "zone"],
    static_reals=["latitude", "longitude"],

    # Time-varying known (weather forecasts, load forecasts, outages)
    time_varying_known_categoricals=["is_outage", "constraint_active"],
    time_varying_known_reals=["temperature_forecast", "wind_speed_forecast",
                               "load_forecast", "solar_generation_forecast"],

    # Time-varying unknown (what you're trying to predict)
    time_varying_unknown_reals=["lmp_spread", "congestion_component"]
)
```

### 2. **Gradient Boosting (XGBoost/LightGBM/CatBoost)** ⭐ EASIEST TO START

**Why it's great:**
- Handles **any feature you can engineer**
- Weather, outages, constraints all become columns
- Fast training and interpretable
- Feature importance tells you what matters

**Feature engineering example:**
```python
def engineer_electricity_features(df):
    # Weather features
    df['temp_deviation'] = df['temperature'] - df['temperature'].rolling(168).mean()
    df['cooling_degree_hours'] = np.maximum(df['temperature'] - 65, 0)
    df['heating_degree_hours'] = np.maximum(65 - df['temperature'], 0)

    # Weather forecast error (important for RT prices)
    df['temp_forecast_error'] = df['temperature_actual'] - df['temperature_forecast']

    # Renewable generation weather sensitivity
    df['wind_generation_potential'] = wind_power_curve(df['wind_speed'])
    df['solar_generation_potential'] = solar_irradiance(df['solar_radiation'])

    # Weather extremes (spike drivers)
    df['heat_wave'] = (df['temperature'] > df['temperature'].rolling(168).quantile(0.95)).astype(int)

    # Outage features (you would add these)
    df['planned_outage_mw'] = ...
    df['forced_outage_mw'] = ...
    df['transmission_derates'] = ...
```

### 3. **LSTM/GRU with External Inputs**

**How it works:**
- Concatenate external features with price history
- Feed as multivariate input sequence

**Example:**
```python
model = LMPSpreadLSTM(
    input_size=20,  # Features: historical prices, weather, load, outages, etc.
    hidden_size=128,
    num_layers=3,
    output_size=24  # 24-hour ahead spread forecast
)
```

### 4. **Autoformer** (HuggingFace Available)

**Capabilities:**
- Handles multivariate time series
- State-of-the-art for energy forecasting (38% improvement)
- Can include weather and operational data
- Auto-correlation mechanism for capturing seasonal patterns

### 5. **TCN (Temporal Convolutional Networks)**

**Advantages:**
- Multivariate input support
- More efficient than LSTM (5-7.5% better RMSE)
- Good for capturing patterns with external variables
- Parallel processing capability

## Models That DON'T Handle External Features Well

❌ **N-BEATS** - Pure deep learning, univariate only, no feature engineering

❌ **Basic Chronos models** - Designed for univariate time series

## Recommendation for Virtual Trading in MISO/PJM/SPP

For virtual trading with weather and outage data:

### **Start with: LightGBM or XGBoost**

**Reasons:**
- Easiest to add features (weather, outages, constraints)
- Fast iteration on feature engineering
- Interpretable (see which features matter)
- Proven in power markets
- Lower computational requirements

**Implementation:**
```python
import lightgbm as lgb

# Train LightGBM for DA-RT spread prediction
features = engineer_electricity_features(df)
X = features[feature_columns]
y = features['spread_next_hour']

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    importance_type='gain'
)

lgb_model.fit(X, y,
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=50,
             verbose=100)
```

### **Upgrade to: Temporal Fusion Transformer (TFT)**

**When to upgrade:**
- You have complex feature interactions
- Need uncertainty quantification
- Want interpretability (attention weights)
- Have sufficient compute resources (GPU recommended)

**Advantages over gradient boosting:**
- Better captures temporal dependencies
- Provides probabilistic forecasts
- Built-in feature selection
- Handles missing data better

## Critical Feature Categories to Include

### 1. **Weather Features**
- Temperature (actual and forecast)
- Wind speed
- Solar radiation
- Humidity
- Heating/cooling degree hours

### 2. **Weather Forecasts**
- Day-ahead temperature predictions
- Wind speed forecasts
- Solar generation forecasts
- Forecast errors (actual vs predicted)

### 3. **Outage Information**
- Planned generation outages (MW)
- Forced generation outages (MW)
- Transmission derates
- Constraint limits
- Historical binding frequency

### 4. **Transmission & Congestion**
- Constraint limits
- Historical binding frequency
- Shadow prices
- Transmission flow limits
- Interface limits

### 5. **Load Data**
- Actual load
- Load forecast
- Forecast error (actual - forecast)
- Load deviation from average

### 6. **Temporal Features**
- Hour of day
- Day of week
- Month
- Holidays
- Peak/off-peak indicators
- Weekend flags

### 7. **Price & Spread Features**
- Historical LMP (lag features)
- DA-RT spread history
- Rolling statistics (mean, std, max)
- Congestion component
- Loss component

## Data Sources for External Features

### Weather Data
- **NOAA API**: Free, historical and forecast weather data
- **Open-Meteo**: Free weather API, good coverage
- **Weather Underground**: Comprehensive historical data
- **ISO websites**: Many ISOs publish weather data

### Outage Data
- **PJM**: Generator Availability Data System (GADS)
- **MISO**: Market Reports - Outage data
- **SPP**: Integrated Marketplace - Outage schedules
- **EIA Form 860M**: Monthly generator outages

### Load Data
- Available directly from ISO market data portals
- PJM Data Miner 2
- MISO Market Reports
- SPP Marketplace

## Next Steps

To implement a model with external features:

1. **Data Collection**: Set up pipelines for weather, outages, load data
2. **Feature Engineering**: Create derived features (degree hours, forecast errors)
3. **Model Selection**: Start with LightGBM for rapid iteration
4. **Validation**: Use walk-forward validation (not random splits)
5. **Feature Importance**: Analyze which features drive predictions
6. **Refinement**: Add/remove features based on importance
7. **Upgrade**: Move to TFT if you need better temporal modeling

## Practical Considerations

**Feature Availability:**
- Ensure features are available at prediction time
- Weather forecasts: Use day-ahead forecasts available at bid time
- Outages: Use planned outages published 7+ days ahead
- Don't use "future" data that wouldn't be available in real trading

**Feature Quality:**
- Weather forecast errors are often more important than forecasts themselves
- Planned outages more predictive than forced outages (unpredictable)
- Recent data (last 24-168 hours) often most relevant

**Computational Trade-offs:**
- XGBoost/LightGBM: Fast, CPU-only, real-time inference
- TFT: Slower, needs GPU, richer temporal modeling
- LSTM/GRU: Middle ground, moderate compute needs

**Model Refresh:**
- Retrain regularly (weekly/monthly) as market conditions change
- Monitor performance degradation
- Update feature engineering as new data sources become available
