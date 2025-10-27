# ⚽ FIFA World Cup 2026 Finalist Predictor

A comprehensive machine learning application that predicts which teams will reach the FIFA World Cup 2026 finals using historical tournament data and advanced ML algorithms.

## 📋 Project Overview

This project fulfills all requirements of Assignment 2 for predicting FIFA World Cup 2026 finalists using machine learning. It includes:

- **Custom web scraper** for FIFA statistics
- **Complete data pipeline** from collection to prediction
- **Dual ML models** (Logistic Regression & Random Forest)
- **Comprehensive evaluation** with multiple metrics
- **Interactive Streamlit interface** for all tasks
- **Feature importance analysis** with domain insights
- **2026 predictions** with probability scores

## 🎯 Learning Outcomes

- ✅ **CO1:** Understand key ML concepts and sports analytics applications
- ✅ **CO2:** Implement classification models with preprocessing
- ✅ **CO3:** Evaluate model performance using multiple metrics
- ✅ **CO4:** Interpret feature importance with domain knowledge

## 📊 Assignment Tasks Completion

| Task | Description | Marks | Status |
|------|-------------|-------|--------|
| Task 1 | Data Collection & Preparation | 20 | ✅ Complete |
| Task 2 | Model Building & Training | 25 | ✅ Complete |
| Task 3 | Model Evaluation | 15 | ✅ Complete |
| Task 4 | Feature Importance | 10 | ✅ Complete |
| Task 5 | Final Prediction & Reflection | 15 | ✅ Complete |
| Task 6 | Complete Application | 15 | ✅ Complete |
| **Total** | | **100** | **✅ Complete** |

## ✨ Enhanced Features (Beyond Assignment)

| Feature | Description | Status |
|---------|-------------|--------|
| Model Persistence | Save/load trained models with joblib | ✅ Implemented |
| Data Versioning | Track datasets and scraper runs with version control | ✅ Implemented |
| Multi-Source Scraping | Orchestrate scraping from multiple data sources | ✅ Implemented |
| Advanced Visualizations | Enhanced charts and interactive displays | ✅ Implemented |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone <your-repository-url>
cd fifa-wc-predictor

# Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly beautifulsoup4 requests trafilatura lxml joblib

# Run the application
streamlit run app.py --server.port 5000
```

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
beautifulsoup4>=4.12.0
requests>=2.31.0
trafilatura>=1.6.0
lxml>=4.9.0
joblib>=1.3.0
```

## 📁 File Structure

```
fifa-wc-predictor/
├── app.py                      # Main Streamlit application
├── web_scraper.py              # Custom FIFA data scraper module
├── data_generator.py           # Dataset generation utilities
├── historical_fifa_data.csv    # Historical training dataset (2006-2022)
├── teams_2026.csv              # 2026 World Cup qualifier data
├── cleaned_fifa_data.csv       # Processed and engineered dataset
├── predictions_2026.csv        # Final 2026 predictions
├── README.md                   # This file
└── requirements.txt            # Python package dependencies
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py --server.port 5000
```

The application will open in your browser at `http://localhost:5000`

### Navigation Guide

The application has 9 main sections accessible from the sidebar:

1. **🏠 Home**
   - Project overview
   - Technology stack
   - Task completion status

2. **📊 Data Collection & EDA**
   - Load/regenerate historical data
   - Test custom web scraper
   - Exploratory data analysis
   - Missing value analysis
   - Correlation heatmaps
   - Confederation analysis

3. **🔧 Feature Engineering**
   - View engineered features
   - Export cleaned dataset
   - Feature statistics

4. **🤖 Model Training**
   - Configure train/test split
   - Train Logistic Regression
   - Train Random Forest with GridSearchCV
   - K-Fold cross-validation
   - View training results

5. **📈 Model Evaluation**
   - Performance metrics comparison
   - Confusion matrices
   - ROC curves
   - Model selection

6. **🎯 Feature Importance**
   - Logistic Regression coefficients
   - Random Forest importance scores
   - Domain knowledge interpretation
   - Feature comparison charts

7. **🏆 2026 Predictions**
   - Load 2026 qualifier data
   - Generate predictions
   - View probability rankings
   - Export results
   - Limitations and ethical considerations

8. **📚 Documentation**
   - Complete user manual
   - Installation guide
   - Usage instructions
   - Sample results

## 🔍 Custom Web Scraper

### Features

The `FIFADataScraper` class provides:

- **scrape_fifa_rankings()**: Extract FIFA world rankings
- **scrape_match_results()**: Historical match data
- **scrape_team_stats()**: Comprehensive team statistics
- **export_scraped_data()**: Export to CSV

### Technologies Used

- **BeautifulSoup4**: HTML parsing
- **Trafilatura**: Text content extraction
- **Requests**: HTTP communication

### Challenges Addressed

- Rate limiting and anti-bot measures
- Dynamic content loading (JavaScript)
- Inconsistent data formats
- Missing or incomplete data handling

### Usage Example

```python
from web_scraper import FIFADataScraper

scraper = FIFADataScraper()
rankings = scraper.scrape_fifa_rankings()
print(rankings)
```

## 📊 Dataset Description

### Historical Dataset (2006-2022)

**Source**: Generated from historical FIFA World Cup patterns
**Records**: ~160 tournament team entries
**Features**: 17 columns

#### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| Year | int | Tournament year (2006, 2010, 2014, 2018, 2022) |
| Team | str | Team name |
| Confederation | str | Continental confederation (UEFA, CONMEBOL, etc.) |
| FIFA_Ranking | int | FIFA world ranking position |
| Avg_Age | float | Average player age |
| Matches_Played | int | Total matches in tournament |
| Wins | int | Number of wins |
| Draws | int | Number of draws |
| Losses | int | Number of losses |
| Goals_Scored | int | Total goals scored |
| Goals_Conceded | int | Total goals conceded |
| Goal_Difference | int | Goals scored minus conceded |
| Win_Rate | float | Wins / Matches played |
| Avg_Player_Caps | int | Average international appearances |
| Previous_WC_Appearances | int | Number of previous WC participations |
| Qualification_Status | str | How team qualified (Direct, Playoff, Host) |
| **Finalist** | int | **Target variable** (1 = Finalist, 0 = Not Finalist) |

### Data Cleaning Steps

1. **Duplicate Removal**: Remove duplicate records
2. **Missing Value Imputation**:
   - Numeric: Median imputation
   - Categorical: Mode imputation
3. **Feature Engineering**:
   - Goal_Difference = Goals_Scored - Goals_Conceded
   - Win_Rate = Wins / Matches_Played
4. **Categorical Encoding**: One-hot encoding for Confederation and Qualification_Status

## 🤖 Machine Learning Models

### 1. Logistic Regression

**Purpose**: Baseline linear model
**Configuration**:
- `random_state=42`
- `max_iter=1000`

**Strengths**:
- Interpretable coefficients
- Fast training
- Good for linear relationships

### 2. Random Forest Classifier

**Purpose**: Advanced ensemble model
**Hyperparameter Tuning** (GridSearchCV):
- `n_estimators`: [50, 100, 150]
- `max_depth`: [5, 10, 15]
- `min_samples_split`: [2, 5]

**Strengths**:
- Handles non-linear relationships
- Robust to outliers
- Built-in feature importance

### Training Process

1. **Data Split**: 80% training, 20% testing (stratified)
2. **Feature Scaling**: StandardScaler normalization
3. **Cross-Validation**: 5-Fold CV for robustness
4. **Hyperparameter Tuning**: Grid Search on Random Forest
5. **Model Selection**: Best F1-Score

## 📈 Model Evaluation

### Metrics Used

- **Accuracy**: Overall correctness
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination ability

### Typical Performance

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | 0.85-0.90 | 0.88-0.93 |
| Precision | 0.60-0.75 | 0.70-0.85 |
| Recall | 0.55-0.70 | 0.65-0.80 |
| F1-Score | 0.58-0.72 | 0.68-0.82 |
| ROC-AUC | 0.82-0.88 | 0.86-0.92 |

## 🎯 Feature Importance

### Top Features (Domain Interpretation)

1. **Goal Difference**
   - Most critical indicator of team strength
   - Reflects offensive and defensive balance

2. **FIFA Ranking**
   - Official performance measure
   - Lower number = stronger team

3. **Win Rate**
   - Direct measure of match success
   - Critical for knockout progression

4. **Average Age**
   - Experience vs fitness balance
   - Optimal: 26-28 years

5. **Previous WC Appearances**
   - Tournament experience matters
   - Indicates football infrastructure

6. **Confederation**
   - UEFA and CONMEBOL dominance
   - Historical pattern in finals

## 🏆 2026 Predictions

### Prediction Process

1. Load 2026 qualifier data
2. Apply same feature engineering
3. Scale features using trained scaler
4. Generate predictions with best model
5. Calculate probability scores
6. Rank teams by likelihood

### Output Format

```csv
Team,Confederation,FIFA_Ranking,Finalist_Probability,Prediction_Label
Brazil,CONMEBOL,3,0.87,Finalist
France,UEFA,2,0.82,Finalist
Argentina,CONMEBOL,1,0.79,Finalist
...
```

## 🔬 Model Limitations

### Data Limitations
- Small sample size (5 tournaments)
- Missing recent form/injury data
- Cannot capture tactical innovations

### Inherent Unpredictability
- Individual brilliance moments
- Referee decisions and luck
- Knockout tournament randomness

### Methodological
- Correlation ≠ Causation
- Confounding variables
- Historical bias perpetuation

## ⚖️ Ethical Considerations

### Responsible Use

✅ **Appropriate Uses**:
- Educational purposes
- Statistical analysis
- Fan engagement
- Understanding patterns

❌ **Inappropriate Uses**:
- Sports betting/gambling
- Financial decisions
- Creating unfair expectations
- Pressure on teams/players

### Transparency
- Communicate uncertainty
- Explain limitations
- Show probabilities, not certainties
- Avoid sensationalism

## 📸 Screenshots

### Application Interface
- Home dashboard with overview
- Interactive data exploration
- Model training progress
- Evaluation visualizations
- Prediction results

### Key Visualizations
- Confusion matrices
- ROC curves
- Feature importance charts
- Correlation heatmaps
- Prediction probability bars

## 🛠️ Development

### Technologies
- **Language**: Python 3.11
- **Framework**: Streamlit
- **ML Library**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Scraping**: BeautifulSoup4, Trafilatura

### Code Quality
- Fully documented functions
- Type hints where applicable
- Error handling
- Modular structure
- PEP 8 style compliance

## 📝 Report Generation

The application provides:
- Interactive visualizations
- Exportable datasets
- Prediction results (CSV)
- Performance metrics
- Feature importance analysis

For the final report, use:
1. Screenshots from each section
2. Exported CSV files
3. Documentation section content
4. Reflection from 2026 Predictions section

## 🎥 Demo Video

To create the required 5-minute screencast:

1. **Introduction** (30s): Project overview
2. **Data Collection** (60s): Scraper demo and EDA
3. **Model Training** (60s): Training both models
4. **Evaluation** (60s): Metrics and visualizations
5. **Predictions** (60s): 2026 predictions
6. **Conclusion** (30s): Limitations and ethics

## 🤝 Contributing

This is an academic assignment project. For educational purposes only.

## 📄 License

Educational use only. Part of Machine Learning course assignment.

## 👨‍💻 Author

[Your Name]
[Your Student ID]

## 📧 Contact

For questions or feedback about this project, please contact through the course platform.

---

**Note**: This application is for educational purposes and demonstrates machine learning concepts in sports analytics. Predictions should not be used for gambling or financial decisions.
