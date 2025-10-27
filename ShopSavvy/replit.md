# FIFA World Cup 2026 Finalist Predictor

## Overview

This is a machine learning application that predicts which teams will reach the FIFA World Cup 2026 finals. The system uses historical tournament data and implements dual classification models (Logistic Regression and Random Forest) to forecast finalist probabilities. The application features a custom web scraper for FIFA statistics, a complete data pipeline from collection to prediction, and an interactive Streamlit interface for visualization and predictions.

The project demonstrates end-to-end ML workflow including data collection, preprocessing, model training, evaluation, feature importance analysis, and deployment through a web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Streamlit Web Application**
- Single-page application structure with wide layout
- Custom CSS styling for headers, metrics, and visual components
- Interactive visualizations using Plotly and Matplotlib/Seaborn
- Sidebar navigation for different tasks and features
- Rationale: Streamlit chosen for rapid prototyping and native Python integration, eliminating need for separate frontend framework

### Backend Architecture

**Core ML Pipeline**
- Scikit-learn based classification system with two models:
  - Logistic Regression: Baseline linear model for binary classification
  - Random Forest Classifier: Ensemble model for complex pattern recognition
- StandardScaler for feature normalization
- Train/test split with cross-validation (KFold)
- GridSearchCV for hyperparameter optimization
- Rationale: Dual-model approach provides comparison between linear and ensemble methods, standard practice for classification tasks

**Data Processing Components**
1. **Web Scraper** (`web_scraper.py`): 
   - Uses Trafilatura for content extraction
   - BeautifulSoup for HTML parsing
   - Requests library with custom headers for HTTP operations
   - Targets FIFA rankings and team statistics
   - Rationale: Custom scraper provides control over data sources and update frequency

2. **Data Generator** (`data_generator.py`):
   - Creates synthetic historical FIFA data for training
   - Simulates realistic team performance metrics
   - Handles multiple confederations (UEFA, CONMEBOL, CONCACAF, AFC, CAF)
   - Generates features: goals, rankings, player age/experience
   - Rationale: Synthetic data generation allows for controlled experimentation and fills gaps in historical data

**Feature Engineering Strategy**
- Target variable: Binary classification (Finalist=1, Not_Finalist=0)
- Numeric features: Goal difference, FIFA rankings, average player age
- Categorical encoding: Team confederations using OneHotEncoder
- Performance metrics: Wins, draws, losses, goals scored/conceded
- Rationale: Domain-specific features based on football analytics best practices

### Evaluation Framework

**Multi-Metric Assessment**
- Accuracy, Precision, Recall, F1-Score for classification performance
- ROC-AUC for probability calibration
- Confusion Matrix for error analysis
- Cross-validation scores for generalization assessment
- Feature importance analysis for interpretability
- Rationale: Comprehensive metrics required for academic assignment (Assignment 2 learning outcomes CO1-CO4)

### Application Structure

**Modular Design**
- `app.py`: Main Streamlit application with UI logic
- `web_scraper.py`: Data collection layer
- `data_generator.py`: Synthetic data creation
- `main.py`: Entry point (minimal, likely for alternative execution)
- Rationale: Separation of concerns allows independent development and testing of components

**Data Flow**
1. Historical data generation/scraping
2. Feature engineering and preprocessing
3. Model training with cross-validation
4. Evaluation with multiple metrics
5. 2026 predictions with probability scores
6. Interactive visualization and reporting

## External Dependencies

### Machine Learning & Data Science
- **scikit-learn**: Core ML framework for models, preprocessing, and evaluation
- **pandas**: Data manipulation and DataFrame operations
- **numpy**: Numerical computations and array operations
- **matplotlib**: Static plotting and visualizations
- **seaborn**: Statistical visualizations built on matplotlib
- **plotly**: Interactive charts and graphs

### Web Framework
- **streamlit**: Web application framework for ML model deployment

### Web Scraping
- **trafilatura**: Main content extraction from websites
- **beautifulsoup4**: HTML/XML parsing
- **requests**: HTTP library for web requests

### Data Sources
- FIFA rankings from web sources (likely Wikipedia based on scraper implementation)
- Historical World Cup data (2006, 2010, 2014, 2018, 2022)
- Team statistics from various sports websites
- Note: Current implementation uses synthetic data generation as primary source

### Configuration
- Python 3.8+ required
- No database currently configured (data stored as CSV files)
- No authentication system implemented
- No external API integrations beyond web scraping