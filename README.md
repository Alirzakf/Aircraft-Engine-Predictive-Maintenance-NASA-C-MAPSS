# Aircraft Engine Predictive Maintenance (NASA C-MAPSS)

This project focuses on predictive maintenance for aircraft engines using the NASA C-MAPSS dataset (FD001). The goal is to analyze sensor data over time and predict engine degradation and remaining useful life (RUL).

---

## Objective

The main objective is to develop a data-driven approach for:

- Understanding engine degradation patterns
- Performing time-series exploratory data analysis
- Engineering features from sensor signals
- Building predictive models for Remaining Useful Life (RUL)

---

## Dataset

NASA C-MAPSS FD001 dataset is used in this project.

It contains:
- Multivariate time-series sensor data
- Engine operational cycles until failure
- Multiple sensor measurements per cycle

---

## Project Structure

data/ → raw and processed datasets  
notebooks/ → exploratory analysis and experiments  
src/ → reusable Python modules  
outputs/ → visualizations and reports  

---

## Workflow

1. Data loading and inspection  
2. Exploratory Data Analysis (EDA)  
3. Data preprocessing and cleaning  
4. Feature engineering  
5. Model development  
6. Evaluation and interpretation  

---

## Requirments

### macOS requirement for XGBoost

XGBoost requires the OpenMP runtime on macOS.

Install it with Homebrew:

```bash
brew install libomp

---

## Status

This project is actively developed as a learning project.