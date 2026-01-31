
# Portfolio Optimization & Hierarchical/Clustering Based Risk Parity Research

This repository contains my Honors Thesis library for Actuarial Science. It focuses on comparative portfolio optimization, specifically evaluating **Hierarchical Risk Parity (HRP)** against traditional and mu-free frameworks.

## Project Structure
* **`portafolios/`** : The core Python library containing logic for constructors, metrics, and HRP style optimization.
* **`test.ipynb`**: Interactive development environment used for library prototyping and testing.
* **`original/`**: Legacy versioning and early-stage experimental code.
* **`plots/`**: Storage for generated static and interactive visualizations (Work in Progress).

## Key Technical Features
* **Time Series Driven**: The engine is designed to ingest and process time series data for risk estimation.
* **HRP & Clustering**: Native implementation of recursive bisection and diverse distance metrics (De Prado, etc.).
* **Modular Design**: Separated concerns between data loading, portfolio construction, and evaluation.

##  Roadmap (Ongoing)
- [ ] **API Integration**: Transitioning from local loaders to real-time data extraction via financial APIs.
- [ ] **Plot Management**: Standardizing the export of Plotly/Matplotlib visuals to the `/plots` directory.
- [ ] **Backtesting Engine**: Finalizing the Monte Carlo and historical backtest modules for strategy validation.

###  Core Infrastructure
* **Optimization Engine:** Built on `SciPy.optimize` and `NumPy` for high-performance matrix calculations.
* **Data Engineering:** Custom-built data pipeline using `Pandas` for time-series cleaning and ingestion.
* **Visualization Suite:** Proprietary wrappers around `Plotly` and `Matplotlib` for specialized HRP analysis (Dendrograms, Distance Matrices, and Heatmaps).
* **Modular Architecture:** Fully decoupled system following Object-Oriented Programming (OOP) principles for scalability.
* **Environment:** Managed via `requirements.txt`.




```mermaid
classDiagram
%% Relaciones de Herencia
Portfolio <|-- Markowitz
Portfolio <|-- Naive
Portfolio <|-- NaiveRiskParity

    %% Definición de Clases
    class Portfolio{
        <<Core>>
        +weights
        +optimize()
    }
    class Markowitz{
        <<Constructor>>
        +efficient_frontier()
    }
    class Naive{
        <<Constructor>>
        +equal_weights()
    }
    class NaiveRiskParity{
        <<Constructor>>
        +risk_contribution()
    }

    %% Conexión con Datos y Plots (Estructural)
    Portfolio ..> Data_Loader : usa
    Portfolio ..> Plots : visualiza
