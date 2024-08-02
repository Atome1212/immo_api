
# 🏡 Immo API 🏡

Welcome to the Immo API project! This repository contains all the necessary code to load a pre-trained model and predict real estate prices based on various property features. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project aims to predict real estate prices using a pre-trained machine learning model. The code is structured to easily load the model, process input data, and provide price predictions via an API.

![Real Estate](https://miro.medium.com/v2/resize:fit:1000/1*1C3GnoY-FzhqzL0MzTlWyQ.gif)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone git@github.com:Atome1212/immo_api.git
cd immo_api
pip install -r requirements.txt
```

Download and put the model into `./data/model.pkl`:

```
https://drive.google.com/file/d/1fHsNmA9dE_eYHE7Ep70yhNxxVCO6yhQN/view?usp=sharing
```

## Usage

Follow these steps to set up the API and make predictions:

1. **Start the API**: Run the Streamlit app to start the prediction API.

```bash
streamlit run main.py
```

2. **Input Data**: Enter the property characteristics to get an estimated price via the Streamlit interface.

### Detailed Explanation

- **Prediction**:
   - Prepare new data based on user input.
   - Load the saved model and scaler.
   - Predict the price using the pre-trained model.

Creator Atome1212
