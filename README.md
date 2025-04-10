# transient_localisation

**Transient Localisation** is a machine learning project focused on detecting and localizing failures in power systems by analyzing transient harmonic responses. This repository implements several deep neural network approaches to identify failures by evaluating the relative importance of different harmonic orders.

It is going to be published during the SDEWES '25 conference as an archival paper.

## Overview

Modern power systems and grid-connected electronics can exhibit subtle changes in harmonic content when faults occur. By analyzing transient behaviors—especially in the harmonics—this project aims to identify failures locations even in scenarios with limited observation windows. 

## Key Features

- **Harmonic Failure Analysis:**  
  Evaluates multiple harmonic orders (from 50Hz to 750Hz) to determine which are most effective for fault localization.

- **Deep Neural Network:**  
  Builds and trains specialised neural networks to classify failure conditions based on transient harmonic data.

- **Data Handling and Visualization:**  
  Includes modules for loading and preprocessing transient harmonic datasets, as well as visualising the results.

### Create a virtual environment and install the requirements:
```bash
pip install -r requirements.txt
```
