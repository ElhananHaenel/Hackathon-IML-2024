# Hackathon 2024 - Optimizing Public Transportation Routes


**HU.BER**

---

## Overview

Welcome to the GitHub repository for our project during Hackathon 2024, where we focused on optimizing public transportation routes in Israel using machine learning techniques. This project was undertaken as part of the Introduction to Machine Learning course (67577) at Hebrew University, under the guidance of Dr. Gabriel Stanovsky and Dr. Roy Schwartz, along with our dedicated TAs and Tzars.

## Motivation

The project addresses issues such as irregular bus schedules and overcrowding in public transportation systems. Our goal was to leverage machine learning to improve the reliability, efficiency, and user experience of public transit across Israel.

## Dataset

We analyzed `train_bus_schedule.csv`, a comprehensive dataset containing records of bus routes and relevant features at various stops in Israel. This dataset, with 226,112 entries, served as the foundation for developing our machine learning models.

For detailed descriptions of dataset features, please refer to `bus_column_description.md`.

## Tasks and Objectives

### Task 1: Predicting Passenger Boardings at Bus Stops

**Objective:** Predict the number of passengers boarding buses at specific stops.

- **Input:** `X_passengers_up.csv`
- **Output:** `y_passengers_up_predictions.csv`
- **Evaluation:** Mean Squared Error (MSE)

### Task 2: Predicting Trip Duration

**Objective:** Estimate the duration of bus trips from their first to last stops.

- **Input:** `X_trip_duration.csv`
- **Output:** `y_trip_duration_predictions.csv`
- **Evaluation:** Mean Squared Error (MSE)

### Task 3: Improving Public Transportation

**Objective:** Provide actionable insights and strategies based on our models' results.

- **Output:** `conclusions_and_suggestions.pdf`

## Files Provided

### Training Data

- `train_bus_schedule.csv`: Dataset used for training machine learning models.

### Test Sets

- `X_passengers_up.csv`: Test set for Task 1 predictions.
- `X_trip_duration.csv`: Test set for Task 2 predictions.

### Example Outputs

- `y_passengers_up_example.csv`: Example output format for Task 1 predictions.
- `y_trip_duration_example.csv`: Example output format for Task 2 predictions.

### Evaluation Scripts

- `eval_passengers_up.py`: Script for evaluating Task 1 predictions.
- `eval_trip_duration.py`: Script for evaluating Task 2 predictions.

## Implementation Tips

- Use `encoding="ISO-8859-8"` when loading or saving CSV files with Hebrew text fields.
- Consider time-series data techniques for analyzing sequences of bus stop events.

## Conclusion

Our participation in Hackathon 2024 enabled us to gain valuable insights into optimizing public transportation routes through machine learning. By analyzing data, building predictive models, and proposing practical improvements, we aim to contribute to a more efficient and reliable public transit system in Israel.

For detailed results and additional information, please explore the repository and accompanying documentation.
