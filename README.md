# Health Terms Classification

This project classifies medical and health-related terms into one of **39 predefined categories** for a health directory, such as *Diabetes*, *Mental Health and Wellness*, and *Asthma*. This was my first practical exercise in learning supervised machine learning.

The model uses both the **Term Name** and **Definition** to predict the correct category. While the accuracy achieved is not exemplary as a consequence of insufficient training data, the model still produced great results classifying 10K+ health directory terms for a project at my internship. 

## How It Works
- **Data Processing:**  
  - Combined Term Name + Definition into a single text feature  
  - Lowercased text and removed punctuation/non-alphanumeric characters
- **Model:**  
  - **TF-IDF Vectorizer** to convert text into numerical features  
  - **LinearSVC** classifier with `class_weight="balanced"` to handle class imbalance
- **Libraries:**  
  - `pandas` for data handling  
  - `scikit-learn` for ML pipeline, vectorization, and classification  
  - `tkinter` for GUI interface

## Training & Evaluation
- **Train/Test Split:** 80/20 with stratification
- **Accuracy:** ~58.65%  
- **Reason for Moderate Accuracy:**  
  - Significant class imbalance (some categories have <10 samples)  
  - Overlapping terminology between categories

## Demo
<p align="center">
  <img src="https://github.com/m-aziz1/Healthcare-Terms-Classification/blob/main/assets/health-terms-classifier.gif" alt="Health Term Classifer demo" />
</p>

## Features
- **GUI** built with Tkinter for selecting datasets, training, and making predictions
- Outputs predictions to CSV with updated `Suggested TOPIC_DESC` column

## Future Improvements
- Collect more balanced training data
- Experiment with ensemble methods or deep learning approaches
