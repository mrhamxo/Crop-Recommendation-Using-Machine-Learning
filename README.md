# Crop Recommendation Using Machine Learning

Precision agriculture is rapidly transforming farming practices by enabling farmers to make informed decisions about their farming strategies. This project leverages machine learning to recommend the most suitable crops to grow based on specific soil and weather conditions.

## Project Summary

This project builds a predictive model that recommends the most suitable crops for cultivation based on environmental and soil parameters. The dataset used includes data on nitrogen, phosphorus, and potassium content in the soil, temperature, humidity, pH, and rainfall. This approach aims to enhance crop yield, optimize resources, and contribute to sustainable agriculture.

This project involves multi-class classification, predicting which crop is most suitable based on soil and weather parameters. The dataset includes 22 different crops, each represented as a unique class. The crops are:

1. Rice
2. Maize
3. Jute
4. Cotton
5. Coconut
6. Papaya
7. Orange
8. Apple
9. Muskmelon
10. Watermelon
11. Grapes
12. Mango
13. Banana
14. Pomegranate
15. Lentil
16. Blackgram
17. Mungbean
18. Mothbeans
19. Pigeonpeas
20. Kidneybeans
21. Chickpea
22. Coffee

## Objectives

- To analyze various soil and environmental parameters to recommend the best crop for cultivation.
- To build multiple machine learning models and evaluate their performance for crop recommendation.
- To develop an easy-to-use web application for farmers using Streamlit.

## Project Workflow

[Dataset Link](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
1. **Data Collection and Preprocessing**: The dataset used was built by augmenting datasets of rainfall, climate, and fertilizer data.
   
2. **Feature Engineering**: Key features like nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall were analyzed and scaled for better model performance.
   - **Scaling**: Data was scaled using MinMaxScaler and StandardScaler to improve the performance of the models.
   - **Label Encoding**: Crop labels were encoded to numerical values for model compatibility.

3. **Model Training**:
   - The dataset was split into training and testing sets with an 80-20 ratio.
   - StandardScaler was applied to ensure that features were scaled appropriately.
   - Models were trained using default parameters and tuned to achieve optimal performance.

4. Several machine learning models were trained on the preprocessed data, including:
   - Logistic Regression
   - Gaussian Naive Bayes
   - Support Vector Classifier (SVC)
   - K-Nearest Neighbors
   - Decision Tree
   - Extra Trees Classifier
   - Random Forest
   - Bagging Classifier
   - Gradient Boosting Classifier
   - AdaBoost Classifier

5. **Model Evaluation**: Each model was evaluated based on accuracy on both training and test datasets to determine the best-performing model.

| **Model**                     | **Training Accuracy (%)** | **Test Accuracy (%)** | 
|-------------------------------|---------------------------|-----------------------|
| **Logistic Regression**       | 97.78                     | 96.36                 |
| **Gaussian Naive Bayes**      | 99.49                     | 99.54                 | 
| **Support Vector Classifier** | 98.81                     | 96.82                 | 
| **K-Nearest Neighbors**       | 98.69                     | 95.91                 | 
| **Decision Tree Classifier**  | 100.00                    | 98.18                 | 
| **Extra Tree Classifier**     | 100.00                    | 87.73                 | 
| **Random Forest Classifier**  | 100.00                    | 99.32                 | 
| **Bagging Classifier**        | 99.83                     | 98.64                 | 
| **Gradient Boosting Classifier** | 100.00                 | 98.18                 | 

6. **Algorithms and Accuracy**

The models were trained and tested with the following results:

- **Gaussian Naive Bayes**: Selected as the best model due to its balanced accuracy and performance on unseen data.
  
The models were evaluated using metrics like accuracy, precision, and recall, with Gaussian Naive Bayes achieving the highest performance on the test set.

7. **Deployment**: The best model, Gaussian Naive Bayes, was deployed using Streamlit to create a user-friendly interface for crop recommendation.

## GUI Interface Snapshot
![result1](https://github.com/user-attachments/assets/0ed30620-4179-490b-9038-26d612668812)

## Conclusion

The project successfully developed a machine learning model to recommend crops based on soil and weather conditions. The deployed model can guide farmers in making data-driven decisions, improving yield and sustainability in agriculture.

## Future Work

- Extend the model to include more crops and weather parameters.
- Integrate real-time data from IoT devices for dynamic crop recommendations.
- Implement advanced deep learning models to improve accuracy.
