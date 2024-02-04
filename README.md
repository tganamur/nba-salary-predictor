![Result](validation.png?raw=true "Title")

This project leverages data analytics on a comprehensive NBA dataset (2006-2023) to develop a predictive model for player salaries. We use various regression methods and machine learning algorithms to identify critical factors influencing salaries and quantify their impact. The goal is to create a reliable predictive model for NBA franchises, analysts, and enthusiasts, offering insights into on-court performance and player characteristics translating into monetary value. This project, at the intersection of sports analytics and economic forecasting, enhances strategic planning in professional basketball, aiding decisions on player acquisitions, contract renewals, and team budgeting.

The data was split into 80/20 training and testing sets. Y-values were player salaries, and X-values were all other dataset features. Training data (both x and y) were normalized, with x using StandardScaler and y by mean and standard deviation. After preprocessing, models were created based on data and parameters chosen via grid search. Models were trained, predicted by values computed using .predict, and metrics (RMSE and R2) calculated using sklearn. The chosen models were Classic Linear Regression, Random Forests, Gradient Boosting, and Neural Networks.

The following is the methodology used for Random Forests and Neural Networks: 

![Feature Importance](feature_importance.png?raw=true "Title")

Random Forests

The global methodology, extended to assess Random Forest Regressor, optimized parameters via grid search, configuring it with 1000 estimators. For decision trees, max_depth was optimized to 5. See sections 2.5 and 2.6 of code for details.

1. Flexibility:

Random Forest: Excels with an ensemble of decision trees for complex data.

Decision Trees: Capture non-linear patterns but risk overfitting.

2. Ensemble Strategy:

Random Forest: Averages predictions from independent trees, reducing overfitting.

Decision Trees: Make predictions independently, susceptible to overfitting in complex datasets.

3. Handling Non-linearity:

Random Forest: Robustly captures non-linear relationships, reducing overfitting.

Decision Trees: Capture non-linear patterns but may struggle with complexity and overfitting.

4. Model Combination:

Random Forest: Combines trees for a more robust prediction.

Decision Trees: Make predictions independently without ensemble averaging.

Neural Networks

We tried different neural network architectures for creating a model for our dataset, with different number of layers, number of neurons per layer, and different activation functions for those neurons. We dove deep to find which option makes a better fit. After iterating on the architecture of the model, we decided to retain the following architecture, which gave good results while limiting overfitting.

Input and Output Layers:

The input layer has 54 neurons, indicating the model receives 54 features/statistics related to NBA players that will be used for predicting salaries.

The output layer consists of a single neuron since the objective is to predict a continuous value (salary).

Hidden Layers:

The architecture consists of three hidden layers with 20, 15, and 10 neurons, respectively, each employing the ELU (Exponential Linear Unit) activation function.

The ELU activation function helps mitigate the vanishing gradient problem by allowing negative inputs, potentially aiding in faster convergence during training.

Dropout Layers:

Two dropout layers with dropout rates of 0.25 are included after the second and third hidden layers. Dropout helps prevent overfitting by randomly dropping a proportion of neurons during training, forcing the network to learn more robust features.

Kernel Initialization:

The choice of 'normal' as the kernel initializer initializes the weights of the network's layers following a normal distribution is a common practice to ensure that weights start at reasonable values.

Adam optimizer choice rationale:

Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that is well-suited for a wide range of problems.

Adam adjusts the learning rates for each parameter, making it less sensitive to the initial learning rate and generally requiring less hyperparameter tuning compared to other optimizers.

![Neural Network Validation](nn_validation.png?raw=true "Title")
