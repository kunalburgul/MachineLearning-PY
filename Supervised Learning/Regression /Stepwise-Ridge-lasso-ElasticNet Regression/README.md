## **Stepwise Regression**

- This form of regression is used when we deal with multiple independent variables. In this technique, the selection of independent variables is done with the help of an automatic process, which involves no human intervention.

- This feat is achieved by observing statistical values like R-square, t-stats and AIC metric to discern significant variables. 

- Stepwise regression basically fits the regression model by adding/dropping co-variates one at a time based on a specified criterion. 

- Some of the most commonly used Stepwise regression methods are listed below:

          - Standard stepwise regression does two things. It adds and removes predictors as needed for each step.
          - Forward selection starts with most significant predictor in the model and adds variable for each step.
          - Backward elimination starts with all predictors in the model and removes the least significant variable for each step.
          - The aim of this modeling technique is to maximize the prediction power with minimum number of predictor variables. It is one of the method to handle higher dimensionality of data set.


## **Ridge Regression**

- Ridge Regression is a technique used when the data suffers from multicollinearity ( independent variables are highly correlated). In multicollinearity, even though the least squares estimates (OLS) are unbiased, their variances are large which deviates the observed value far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.

- A Linear equation y=a+ b*x also has an error term. The complete equation becomes:

          y=a+b*x+e (error term),  [error term is the value needed to correct for a prediction error between the observed and predicted value]

- y=a+y= a+ b1x1+ b2x2+....+e, for multiple independent variables.
In a linear equation, prediction errors can be decomposed into two sub components. First is due to the biased and second is due to the variance. Prediction error can occur due to any one of these two or both components. Here, we’ll discuss about the error caused due to variance.

- Ridge regression solves the multicollinearity problem through shrinkage parameter λ (lambda)

### Important Points :

- The assumptions of this regression is same as least squared regression except normality is not to be assumed
- It shrinks the value of coefficients but doesn’t reaches zero, which suggests no feature selection feature
- This is a regularization method and uses l2 regularization.



## **Lasso Regression**

- Similar to Ridge Regression, Lasso (Least Absolute Shrinkage and Selection Operator) also penalizes the absolute size of the regression coefficients. In addition, it is capable of reducing the variability and improving the accuracy of linear regression models.  

- Lasso regression differs from ridge regression in a way that it uses absolute values in the penalty function, instead of squares. This leads to penalizing (or equivalently constraining the sum of the absolute values of the estimates) values which causes some of the parameter estimates to turn out exactly zero. Larger the penalty applied, further the estimates get shrunk towards absolute zero. This results to variable selection out of given n variables.

### Important Points:

- The assumptions of this regression is same as least squared regression except normality is not to be assumed
- It shrinks coefficients to zero (exactly zero), which certainly helps in feature selection
- This is a regularization method and uses l1 regularization
- If group of predictors are highly correlated, lasso picks only one of them and shrinks the others to zero


## **ElasticNet Regression**

- ElasticNet is hybrid of Lasso and Ridge Regression techniques. It is trained with L1 and L2 prior as regularizer. Elastic-net is useful when there are multiple features which are correlated. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

- A practical advantage of trading-off between Lasso and Ridge is that, it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

### Important Points:

- It encourages group effect in case of highly correlated variables
- There are no limitations on the number of selected variables
- It can suffer with double shrinkage

### Beyond these most commonly used regression techniques, you can also look at other models like Bayesian, Ecological and Robust regression.

## **How to select the right regression model?**

- Life is usually simple, when you know only one or two techniques. One of the training institutes I know 
          – if the outcome is continuous apply linear regression. 
          - If it is binary – use logistic regression! 

- However, higher the number of options available at our disposal, more difficult it becomes to choose the right one. A similar case happens with regression models. Within multiple types of regression models, it is important to choose the best suited technique based on type of independent and dependent variables, dimensionality in the data and other essential characteristics of the data. Below are the key factors that you should practice to select the right regression model:

- Data exploration is an inevitable part of building predictive model. It should be you first step before selecting the right model like identify the relationship and impact of variables
- To compare the goodness of fit for different models, we can analyse different metrics like statistical significance of parameters, R-square, Adjusted r-square, AIC, BIC and error term. Another one is the Mallow’s Cp criterion. This essentially checks for possible bias in your model, by comparing the model with all possible submodels (or a careful selection of them).
- Cross-validation is the best way to evaluate models used for prediction. Here you divide your data set into two group (train and validate). A simple mean squared difference between the observed and predicted values give you a measure for the prediction accuracy.
- If your data set has multiple confounding variables, you should not choose automatic model selection method because you do not want to put these in a model at the same time.
- It’ll also depend on your objective. It can occur that a less powerful model is easy to implement as compared to a highly statistically significant model.
Regression regularization methods(Lasso, Ridge and ElasticNet) works well in case of high dimensionality and multicollinearity among the variables in the data set.
