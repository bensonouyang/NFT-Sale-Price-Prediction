# Regression to Predict NFT Sale Price

## Table of Contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Files](#files)
* [Source](#source)

## Introduction
Non-fungible tokens (NFTs) are unique digital items that are bought and sold via cryptocurrency. The hype behind these NFTs is the insanely high amounts of Ethereum spent for them. Many celebrities and big brands have launched their own NFTs that many people are hoping to collect. The big question is what makes certain NFTs worth more than others. For our analysis, we gathered the NFT auction data and did some data cleaning. We also did some image and text processing for more features. Afterwards, we fit a linear model with stochastic gradient descent with different explanatory variables. By trying to improve the Mean Absolute Error, we compared some models with default parameters and ran a fivefold Cross Validation process. To further improve our analysis, we tried implementing a Deep Neural Network and a Convolutional Neural Network. With the Keras package, we were able to combine a Multilayer Perceptron Model with a Convolutional Neural Network to produce NFT auction price predictions.

## Technologies
Project is created with:
* Python

Report is made with:
* R Markdown
* Latex

## Libraries
* Pandas
* Numpy
* imageio
* PIL
* Tensorflow
* Keras
* Scikit-learn

## Files
To run the code and see results:
* Open [A10 - regression.ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/A10%20-%20regression.ipynb) in Jupyter Notebook. This file contains the process of using a linear regression model with scaled training data. 
* Open [B10 - feature engineering (images).ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/B10%20-%20feature%20engineering%20(images).ipynb) and [B10 - feature engineering (images)b.ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/B10%20-%20feature%20engineering%20(images)b.ipynb) in Jupyter Notebook. These two files contains process of importing the image data and some data cleaning. 
* Open [C10 - feature engineering (text).ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/C10%20-%20feature%20engineering%20(text).ipynb) and [C10 - feature engineering (text)b.ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/C10%20-%20feature%20engineering%20(text)b.ipynb) in Jupyter Notebook. These two files contains process of importing the text data and some data cleaning. 
* Open [E10 - regression with image and text-1.ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/E10%20-%20regression%20with%20image%20and%20text-1.ipynb) in Jupyter Notebook. This file contains process of training a few regression models with the combined original, image and text data. 
* Open [Z10-DNNMLP.ipynb](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/Z10-DNNMLP.ipynb) in Jupyter Notebook. This file contains the final model of a combined Convolutional Neural Network and a Multilayer Perceptron model.

To see the report:
* Open [440project2-report.pdf](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/report/440project2-report.pdf) for PDF version
* Open [440project2-report.Rmd](https://github.com/bensonouyang/NFT-Sale-Price-Prediction/blob/main/report/440project2-report.Rmd) to see how the report was made

## Source
[In class kaggle competition](https://www.kaggle.com/competitions/stat440-21-project2/overview)
