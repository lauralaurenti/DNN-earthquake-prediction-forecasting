# DNN-earthquake-prediction-forecasting

## Deep Learning For Predicting Laboratory Earthquakes and Autoregressively Forecasting Fault Zone Stress State 

#### Laurenti Laura, Tinti Elisa, Galasso Fabio, Franco Luca, Marone Chris
Sapienza University of Rome (IT) <br/>

## Abstract

Earthquake forecasting and prediction have long and in some cases sordid histories but recent work has rekindled interest based on advances in early warning, hazard assessment for induced seismicity and successful prediction of laboratory earthquakes. In the lab, frictional stick-slip events provide an analog for earthquakes and the seismic cycle. Labquakes are also ideal targets for machine learning (ML) because they can be produced in long sequences under controlled conditions. Indeed, recent works show that ML can predict several aspects of labquakes using fault zone acoustic emissions (AE). Here, we generalize these results and explore deep learning (DL) methods for labquake prediction and autoregressive (AR) forecasting. The AR methods allow forecasting at future horizons via iterative predictions. We address questions of whether DL methods can outperform existing ML models, including prediction based on limited training or forecasts beyond a single seismic cycle for aperiodic failure. We describe significant improvements to existing methods of labquake prediction. We demonstrate: 1) that DL models based on Long-Short Term Memory and Convolution Neural Networks predict labquakes under several conditions, including pre-seismic creep, aperiodic events and alternating slow/fast events and 2) that fault zone stress can be predicted with fidelity, confirming that acoustic energy is a fingerprint of fault zone stress. We predict also time to start of failure (TTsF) and time to the end of Failure (TTeF) for labquakes. Interestingly, TTeF is successfully predicted in all seismic cycles, while the TTsF prediction varies with the amount of preseismic fault creep. We report AR methods to forecast the evolution of fault stress using three sequence modeling frameworks: LSTM, Temporal Convolution Network and Transformer Network. AR forecasting is distinct from existing predictive models, which predict only a target variable at a specific time. The results for forecasting beyond a single seismic cycle are limited but encouraging. Our ML/DL models outperform the state-of-the-art and our autoregressive model represents a novel framework that could enhance current methods of earthquake forecasting.



## Code information

Prediction (section 4.1 of the paper): 
- The notebook generic_prediction.ipynb uses .csv files that are in the same directory  “prediction”, they are preprocessed  according to Hulbert [Rouet-Leduc, B., Hulbert, C., Lubbers, N., Barros, K., Humphreys, C.J., Johnson, P.A., 2017. Machine learning predicts laboratory earthquakes. Geophysical Research Letters 44, 9276–9282. doi:https://doi.org/10.1002/2017GL074677] procedure. The code allows the user to choose the experiment of interest and the target. It is also possible to choose 2 targets at the same time. 

Forecasting (section 4.2  of the paper): 
- transformer folder comes directly  from https://github.com/FGiuliari/Trajectory-Transformer/tree/master/transformer [Giuliari, F., Hasan, I., Cristani, M., Galasso, F., 2020. Transformer networks for trajectory forecasting. arXiv:2003.08111]. 
Data can be downloaded from http://psudata.s3-website.us-east-2.amazonaws.com/. Only the data about shear stress is utilized (e.g. "p4581_AE.mat" file for experiment p4581).
File “utils.py” contains functions to preprocess data and split into train/validation/test datasets. It also includes the classes for the models and functions used by them. The notebook generic_forecasting.ipynb allows the user to choose the experiment and the model of interest. Files in .pt format called "TF_pretrained_bestmodel_" are the model used by the pretrained transformer



