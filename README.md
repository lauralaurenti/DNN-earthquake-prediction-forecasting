# DNN-earthquake-prediction-forecasting

## Deep Learning For Predicting Laboratory Earthquakes and Autoregressively Forecasting Fault Zone Stress State 

#### Laurenti Laura, Tinti Elisa, Galasso Fabio, Franco Luca, Marone Chris
Faculty of Information Engineering, Informatics, and Statistics. Sapienza University of Rome, Italy <br/>
Department of Computer Science. Sapienza University of Rome (IT) <br/>
Earth Science Department. Sapienza University of Rome (IT) <br/>
Earth Science Department. PennState University (PA) <br/>

## Abstract

Earthquake forecasting and prediction have long and in some cases sordid histories but recent work has rekindled interest based on advances in short-term early warning, hazard assessment for human induced seismicity and successful prediction of laboratory earthquakes. 
In the lab, frictional stick-slip events provide an analog for  tectonic earthquakes and the seismic cycle.
Lab earthquakes are also ideal targets for machine learning (ML) because they can be produced in long sequences under a wide range of controlled conditions. Indeed, recent works show that ML can predict several aspects of labquakes using fault zone acoustic emissions (AE). 
Here, we generalize these results and explore deep learning (DL) methods for labquake prediction and autoregressive (AR) forecasting. The AR methods are novel and allow forecasting at future horizons by proceeding via iterative predictions.
We address questions of whether DL methods can outperform existing ML models, including prediction based on limited-dataset training, or if such methods can successfully forecast beyond a single seismic cycle for aperiodic failure. We describe significant improvements to existing methods of labquake prediction using simple AE statistics (variance).
We demonstrate: 1) that DL models based on Long-Short Term Memory (LSTM) and Convolution Neural Networks (CNN) predict labquakes under a variety of conditions, including pre-seismic creep, aperiodic events and alternating slow and fast events and 2) that fault zone stress can be predicted with fidelity (accuracy in terms of R^2 > 0.92), confirming that acoustic energy is a fingerprint of the fault zone stress. We predict also time to start of failure (TTsF) and time to the end of Failure (TTeF) for labquakes. Interestingly, TTeF is successfully predicted in all seismic cycles, while the TTsF prediction varies with the amount of preseismic fault creep. 
We also report AR methods to forecast the evolution of fault stress using three sequence modeling frameworks: LSTM, Temporal Convolution Network (TCN) and Transformer Network (TF). AR forecasting is distinct from existing predictive models, which predict only a target variable at a specific time. The results for forecasting shear stress beyond a single seismic cycle are limited but encouraging.
Our ML/DL prediction models outperform the state-of-the-art and our autoregressive model represents a novel framework that could enhance current methods of earthquake forecasting.



## Code will be released soon!

