# DNN-earthquake-prediction-forecasting

## Deep Learning For Predicting Laboratory Earthquakes and Autoregressively Forecasting Fault Zone Stress State 

#### Laurenti, Tinti, Galasso, Franco, Marone 
Faculty of Information Engineering, Informatics, and Statistics. Sapienza University of Rome, Italy
Department of Computer Science. Sapienza University of Rome (IT)
Earth Science Department. Sapienza University of Rome (IT)
Earth Science Department. PennState University (PA)

## Abstract

Laboratory earthquakes are ideal targets for machine learning (ML) and deep learning (DL) because they can be produced in long sequences under a wide range of controlled conditions. Indeed, recent work shows that labquakes and fault zone stress can be predicted from fault zone acoustic emissions (AE).  Here we describe work aimed at generalizing these results and assessing possible application to tectonic faulting.  Key questions include whether improved ML/DL methods can outperform existing models, including prediction based on limited training, or if such methods can successfully forecast beyond a single seismic cycle for aperiodic failure. We find that significant improvements over existing methods of labquake prediction can be made using simple AE statistics (variance) and DL models such as Long-Short Term Memory (LSTM) and Convolution Neural Network (CNN).  We demonstrate that LSTMs and CNNs predict labquakes under a variety of conditions, including pre-seismic creep, aperiodic events and alternating slow and fast events. Shear stress is predicted with fidelity confirming that the acoustic energy is a fingerprint of the fault zone stress. We predict also Times To Failure (TTF) and Time To the end of Failure (TTeF).  Interestingly, TTeF is successfully predicted in all seismic cycles, while the TTF prediction varies with the amount of creep before an event. We also report on a novel autoregressive forecasting method to predict the future fault zone stress state.  Our forecasting model is distinct from existing predictive models, which only predict the current state. We compare three modern approaches: LSTM, Temporal Convolution Network (TCN) and Transformer Network (TF).  Results are encouraging, especially for the use of TCN and TF to forecast at long-term future horizons.  Our methods for labquake prediction outperform the state of the art and our forecasting models yield promising results that suggest a framework for advancement.



## Code will be released soon!

