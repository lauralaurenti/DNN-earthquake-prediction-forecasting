import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt
from scipy.io import loadmat
import scipy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import random
import copy
import math


from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def CreateSubwindows(df, n=100):
  """Creates running windows to interpolate the DataFrame.
  It takes 1 point every n to low the resolution.

  Args:
  ----------
  df : DataFrame
        Input DataFrame
  n : int (default=100)
        Number of points to take in lowering the resolution.

  Returns:
  ----------
  df : resulting DataFrame
  """

  # running windows to interpolate the df
  df=df.rolling(1000).apply(lambda w: scipy.stats.trim_mean(w, 0.05)) #mean without outliers (5 and 95 percentile)
  df=df[1000:(len(df))]
  df=df.reset_index()
  df=df.drop(['index'], axis=1)
  # now we take 1 point every n to low the resolution
  subwindows=[list(i) for i in zip(*[df.values.reshape(-1)[i:i+n] for i in range(0, len(df.values.reshape(-1)), n)])]
  df = pd.DataFrame(subwindows)
  df=df.T
  df=df.apply(np.float32)
  print("A plot of example:")
  plt.plot(df[40])
  plt.show()
  return df

def train_val_test_split(df, train_percentage, val_percentage, test_percentage,window_shift, batch_size,steps_in=200, steps_out=100):
  """Splits the input DataFrame in train, validation and test.

  Args:
  ----------
  df : DataFrame
        Input DataFrame
  train_percentage : float 0<=train_percentage<=1 (suggestion=0.7)
        Percentage of df length to use as training data.
  val_percentage : float 0<=val_percentage<=1 (suggestion=0.1)
        Percentage of df length to use as validation data.
  test_percentage : float 0<=test_percentage<=1 (suggestion=0.2)
        Percentage of df length to use as testing data.
        Note: train_percentage+val_percentage+test_percentage must be <1 and should be =1.
  window_shift : int (suggestion=10)
        Shift between one window and the following.
  batch_size : int
        Size of the batch used during the training of the model.
  steps_in : int (default=200)
        Number of points in input for the model, representing the known past.
  steps_out : int (default=100)
        Number of points in output from the model, representing the future to be forecasted.

  Returns:
  ----------
  tr_dl : torch.utils.data.DataLoader
        Resulting training set
  val_dl : torch.utils.data.DataLoader
        Resulting validation set
  test_dl : torch.utils.data.DataLoader
        Resulting testing set

  """
  train_size = int(len(df) * train_percentage)
  dataset_train = df[0:train_size]
  training_set = dataset_train.iloc[:, 0:dataset_train.shape[1]].values

  val_size = int(len(df) * val_percentage)
  dataset_val = df[train_size-steps_out:train_size+val_size+steps_out]
  val_set= dataset_val.iloc[:, 0:dataset_val.shape[1]].values

  test_size = int(len(df) * test_percentage)
  dataset_test = df[train_size+val_size-steps_out:train_size+val_size+test_size]
  test_set= dataset_test.iloc[:, 0:dataset_test.shape[1]].values

  print('training_set.shape: ',training_set.shape,' val_set.shape: ', val_set.shape,' test_set.shape: ', test_set.shape)

  df_tr = []
  for j in range(0,training_set.shape[1]):
    for i in range(0,training_set.shape[0] - steps_in - steps_out , window_shift):
      df_tr.append(training_set[:,j][i:i+steps_in+steps_out])

  X_train = []
  y_train = []
  label_train = []
  for elem in df_tr:
      X_train.append(np.expand_dims(elem[0:steps_in], axis=1))
      y_train.append(np.expand_dims(elem[steps_in:-1], axis=1))
      label_train.append(np.expand_dims(elem[steps_in:], axis=1))
  X_train, y_train, label_train  = np.array(X_train), np.array(y_train), np.array(label_train)
  print("X_train.shape: ", X_train.shape," y_train.shape: ", y_train.shape," label_train.shape: ", label_train.shape)

  df_val = []
  for j in range(0,val_set.shape[1]):
    for i in range(0,val_set.shape[0] - steps_in - steps_out , window_shift):
      df_val.append(val_set[:,j][i:i+steps_in+steps_out])

  X_val = []
  y_val = []
  label_val = []
  for elem in df_val:
      X_val.append(np.expand_dims(elem[0:steps_in], axis=1))
      y_val.append(np.expand_dims(elem[steps_in:-1], axis=1))
      label_val.append(np.expand_dims(elem[steps_in:], axis=1))
  X_val, y_val, label_val  = np.array(X_val), np.array(y_val), np.array(label_val)
  print("X_val.shape: ", X_val.shape," y_val.shape: ", y_val.shape," label_val.shape: ", label_val.shape)

  df_test = []
  for j in range(0,test_set.shape[1]):
    for i in range(0,test_set.shape[0] - steps_in - steps_out , window_shift):
      df_test.append(test_set[:,j][i:i+steps_in+steps_out])

  X_test = []
  y_test = []
  label_test = []
  for elem in df_test:
      X_test.append(np.expand_dims(elem[0:steps_in], axis=1))
      y_test.append(np.expand_dims(elem[steps_in:-1], axis=1))
      label_test.append(np.expand_dims(elem[steps_in:], axis=1))
  X_test, y_test, label_test  = np.array(X_test), np.array(y_test), np.array(label_test)
  print("X_test.shape: ", X_test.shape," y_test.shape: ", y_test.shape," label_test.shape: ", label_test.shape)
  src_train, trg_train, lab_train = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(label_train)
  src_val, trg_val, lab_val = torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(label_val)
  src_test, trg_test, lab_test = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(label_test)
  train_dataset = TensorDataset(src_train, trg_train, lab_train )
  val_dataset = TensorDataset(src_val, trg_val, lab_val)
  test_dataset = TensorDataset(src_test, trg_test, lab_test)
  tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
  return tr_dl, val_dl, test_dl


def update_lr(optimizer, lr):
    """Updates the learning rate of the model optimizer.

        Args:
        ----------
        optimizer : torch.optim
            Model optimizer.
        lr : float
            New learning rate to be used.

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class RMSELoss(nn.Module):
    """Loss function for the training of the model.

    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class TCN(nn.Module):
  """ Temporal Convolution Network

  """
  def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(TCN, self).__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,output_size)
        self.relu = nn.ReLU(inplace=True)

  def forward(self, batch_init, batch_next, batch_size, steps_in, steps_out, tf_prob=1):

        pred_out = []

        # observed data
        x = self.fc1(batch_init.view(batch_size,1,steps_in))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        pred_out.append(output)

        # data to predict
        for idx in range(batch_next.size(1)):
            if random.random() < tf_prob:
                if idx<200:
                  inp=torch.cat((batch_init[:,idx+1:,:],batch_next[:,:idx+1,:]), dim=1)
                else:
                  inp=batch_next[:,(idx-199):(idx+1),:]
            else:
                if idx<200:
                  inp=torch.cat((batch_init[:,idx+1:,:],torch.stack((pred_out), dim=1).reshape((batch_size,idx+1,1))), dim=1)
                else:
                  inp=torch.stack((pred_out), dim=1).reshape((batch_size,idx+1,1))[:,(idx-199):(idx+1),:]

            x = self.fc1(inp.view(batch_size,1,steps_in))
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            output = self.fc3(x)
            pred_out.append(output)

        return torch.cat(pred_out,1)


class LSTM(nn.Module):
    """ Long-Short Term Memory Network

    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, batch_init, batch_next, batch_size, steps_in, steps_out, tf_prob=1):

        h_t = torch.zeros(self.num_layers, batch_init.size(0), self.hidden_size).to(dev)
        c_t = torch.zeros(self.num_layers, batch_init.size(0), self.hidden_size).to(dev)
        hidden = (h_t, c_t)

        pred_out = []

        # observed data
        for idx in range(batch_init.size(1)):
            out_lstm, hidden = self.lstm(batch_init[:,idx,:].unsqueeze(1), hidden)
            output = self.linear(out_lstm)

        pred_out.append(output)

        # data to predict
        for idx in range(batch_next.size(1)):

            if random.random() < tf_prob:
                inp = batch_next[:,idx,:].unsqueeze(1)
            else:
                inp = output

            out_lstm, hidden = self.lstm(inp, hidden)
            output = self.linear(out_lstm)
            pred_out.append(output)


        return torch.cat(pred_out,1)

# The following is for the Transformer Network
class IndividualTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,mean=[0,0],std=[0,0]):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.mean=np.array(mean)
        self.std=np.array(std)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model), c(position)),
            Generator(d_model, dec_out_size))

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *input):
        return self.model.generator(self.model(*input))

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)
