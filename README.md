These programs were used for my google science fair project and paper (https://onlinelibrary.wiley.com/doi/abs/10.1111/itor.12887). Note: all work over last 2 years was done in a private repo under my professor's name -- I have only recently made this new public repo with the important code for the paper. 

Each python program is a different variation of the ConvLSTM and can run standalone (without the other python programs in the folder). 

AnnualGateConvLSTMEncoderDecoder.py - This is the version that achieved the highest accuracy on the data we tested and can do multi-step forecasts if necessary. This modification has the "annual gate" modification that allows it to remember the vegetation of the previous year.
  - for more information on this model, check the "annualgate1.png" and "annualgate2.png" images in the folder. 

AnnualGateConvLSTM.py - This version adds an "annual gate" to the normal ConvLSTM to improve accuracies. Does not have an encoder-decoder.

convLSTM_numpy.py - standard convLSTM design derived from https://arxiv.org/pdf/1506.04214.pdf. 

convLSTM_keras.py - standard convLSTM design, but in Keras Language.

vanillaConvRNN.py - Similar to convLSTM but with a vanilla RNN structure instead of an LSTM. 

vanillaConvRNNAttention.py - A vanillaConvRNN model fitted with an attention model to remember NDVI from the previous year. 

