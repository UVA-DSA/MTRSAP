import torch.optim as optim
import torch.nn as nn


from .recognition.transtcn import TransformerModel
from .recognition.compasstcn import TCN
from .prediction.transformer import TransformerEncoderDecoderModel
from .utils import ReduceLROnPlateau

def initiate_model(input_dim, output_dim, transformer_params, learning_params, tcn_model_params, model_name):

    d_model, nhead, num_layers, hidden_dim, layer_dim, encoder_params, decoder_params = transformer_params.values()

    lr, epochs, weight_decay, patience = learning_params.values()

    if (model_name == 'transformer'):
        print("Creating Transformer")
        model = TransformerModel(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                 hidden_dim=hidden_dim, layer_dim=layer_dim, encoder_params=encoder_params, decoder_params=decoder_params)

    elif (model_name == 'tcn'):
        print("Creating TCN")
        model = TCN(input_dim=input_dim, output_dim=output_dim,
                    tcn_model_params=tcn_model_params)

    model = model.cuda()

    # Define the optimizer (Adam optimizer with weight decay)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay, betas=(0.9,0.98), eps=1e-9)


    # Define the learning rate scheduler (ReduceLROnPlateau scheduler)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion