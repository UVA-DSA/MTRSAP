import torch.optim as optim
import torch.nn as nn


from .recognition.transtcn import TransformerModel
from .recognition.compasstcn import TCN
from .prediction.transformer import TransformerEncoderDecoderModel
from .utils import ReduceLROnPlateau

def initiate_model(input_dim, output_dim, transformer_params, learning_params, tcn_model_params, model_name):

    d_model = transformer_params["d_model"]
    nhead = transformer_params["nhead"]
    num_layers = transformer_params["num_layers"]
    batch_first = transformer_params.get("batch_first", False)
    hidden_dim = transformer_params["hidden_dim"]
    layer_dim = transformer_params["layer_dim"]
    encoder_params = transformer_params["encoder_params"]
    decoder_params = transformer_params["decoder_params"]

    lr = learning_params["lr"]
    weight_decay = learning_params["weight_decay"]
    patience = learning_params["patience"]

    if (model_name == 'transformer'):
        print("Creating Transformer")
        model = TransformerModel(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                 hidden_dim=hidden_dim, layer_dim=layer_dim, encoder_params=encoder_params,
                                 decoder_params=decoder_params, batch_first=batch_first)

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
