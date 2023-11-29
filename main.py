from model import ClassificationModel
from data import CTScanDataset
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

encoder = "beit"
decoder = "neuraltree"
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(DEVICE)

model = ClassificationModel(
    force_2d = False,  # if set to True, the model will be trained on 2D images by only using the center slice as the input
    use_pretrained = True,  # whether to use pretrained backbone (only applied to BEiT)
    bootstrap_method = "centering",  # whether to inflate or center weights from 2D to 3D
    in_channels = 1,
    out_channels = 1,  # number of classes
    patch_size = 16,  # no depthwise
    img_size = (512, 512, 5),
    hidden_size = 768,
    mlp_dim = 3072,
    num_heads = 12,
    num_layers = 12,
    encoder = encoder,
    decoder = decoder,
    loss_type = "ce",
    save_preds = False,
    dropout_rate = 0.0,
    learning_rate = 1e-4,
    weight_decay = 1e-5,
    warmup_steps = 500,
    max_steps = 20000,
    adam_epsilon = 1e-8,
)
model.to("cuda:0")
#device_of_model = next(model.parameters()).device
#print("MODEL DEVICE IS ", device_of_model)
device_of_model = next(model.parameters()).device
print("MODEL DEVICE IS", device_of_model)


# TODO: Make sure the optimizer is performing correctly according to our parameters!
def train(model, iterator, optimizer):
    
    # optimizer, scheduler = model.configure_optimizers()
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        print("BATCH IMAGE SIZE")
        print(batch["image"][0].size())
        optimizer.zero_grad()

        loss = model.training_step(batch, i)
        print(loss)
        # print(loss.data())
        print("this is literally right before the backwards step")
        loss.backward()

        print("we did the backwards step??")

        optimizer.step()

        print("we got here!")

        epoch_loss += loss.item()
        print("we got here too!")
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # output = model(src, trg[:, :-1])
            # output_reshape = output.contiguous().view(-1, output.shape[-1])
            # trg = trg[:, 1:].contiguous().view(-1)

            # loss = model.criterion(output_reshape, trg)
            # loss.backward()
            out = model.validation_step(batch, i)
            loss = out["loss"]
            
            epoch_loss += loss.item()
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

import pandas as pd
import numpy as np

def fit(model, num_epochs, train_iterator, val_iterator):
    mod = nn.ModuleList([model.encoder, model.decoder])
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in mod.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": model.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in mod.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=model.learning_rate,
            eps=model.adam_epsilon,
        )
    
    for i in range(num_epochs):
        if train_iterator is not None:
            train_loss = train(model, train_iterator, optimizer)
        if val_iterator is not None:
            val_loss = evaluate(model, val_iterator)
        print("we're done with one iteration!")

train_features = np.load("./data/0.npy")
train_features = train_features[0:5, :, :]

test_features = np.load("./data/1.npy")
test_features = test_features[0:5, :, :]

# train_features = torch.transpose(torch.from_numpy(train_features), 0, 2)
# print(train_features.size())


# labels = pd.read_csv('./data/Labels.csv', delimiter=",")
# labels = torch.from_numpy(labels)

labels = np.array([[0, 1], [1, 0], [0, 1]])
labels = torch.from_numpy(labels)

print(labels.size())

train_features = np.reshape(train_features, (1, 1, train_features.shape[0], train_features.shape[1], train_features.shape[2]))
train_features = torch.transpose(torch.from_numpy(train_features), 2, 4)
print(train_features.size())
# out = model(train_features)
# print(out)
# print(out.size())

test = torch.from_numpy(np.array([1], dtype=float))

ct_set = CTScanDataset(
    npy_file="./data/0.npy",
    labels_dir=test,
    transform=None
)

print(len(ct_set))
trainloader = DataLoader(ct_set, batch_size=2,
                        shuffle=True, num_workers=0, pin_memory=True)
# for data, target in trainloader:
#     data, target = data.to("cuda:0"), target.to("cuda:0")

train_dict = next(iter(trainloader))
feat = train_dict["image"]
lab = train_dict["label"]
print("Checking!")
print(len(feat))
print(feat[0].size())
print(lab)

fit(model, 10, trainloader, None)