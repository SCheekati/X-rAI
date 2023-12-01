from model import ClassificationModel
from data import CTScanDataset, custom_collate, list_blobs_with_prefix
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


encoder = "beit"
decoder = "neuraltree"

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


def train(model, iterator, optimizer, epoch):
    
    # optimizer, scheduler = model.configure_optimizers()
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        epoch_loss += loss.item()
        print('Epoch ', epoch, ':', round((i / len(iterator)) * 100, 2), '%, Loss: ', loss.item())

    model_loss = epoch_loss / len(iterator)
    print('Epoch ', epoch, ':', ' Final training loss: ', model_loss)

    return model_loss


def evaluate_val(model, iterator, epoch):
    model.eval()
    epoch_loss = 0
    out = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # output = model(src, trg[:, :-1])
            # output_reshape = output.contiguous().view(-1, output.shape[-1])
            # trg = trg[:, 1:].contiguous().view(-1)

            # loss = model.criterion(output_reshape, trg)
            # loss.backward()
            local_out = model.validation_step(batch, i)
            out.append(local_out)
            #print(local_out)
            loss = local_out["loss"]

            
            epoch_loss += loss
            print('Epoch ', epoch, ':', round((i / len(iterator)) * 100, 2), '%, Loss: ', loss)

        stats = model.validation_epoch_end(out)
        acc = stats['acc']
        model_loss = stats['loss']

        print('Epoch ', epoch, ':', ' Final validation loss: ', model_loss, ', Validation accuracy: ', acc)

    return stats


def evaluate_test(model, iterator):
    model.eval()
    epoch_loss = 0
    out = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # output = model(src, trg[:, :-1])
            # output_reshape = output.contiguous().view(-1, output.shape[-1])
            # trg = trg[:, 1:].contiguous().view(-1)

            # loss = model.criterion(output_reshape, trg)
            # loss.backward()
            local_out = model.test_step(batch, i)
            out.append(local_out)
            loss = local_out["loss"]

            
            epoch_loss += loss
            # print('Epoch ', epoch, ':', round((i / len(iterator)) * 100, 2), '%, Loss: ', loss.item())

        stats = model.test_epoch_end(out)
        acc = stats['acc']
        model_loss = stats['loss']

        print('Test loss: ', model_loss, ', Accuracy: ', acc)

    return stats

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
            train_loss = train(model, train_iterator, optimizer, i)
        if val_iterator is not None:
            val_stats = evaluate_val(model, val_iterator, i)
        print("we're done with one iteration!")

def eval(model, test_iterator):
    test_stats = evaluate_test(model, test_iterator)



bucket_name = "x_rai-dataset"
prefix = "resized/pre_processed/multimodalpulmonaryembolismdataset/" 
print(list_blobs_with_prefix(bucket_name, prefix))
pass
ct_set = CTScanDataset(
    bucket_name="x_rai-dataset",
    npy_files=list_blobs_with_prefix(bucket_name, prefix),
    labels_dir="data/Labels.csv",
    transform=None,
    stride = 5
)

trainloader = DataLoader(ct_set, batch_size=2,
                        shuffle=False, num_workers=3, collate_fn=custom_collate)
batch = next(iter(trainloader))


fit(model, 10, trainloader, None)
eval(model, None)