from encoders.beit3d import BEiT3D
from encoders.vit3d import VisionTransformer3D
from decoders.mlp import MlpHead
from decoders.neuraltree import NeuralDecisionTree
from decoders.BNetMCD import BNetMCD
from decoders.convtrans import ConvTransHead
from decoders.neuraltree import NeuralDecisionTree
from decoders.BNetMCD import BNetMCD
from monai.networks.nets.vit import ViT
from utils import to_list, get_linear_schedule_with_warmup
import gc

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
import numpy as np
from monai.losses import DiceCELoss, DiceFocalLoss

import datetime
import pickle

from sklearn import metrics

class ClassificationModel(nn.Module):
    def __init__(
        self,
        force_2d: bool = False,                 # if true, model only cares about center slice of input
        use_pretrained: bool = True,            # whether or not to use pretrained weights
        bootstrap_method: str = "centering",    # whether to inflate or center weights from 2D to 3D
        in_channels: int = 1,                   # number of input channels
        out_channels: int = 14,                 # number of classes
        patch_size: int = 16,                   # no depthwise
        img_size: tuple = (224, 224, 5),        # dimensionality of windowed input
        hidden_size: int = 768,                 # patch embedding dimension
        mlp_dim: int = 3072,                    # dimension of the mlp in encoder blocks
        num_heads: int = 12,                    # number of attention heads
        num_layers: int = 12,                   # number of encoder blocks
        encoder: str = "beit",                  # type of ViT to use
        decoder: str = "mlp",                   # decoder head to use
        aggregator: str = "linear",             # aggregation method to use
        loss_type: str = "ce",                  # loss objective
        save_preds: bool = False,               # whether or not to save pickled predictions during val/testing
        dropout_rate: float = 0.0,              # dropout rate
        learning_rate: float = 1e-4,            # learning rate for optimizer
        weight_decay: float = 1e-5,             # weight decay for optimizer
        warmup_steps: int = 500,                # LR warmup
        max_steps: int = 20000,                 # maximum number of epochs
        adam_epsilon: float = 1e-8,             # constant in adam optimizer for numerical stability
    ):
        super().__init__()
        # self.modified_loss = (
        #     True  # TODO: set True to debug (need to modify MONAI codes)
        # )
        # self.save_hyperparameters()
        self.feat_size = (img_size[0] // patch_size, img_size[1] // patch_size, 1)
        self.force_2d = force_2d
        self.use_pretrained = use_pretrained
        self.bootstrap_method = bootstrap_method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.save_preds = save_preds
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.adam_epsilon = adam_epsilon
        gc.collect()
        if encoder == "vit":
            self.encoder = VisionTransformer3D(
                img_size=img_size if not force_2d else (img_size[0], img_size[1], 1),
                patch_size=(patch_size, patch_size, img_size[-1])
                if not force_2d
                else (patch_size, patch_size, 1),
                in_chans=in_channels,
                num_classes=out_channels,
                depth=num_layers,
                embed_dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_dim // hidden_size,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=dropout_rate,
                attn_drop_rate=0,
                drop_path_rate=0,
                norm_layer=nn.LayerNorm,
            )
        elif encoder == "beit":
          self.encoder = BEiT3D(
                img_size=img_size if not force_2d else (img_size[0], img_size[1], 1),
                patch_size=(patch_size, patch_size, img_size[-1])
                if not force_2d
                else (patch_size, patch_size, 1),
                in_chans=in_channels,
                embed_dim=hidden_size,
                depth=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_dim // hidden_size,
                qkv_bias=True,
                drop_rate=dropout_rate,
                init_values=1,
                use_abs_pos_emb=False,
                use_rel_pos_bias=True,
            )
          if use_pretrained:
                self.encoder.init_weights(bootstrap_method=bootstrap_method)
        else:
          raise

        if decoder == "mlp":
          self.decoder = MlpHead(
              channels=hidden_size,
              num_classes=out_channels,
              dropout_ratio=0.1
          )
          print("MLP Decoder")
        elif decoder == "convtrans":
            self.decoder = ConvTransHead(
                channels=hidden_size,
                num_classes=out_channels,
                norm_name="instance"
            )
        elif decoder == "neuraltree":
            self.decoder = NeuralDecisionTree(
                num_classes=out_channels,
                num_cut=(2, 2) # 3 dimensional? May need to add this as a param to this class
            )
        elif decoder == 'mcdropout':
            #input_size = 401 * 512 * 512  # input size? Changed to 224x224 after preprocessing?
            input_size = [8, 1024, hidden_size] # ?
            self.decoder = BNetMCD(
                input_shape=input_size,
                hidden_size=hidden_size,
                output_size=out_channels,
                p_mc_dropout=0.5  # MC Dropout rate
            )
        else:
          raise

        if aggregator == "linear":
            self.aggregator = torch.nn.Linear(
                in_features=36, # self.img_size[0] - self.img_size[2] + 1,
                out_features=1
            )

        if loss_type == "ce":
          self.criterion = BCEWithLogitsLoss()

    def forward(self, inputs):  # inputs: S x B x Cin x H x W x D
        inputs = torch.stack(inputs, dim=0)
        B, S, C, H, W, D = inputs.shape
        print('iteration')
        x = inputs.flatten(start_dim=0, end_dim=1) # B*S x Cin x D x H x W
        xs = self.encoder(x.to("cuda:0"))
        out = self.decoder(xs).to("cpu")  # B*S x Cout
        gc.collect()
        torch.cuda.empty_cache()
        outs = torch.reshape(out, (B, S, out.shape[1])) # B x S x Cout
        outs = outs.permute(0, 2, 1).contiguous().float()  # B x Cout x S

        outs = self.aggregator(outs)  # B x Cout x 1
        outs = torch.squeeze(outs, 2) 
        return outs
    
    # S, C, H, W, D = inputs[0].shape
    #     all_outputs = []
    #     for s in range(len(inputs)):
    #         print('iteration')
    #         x = inputs[s]  # current window
    #         x = x.view(S, C, D, H, W)  # B x Cin x D x H x W
    #         x = x.to(torch.device('cuda:0'), dtype=torch.float32)
    #         xs = self.encoder(x)
    #         out = self.decoder(xs).to("cpu")  # B x Cout? Move to cpu and then later put it back on gpu?
    #         out = torch.mean(out, dim=0) # Cout
    #         all_outputs.append(out) # B x Cout
    #         gc.collect()
    #         torch.cuda.empty_cache()


    #     outs = torch.stack(all_outputs, dim=0).to("cuda:0")  # B x Cout
    #     # outs = outs.permute(1, 2, 0).contiguous().float()  # B x Cout x S

    #     # outs = self.aggregator(outs)  # B x Cout x 1
    #     # outs = torch.squeeze(outs, 2) 
    #     return outs
    
    # S, C, H, W, D = inputs[0].shape
    #     all_outputs = []
    #     for s in range(len(inputs)):
    #         print('iteration')
    #         x = inputs[s]  # current window
    #         x = x.view(S, C, D, H, W)  # B x Cin x D x H x W
    #         x = x.to(torch.device('cuda:0'), dtype=torch.float32)
    #         xs = self.encoder(x)
    #         out = self.decoder(xs).to("cpu")  # B x Cout? Move to cpu and then later put it back on gpu?
    #         out = torch.mean(out, dim=0) # Cout
    #         all_outputs.append(out) # B x Cout
    #         gc.collect()
    #         torch.cuda.empty_cache()


    #     outs = torch.stack(all_outputs, dim=0).to("cuda:0")  # B x Cout
    #     # outs = outs.permute(1, 2, 0).contiguous().float()  # B x Cout x S

    #     # outs = self.aggregator(outs)  # B x Cout x 1
    #     # outs = torch.squeeze(outs, 2) 
    #     return outs
    
        # outs = None # S x B x Cout x H x W
        # inputs = torch.stack(inputs, dim=0)
        # inputs = inputs.to("cuda:0")
        # S = inputs.shape[0]
        # B = inputs.shape[1]
        # inputs = torch.flatten(inputs, start_dim=0, end_dim=1)
        # x = inputs.permute(0, 1, 4, 2, 3).contiguous().float()  # x: (S * B) x Cin x D x H x W
        # x = x.to("cuda:0")
        # xs = self.encoder(x)  # hiddens: list of (S * B) x T x hidden, where T = H/P x W/P
        # x = self.decoder(xs)  # x: (S * B) x Cout
        # x = torch.reshape(x, (S, B, x.shape[1]))

        # outs = x.permute(1, 2, 0).contiguous().float() # B x Cout x S
        # # outs = outs.reshape(outs, (outs.shape[0], outs.shape[1] * outs.shape[2]))
        # outs = self.aggregator(outs) # B x Cout x 1
        # outs = torch.squeeze(outs, 2)
        # return outs
    
    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]
        n_slices = inputs[0].shape[-1]
        assert n_slices == self.img_size[-1]
        if self.force_2d:
            for i in range(len(inputs)):
                inputs[i] = inputs[i][:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        # labels = labels[:, :, :, :, n_slices // 2].contiguous()
        outputs = self(inputs)
        # outputs = torch.mean(outputs, (2, 3))
        labels = labels.to("cuda:0")
        loss = self.criterion(outputs, labels)
        # dice_loss, ce_loss = torch.tensor(0), torch.tensor(0)
        result = {
            "train/loss": loss.item(),
        }
        print("done with the training step!")
        # self.log_dict(result)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        n_slices = inputs[0].shape[-1]
        assert n_slices == self.img_size[-1]
        # if self.force_2d:
        #     inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        # labels = labels[:, :, :, :, n_slices // 2].contiguous()
        # outputs = self(inputs)
        # loss = self.criterion(outputs, labels)
        if self.force_2d:
            for i in range(len(inputs)):
                inputs[i] = inputs[i][:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        # labels = labels[:, :, :, :, n_slices // 2].contiguous()
        outputs = self(inputs)
        # outputs = torch.mean(outputs, (2, 3))
        labels = labels.to("cuda:0")
        loss = self.criterion(outputs, labels)

        return {
            "loss": loss.item(),
            "labels": to_list(labels.squeeze(dim=1)),
            "preds": to_list(outputs.argmax(dim=1)),
        }

    def validation_epoch_end(self, outputs):
        print(outputs)
        loss = np.array([x["loss"] for x in outputs]).mean()

        labels = np.array([label for x in outputs for label in x["labels"]])  # N of image shape
        preds = np.array([pred for x in outputs for pred in x["preds"]])  # N of image shape
        inputs = [None] * len(preds)
        # acc, accs, ious, dices = eval_metrics(
        #     preds, labels, self.hparams.out_channels, metrics=["mIoU", "mDice"]
        # )

        # acc = (labels == preds).sum().item() / labels.size(0) # ?????
        acc = metrics.roc_auc_score(labels, preds)

        result = {
            "loss": loss,
            "acc": acc,
        }
        # self.log_dict(result, sync_dist=True)

        if self.save_preds:
            cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
            with open(f"dumps/val-{cur_time}.pkl", "wb") as fout:
                pickler = pickle.Pickler(fout)
                for input, pred, label in zip(inputs, preds, labels):
                    pickler.dump({"input": input, "pred": pred, "label": label})

        return result

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        n_slices = inputs[0].shape[-1]
        assert n_slices == self.img_size[-1]
        if self.force_2d:
            for i in range(len(inputs)):
                inputs[i] = inputs[i][:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        outputs = self(inputs)

        if "label" in batch:
            labels = batch["label"].to("cuda:0")
            #labels = labels[:, :, :, :, n_slices // 2].contiguous()
            loss = self.criterion(outputs, labels)

            return {
                "loss": loss.item(),
                "labels": to_list(labels.squeeze(dim=1)),
                "preds": to_list(outputs.argmax(dim=1)),
            }
        else:
            return {
                "preds": to_list(outputs.argmax(dim=1)),
            }

    def test_epoch_end(self, outputs):
        preds = np.array([pred for x in outputs for pred in x["preds"]])  # N of image shape
        inputs = [None] * len(preds)
        if "labels" in outputs[0]:
            loss = np.array([x["loss"] for x in outputs]).mean()
            labels = np.array([
                label for x in outputs for label in x["labels"]
            ])  # N of image shape
            # acc, accs, ious, dices = eval_metrics(
            #     preds, labels, self.hparams.out_channels, metrics=["mIoU", "mDice"]
            # )
            # acc = (labels == preds).sum().item() / labels.size(0) # ?????
            acc = metrics.roc_auc_score(labels, preds)

            result = {
                "loss": loss,
                "acc": acc,
            }
            # self.log_dict(result, sync_dist=True)

            if self.save_preds:
                cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
                with open(f"dumps/test-{cur_time}.pkl", "wb") as fout:
                    pickler = pickle.Pickler(fout)
                    for input, pred, label in zip(inputs, preds, labels):
                        pickler.dump({"input": input, "pred": pred, "label": label})

            return result
        else:
            assert self.save_preds
            cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
            with open(f"dumps/test-{cur_time}.pkl", "wb") as fout:
                pickler = pickle.Pickler(fout)
                for input, pred in zip(inputs, preds):
                    pickler.dump({"input": input, "pred": pred})

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = nn.ModuleList([self.encoder, self.decoder])
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     "Prepare optimizer and schedule (linear warmup and decay)"
    #     model = nn.ModuleList([self.encoder, self.decoder])
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [
    #                 p
    #                 for n, p in model.named_parameters()
    #                 if not any(nd in n for nd in no_decay)
    #             ],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [
    #                 p
    #                 for n, p in model.named_parameters()
    #                 if any(nd in n for nd in no_decay)
    #             ],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = torch.optim.AdamW(
    #         optimizer_grouped_parameters,
    #         lr=self.hparams.learning_rate,
    #         eps=self.hparams.adam_epsilon,
    #     )

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=self.hparams.max_steps,
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return [optimizer], [scheduler]