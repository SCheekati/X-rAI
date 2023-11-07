from encoders.beit3d import BEiT3D
from encoders.vit3d import VisionTransformer3D
from decoders.mlp import MlpHead
from decoders.convtrans import ConvTransHead
from monai.networks.nets.vit import ViT
from utils import to_list, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from monai.losses import DiceCELoss, DiceFocalLoss

import datetime
import pickle

class ClassificationModel(nn.Module):
    def __init__(
        self,
        force_2d: bool = False,  # if set to True, the model will be trained on 2D images by only using the center slice as the input
        use_pretrained: bool = True,  # whether to use pretrained backbone (only applied to BEiT)
        bootstrap_method: str = "centering",  # whether to inflate or center weights from 2D to 3D
        in_channels: int = 1,
        out_channels: int = 14,  # number of classes
        patch_size: int = 16,  # no depthwise
        img_size: tuple = (512, 512, 5),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        encoder: str = "beit",
        decoder: str = "mlp",
        loss_type: str = "ce",
        save_preds: bool = False,
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        max_steps: int = 20000,
        adam_epsilon: float = 1e-8,
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
        elif decoder == "convtrans":
            self.decoder = ConvTransHead(
                channels=hidden_size,
                num_classes=out_channels,
                norm_name="instance"
            )
        else:
          raise

        if loss_type == "ce":
          self.criterion = CrossEntropyLoss()

    def forward(self, inputs):  # inputs: B x Cin x H x W x D
        x = inputs.permute(0, 1, 4, 2, 3).contiguous().float()  # x: B x Cin x D x H x W
        xs = self.encoder(x)  # hiddens: list of B x T x hidden, where T = H/P x W/P
        print(xs[-1].size())
        # xs = [
        #     xs[i]
        #     .view(inputs.shape[0], self.feat_size[0], self.feat_size[1], -1)
        #     .permute(0, 3, 1, 2)
        #     .contiguous()
        #     for i in range(len(xs))
        # ]  # xs: list of B x hidden x H/P x W/P
        # print(len(xs))
        # print(xs[-1].size())
        x = self.decoder(xs)  # x: B x Cout x H x W
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        n_slices = inputs.shape[-1]
        assert n_slices == self.img_size[-1]
        if self.force_2d:
            inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        # labels = labels[:, :, :, :, n_slices // 2].contiguous()
        outputs = self(inputs)
        # outputs = torch.mean(outputs, (2, 3))
        print(outputs.size())
        print(labels.size())
        loss = self.criterion(outputs, labels)
        # dice_loss, ce_loss = torch.tensor(0), torch.tensor(0)
        result = {
            "train/loss": loss.item(),
        }
        # self.log_dict(result)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        n_slices = inputs.shape[-1]
        assert n_slices == self.img_size[-1]
        if self.force_2d:
            inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        labels = labels[:, :, :, :, n_slices // 2].contiguous()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        return {
            "loss": loss.item(),
            "labels": to_list(labels.squeeze(dim=1)),
            "preds": to_list(outputs.argmax(dim=1)),
        }

    def validation_epoch_end(self, outputs):
        loss = np.array([x["loss"] for x in outputs]).mean()

        labels = [label for x in outputs for label in x["labels"]]  # N of image shape
        preds = [pred for x in outputs for pred in x["preds"]]  # N of image shape
        inputs = [None] * len(preds)
        # acc, accs, ious, dices = eval_metrics(
        #     preds, labels, self.hparams.out_channels, metrics=["mIoU", "mDice"]
        # )

        acc = (labels == preds).sum().item() / labels.size(0) # ?????

        result = {
            "val/loss": loss,
            "val/acc": acc,
        }
        self.log_dict(result, sync_dist=True)

        if self.save_preds:
            cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
            with open(f"dumps/val-{cur_time}.pkl", "wb") as fout:
                pickler = pickle.Pickler(fout)
                for input, pred, label in zip(inputs, preds, labels):
                    pickler.dump({"input": input, "pred": pred, "label": label})

        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["image"]
        n_slices = inputs.shape[-1]
        assert n_slices == self.img_size[-1]
        if self.force_2d:
            inputs = inputs[:, :, :, :, n_slices // 2 : n_slices // 2 + 1].contiguous()
        outputs = self(inputs)

        if "label" in batch:
            labels = batch["label"]
            labels = labels[:, :, :, :, n_slices // 2].contiguous()
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
        preds = [pred for x in outputs for pred in x["preds"]]  # N of image shape
        inputs = [None] * len(preds)
        if "labels" in outputs[0]:
            loss = np.array([x["loss"] for x in outputs]).mean()
            labels = [
                label for x in outputs for label in x["labels"]
            ]  # N of image shape
            # acc, accs, ious, dices = eval_metrics(
            #     preds, labels, self.hparams.out_channels, metrics=["mIoU", "mDice"]
            # )
            acc = (labels == preds).sum().item() / labels.size(0) # ?????

            result = {
                "test/loss": loss,
                "test/acc": acc,
            }
            self.log_dict(result, sync_dist=True)

            if self.save_preds:
                cur_time = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
                with open(f"dumps/test-{cur_time}.pkl", "wb") as fout:
                    pickler = pickle.Pickler(fout)
                    for input, pred, label in zip(inputs, preds, labels):
                        pickler.dump({"input": input, "pred": pred, "label": label})

            return loss
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