import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pprint import pprint


class SolarModel(pl.LightningModule):
    
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.epoch_outputs = []

        self.val_results = {}
        self.train_results = {}

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        # print(pred_mask.shape)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        # print(tp.shape)
        metrics = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        return metrics

    def shared_epoch_end(self, stage):
      
        if len(self.epoch_outputs) == 0:
          print("Yeees")
          return
        # outputs = self.epoch_outputs
        outputs = self.epoch_outputs.copy()  # Make a copy of epoch_outputs
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        total_loss = sum([output['loss'] for output in outputs])
        average_loss = total_loss / len(outputs)
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
        specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")
        balanced_accuracy = smp.metrics.balanced_accuracy(tp, fp, fn, tn, reduction="micro-imagewise")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_average_loss":average_loss,
            f"{stage}_f1_score":f1_score,
            f"{stage}_accuracy":accuracy,
            f"{stage}_precision":precision,
            f"{stage}_recall":recall,
            f"{stage}_sensitivity":sensitivity,
            f"{stage}_specificity":specificity,
            f"{stage}_balanced_accuracy":balanced_accuracy,
        }
        metrics = {k:round(float(v.cpu().data.numpy()), 2) for k,v in metrics.items()}
        # pprint(metrics)
        # self.log('train_loss', average_loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

        self.epoch_outputs.clear()
        return metrics

    def training_step(self, batch, batch_idx):
        outs = self.shared_step(batch, "train")            
        self.epoch_outputs.append(outs)
        results = self.shared_epoch_end("train")
        self.train_results[len(self.train_results)] = results
        # print(len(self.epoch_outputs))
        return outs

    def validation_step(self, batch, batch_idx):
        outs = self.shared_step(batch, "valid")
        self.epoch_outputs.append(outs)
        return outs

    def on_validation_epoch_end(self):
        reuslts = self.shared_epoch_end('valid')
        self.val_results[self.current_epoch] = reuslts
        return reuslts

    def test_step(self, batch, batch_idx):
        outs =  self.shared_step(batch, "test")
        self.epoch_outputs.append(outs)
        return outs  

    def on_test_epoch_end(self):
        return self.shared_epoch_end('test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)