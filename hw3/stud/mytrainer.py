# let's start with all the imports
# NOTE: part of this code is taken from notebook #5
import torch
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from mymodel import ProBERT
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict


class Trainer:
    """Utility class to train and evaluate a model."""

    def __init__(self, model: nn.Module, loss_function, optimizer, device, tokenizer):
        """Constructor of our trainer
        Args:
            model (nn.Module): model to train
            loss_function (nn.Loss): loss function to use
            optimizer (nn.Optim): optimizer to use
            label_vocab (Vocab): label vocabulary used to decode the output
            device (torch.device): device where to perform training and validation
            tokenizer (tre.Tokenizer): tokenizer to tokenize sentences and have inputs for the transformer model
        """
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = {0: "A", 1: "B", 2: "NEITHER"}
        self.tokenizer = tokenizer

        self.predictions = []  # will contain validation set predictions, useful to plot confusion matrix
        self.truths = []  # will contain validation set truths, useful to plot confusion matrix

    def train(
        self, train_data: DataLoader, valid_data: DataLoader, epochs: int, patience: int, path: str
    ) -> Dict[str, list]:
        """Train and validate the model using early stopping with patience for the given number of epochs
        Args:
            train_data (DataLoader): a DataLoader instance containing the training dataset
            valid_data (DataLoader): a DataLoader instance used to evaluate learning progress
            epochs: the number of times to iterate over train_data
            patience (int): patience for early stopping
            path (str): path where to save weights of best epoch
        Returns:
            Dict[str, list]: dictionary containing mappings { metric:value }
        """

        train_history = []
        valid_loss_history = []
        valid_acc = []
        patience_counter = 0
        best_acc = 0.0

        if isinstance(self.model, ProBERT):
            which_text = "tagged_text"
            which_mentions = "tagged_mentions"
        else:
            which_text = "text"
            which_mentions = "mentions"

        print("Training on", self.device, "device")
        print("Start training ...")
        for epoch in range(epochs):
            print(" Epoch {:03d}".format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()  # put model in train mode

            for batch in tqdm(train_data, leave=False):
                batch[which_text] = self.tokenizer(
                    batch[which_text], padding=True, return_tensors=True, is_split_into_words=True
                ).to(self.device)

                batch[which_mentions], y_data = batch[which_mentions].to(self.device), batch["labels"].to(self.device)
                if batch["pos"] is not None:
                    batch["pos"] = batch["pos"].to(self.device)  # if using pos, move to device

                self.optimizer.zero_grad()
                logits = self.model(batch)  # forward step output has shape: batchsize, 3 classes

                sample_loss = self.loss_function(logits, y_data)  # compute Cross Entropy Loss

                sample_loss.backward()  # backpropagation
                self.optimizer.step()  # optimize parameters

                epoch_loss += sample_loss.item() * len(batch["id"])
                # avg batch loss * precise number of batch elements

            avg_epoch_loss = epoch_loss / len(train_data.dataset)
            # total loss / number of samples = average sample loss for this epoch
            train_history.append(avg_epoch_loss)
            print("  [E:{:2d}] train loss = {:0.4f}".format(epoch + 1, avg_epoch_loss))

            valid_metrics = self.evaluate(valid_data)  # validation step
            valid_loss_history.append(valid_metrics["loss"])
            valid_acc.append(valid_metrics["acc"])
            print(
                "\t[E:{:2d}] Entity resolution valid acc = {:0.4f}%".format(
                    epoch + 1,
                    valid_metrics["acc"] * 100,
                )
            )

            # save model if the validation metric is the best ever
            if valid_metrics["acc"] > best_acc:
                best_acc = valid_metrics["acc"]
                torch.save(self.model.state_dict(), path)

            stop = epoch > 0 and valid_acc[-1] < valid_acc[-2]  # check if early stopping
            if stop:
                patience_counter += 1
                if patience_counter > patience:  # in case we exhausted the patience, we stop
                    print("\tEarly stop\n")
                    break
                else:
                    print("\t-- Patience")
            print()

        print("Done!")
        return {
            "train_history": train_history,
            "valid_loss_history": valid_loss_history,
            "valid_acc": valid_acc,
        }

    def evaluate(self, valid_data: DataLoader) -> Dict[str, float]:
        """perform validation of the model
        Args:
            valid_data: the DataLoader to use to evaluate the model.
        Returns:
            Dict[str, float]: dictionary containing mappings { metric:value }
        """
        valid_loss = 0.0
        self.predictions = []  # reset predictions and truths lists
        self.truths = []

        if isinstance(self.model, ProBERT):
            which_text = "tagged_text"
            which_mentions = "tagged_mentions"
        else:
            which_text = "text"
            which_mentions = "mentions"

        self.model.eval()  # inference mode
        with torch.no_grad():
            for batch in tqdm(valid_data, leave=False):
                batch[which_text] = self.tokenizer(
                    batch[which_text], padding=True, return_tensors=True, is_split_into_words=True
                ).to(self.device)

                batch[which_mentions], y_data = batch[which_mentions].to(self.device), batch["labels"].to(self.device)
                if batch["pos"] is not None:
                    batch["pos"] = batch["pos"].to(self.device)  # if using pos, move to device

                batch_size: int = len(batch["id"])
                logits = self.model(batch)

                sample_loss = self.loss_function(logits, y_data)  # permute to match CrossEntropyLoss input dim
                predictions = torch.argmax(logits, -1)

                valid_loss += sample_loss.item() * batch_size  # avg batch loss * precise number of batch elements

                # update predictions and gt lists
                self.predictions.extend([self.label_vocab[elem.item()] for elem in predictions])
                self.truths.extend([self.label_vocab[elem.item()] for elem in y_data])

        correct = sum([a == b for a, b in zip(self.predictions, self.truths)])
        return {
            "loss": valid_loss / len(valid_data.dataset),
            # total loss / number of samples = average sample loss for validation step
            "acc": float(correct) / len(self.predictions),
        }

    def generate_cm(self, path: str):
        """save to image the confusion matrix of the validation set of this trainer and plot a classification report
        Args:
            path (str): path where to save the image
        """

        labels = ["A", "B", "NEITHER"]
        print(classification_report(self.truths, self.predictions))
        cm = np.around(
            confusion_matrix(
                self.truths,
                self.predictions,
                labels=labels,
                normalize="true",
            ),  # normalize over ground truths
            decimals=2,
        )

        df_cm = pd.DataFrame(
            cm, index=labels, columns=labels
        )  # create a dataframe just for easy plotting with seaborn
        plt.figure(figsize=(6, 6))
        cm_plot = sn.heatmap(df_cm, annot=True, fmt="g")
        cm_plot.set_xlabel("Predicted labels")  # add some interpretability
        cm_plot.set_ylabel("True labels")
        cm_plot.set_title("Confusion Matrix")
        cm_plot.figure.savefig(path, bbox_inches="tight", pad_inches=0.5)
        return

    @staticmethod
    def plot_logs(logs: Dict[str, list], path: str):
        """Utility function to generate plot for metrics of loss in train vs validation. Code taken from notebook #5
        Args:
            logs (Dict[str, list]): dictionary containing the metrics
            path (str): path of the image to be saved
        """
        plt.figure(figsize=(8, 6))  # create the figure

        # plot losses over epochs
        plt.plot(list(range(len(logs["train_history"]))), logs["train_history"], label="Train loss")
        plt.plot(list(range(len(logs["valid_loss_history"]))), logs["valid_loss_history"], label="Validation loss")

        # add some labels
        plt.title("Train vs Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        plt.savefig(path, bbox_inches="tight")
        return
