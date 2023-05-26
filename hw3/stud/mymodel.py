# let's start with all the imports
# NOTE: part of this code is taken from notebook #6 - POS tagging
import torch
import transformers_embedder as tre
from typing import Dict
from torch import nn


class ModHParams:
    """A wrapper class that contains the hyperparamers of the model"""

    def __init__(
        self,
        num_classes: int = 3,
        mentions_embedding_dim: int = 300,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.35,
        pos_embedding_dim: int = 0,
        fc_layers: int = 2,
        use_transformer: str = "roberta-base",
        fine_tune_trans: bool = False,
        tokenizer_len: int = 50271,
    ):
        """CR model's hyperparameters initialization

        Args:
            num_classes (int, optional): number of output classes. Defaults to 3.
            mentions_embedding_dim (int, optional): dimension of mention embeddings. Defaults to 300.
            hidden_dim (int, optional): dimension of hidden size of the LSTM. Defaults to 256.
            lstm_layers (int, optional): number of LSTM layers. Defaults to 3.
            dropout (float, optional): amount of dropout. Defaults to 0.35.
            pos_embedding_dim (int, optional): dimension of pos embeddings. Defaults to 0.
            fc_layers (int, optional): number of linear layers after the LSTM. Defaults to 1.
            use_transformer (str, optional): type of transformer to use. Defaults to "roberta-base".
            fine_tune_trans (bool, optional): whether to fine tune the transformer or not. Defaults to False.
            tokenizer_len (int, optional): length of the vocabulary of the tokenizer. Defaults to 50271.
        """
        self.hidden_dim = hidden_dim
        self.mentions_embedding_dim = mentions_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.num_classes = num_classes
        self.bidirectional = True
        self.lstm_layers = lstm_layers
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.use_transformer = use_transformer
        self.fine_tune_trans = fine_tune_trans
        self.tokenizer_len = tokenizer_len


class CRModelLSTM(nn.Module):
    """My model to perform CR"""

    def __init__(self, hparams, device, model_type="last_output"):
        """constructor of the model
        Args:
            hparams: an object embedding all the hyperparameters
        """
        super(CRModelLSTM, self).__init__()

        self.use_transformer = hparams.use_transformer
        self.lstm_hidden_dim = hparams.hidden_dim
        self.device = device
        self.model_type = model_type

        # word embeddings layer
        self.word_embedding = tre.TransformersEmbedder(
            self.use_transformer,
            layer_pooling_strategy="mean",
            fine_tune=hparams.fine_tune_trans,
        )

        # mentions embeddings layer
        self.mentions_embedding = None
        if hparams.mentions_embedding_dim != 0:
            self.mentions_embedding = nn.Embedding(
                5, hparams.mentions_embedding_dim, padding_idx=0
            )  # 5 = 4 types of mentions + 1 for padding

        # pos embeddings layer
        self.pos_embedding = None
        if hparams.pos_embedding_dim != 0:
            self.pos_embedding = nn.Embedding(
                18, hparams.pos_embedding_dim, padding_idx=0
            )  # 18 = 17 upos tags + 1 for padding

        # LSTM layer
        self.lstm = nn.LSTM(
            self.word_embedding.hidden_size
            + hparams.pos_embedding_dim
            + hparams.mentions_embedding_dim,  # in forward method I will concatenate the embeddings
            hparams.hidden_dim,
            bidirectional=hparams.bidirectional,
            num_layers=hparams.lstm_layers,
            batch_first=True,
            dropout=hparams.dropout if hparams.lstm_layers > 1 else 0,
        )

        # dropout layer to allow some more regularization
        self.dropout = nn.Dropout(hparams.dropout)

        # compute lstm output dim to create the linear layers
        lstm_output_dim = hparams.hidden_dim if not hparams.bidirectional else hparams.hidden_dim * 2
        # feed forward layers before classification
        modules = []
        for _ in range(1, hparams.fc_layers):
            # iteratively add fc layers before the classification one
            modules.append(nn.Linear(lstm_output_dim, lstm_output_dim // 2))  # halving dimension at each layer
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(hparams.dropout))
            lstm_output_dim = lstm_output_dim // 2

        self.fc = nn.Sequential(*modules)

        # last fc layer for classification
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, sample: Dict) -> torch.Tensor:
        """forward method of the model
        Args:
            sample (Dict): dictionary containing informations on the samples, with padding

        Returns:
            torch.Tensor: logits for each class, for each sentence in the batch
        """

        # extract informations from sample dict
        words = sample["text"]
        mentions = sample["mentions"]
        pos = sample["pos"]

        word_embeddings = self.word_embedding(**words).word_embeddings
        mentions_embeddings = self.mentions_embedding(mentions)
        embeddings = torch.cat(
            [word_embeddings, mentions_embeddings], dim=2
        )  # concatenate word and mentions embeddings
        if pos is not None:
            pos_embeddings = self.pos_embedding(pos)
            embeddings = torch.cat([embeddings, pos_embeddings], dim=2)  # concatenate pos embeddings

        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)

        if self.model_type == "last_output":
            # take final output as a sentence embedding
            forward = o[:, -1, : self.lstm_hidden_dim]
            backward = o[:, 0, self.lstm_hidden_dim :]
            o = torch.cat((forward, backward), dim=1)
        else:
            # take avg embedding for pronoun tokens
            new_o = torch.empty((1, o.shape[-1]), requires_grad=True, device=self.device)
            for idx in range(o.shape[0]):
                mask = mentions[idx] == 2  # 2 is the value for the indexes of pronoun
                elem = torch.mean(o[idx][mask], 0, keepdim=True)
                new_o = torch.cat((new_o, elem), 0)
            o = new_o[1:]

        o = self.dropout(o)
        o = self.fc(o)
        output = self.classifier(o)
        return output


class ProBERT(nn.Module):
    """My model to perform CR inspired by https://aclanthology.org/W19-3820.pdf"""

    def __init__(self, hparams, device):
        """constructor of the model
        Args:
            hparams: an object embedding all the hyperparameters
            device: device on which the model is trained
        """
        super(ProBERT, self).__init__()

        self.use_transformer = hparams.use_transformer
        self.device = device

        # word embeddings layer
        self.word_embedding = tre.TransformersEmbedder(
            self.use_transformer,
            layer_pooling_strategy="mean",
            fine_tune=hparams.fine_tune_trans,
        )
        # adapt to added mention tokens
        self.word_embedding.resize_token_embeddings(hparams.tokenizer_len)

        # dropout layer to allow some more regularization
        self.dropout = nn.Dropout(hparams.dropout)

        # last fc layer for classification
        self.classifier = nn.Linear(self.word_embedding.hidden_size, hparams.num_classes)

    def forward(self, sample: Dict) -> torch.Tensor:
        """forward method of the model
        Args:
            sample (Dict): dictionary containing informations on the samples, with padding

        Returns:
            torch.Tensor: logits for each class, for each sentence in the batch
        """

        # extract informations from sample dict
        words = sample["tagged_text"]
        mentions = sample["tagged_mentions"]

        embeddings = self.word_embedding(**words).word_embeddings
        embeddings = self.dropout(embeddings)

        # take avg embedding for pronoun tokens
        pronouns = torch.empty((1, embeddings.shape[-1]), requires_grad=True, device=self.device)
        for idx in range(embeddings.shape[0]):
            mask = mentions[idx] == 2  # 2 is the value for the indexes of pronoun
            elem = torch.mean(embeddings[idx][mask], 0, keepdim=True)
            pronouns = torch.cat((pronouns, elem), 0)
        o = pronouns[1:]

        output = self.classifier(o)
        return output


class CorefSeq(nn.Module):
    """My model to perform CR inspired by https://arxiv.org/pdf/1906.03695v1.pdf"""

    def __init__(self, hparams, device):
        """constructor of the model
        Args:
            hparams: an object embedding all the hyperparameters
            device: device on which the model is trained
        """
        super(CorefSeq, self).__init__()

        self.use_transformer = hparams.use_transformer
        self.device = device

        # word embeddings layer
        self.word_embedding = tre.TransformersEmbedder(
            self.use_transformer,
            layer_pooling_strategy="mean",
            fine_tune=hparams.fine_tune_trans,
        )

        # dropout layer to allow some more regularization
        self.dropout = nn.Dropout(hparams.dropout)

        # last fc layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.word_embedding.hidden_size * 3, 512), nn.ReLU(), nn.Linear(512, hparams.num_classes)
        )

    def forward(self, sample: Dict) -> torch.Tensor:
        """forward method of the model
        Args:
            sample (Dict): dictionary containing informations on the samples, with padding

        Returns:
            torch.Tensor: logits for each class, for each sentence in the batch
        """

        # extract informations from sample dict
        words = sample["text"]
        mentions = sample["mentions"]

        embeddings = self.word_embedding(**words).word_embeddings
        embeddings = self.dropout(embeddings)

        # take avg embedding for pronoun, entity A and entity B tokens
        pronouns = torch.empty((1, embeddings.shape[-1]), requires_grad=True, device=self.device)
        entities_a = torch.empty((1, embeddings.shape[-1]), requires_grad=True, device=self.device)
        entities_b = torch.empty((1, embeddings.shape[-1]), requires_grad=True, device=self.device)
        for idx in range(embeddings.shape[0]):  # for each element in the batch
            mask = mentions[idx] == 2  # 2 is the value for the indexes of pronoun
            elem = torch.mean(embeddings[idx][mask], 0, keepdim=True)
            pronouns = torch.cat((pronouns, elem), 0)

            mask = mentions[idx] == 3  # 3 is the value for the indexes of entity A
            elem = torch.mean(embeddings[idx][mask], 0, keepdim=True)
            entities_a = torch.cat((entities_a, elem), 0)

            mask = mentions[idx] == 4  # 4 is the value for the indexes of entity B
            elem = torch.mean(embeddings[idx][mask], 0, keepdim=True)
            entities_b = torch.cat((entities_b, elem), 0)
        o = torch.cat((pronouns, entities_a, entities_b), 1)[1:]

        output = self.classifier(o)
        return output
