# this script was executed to train the models, let's start with all the imports
import os
import torch
import random
import numpy as np
import transformers_embedder as tre
from torch import nn
from torch.utils.data import DataLoader
from mydataset import CRDataset
from mytrainer import Trainer
from mymodel import ModHParams, CRModelLSTM, ProBERT, CorefSeq
from myutils import prepare_batch_transformers

# if true, execute the grid search on hyperparameters, else just train the manually-set model at the end of this script
GRID_SEARCH = False

# fix the seed to allow reproducibility
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# data paths, please note that this script ran on my local computer
MODEL_FOLDER = "model/"
DATA_FOLDER = "data/"
TRAIN_DATA = DATA_FOLDER + "train.tsv"
VAL_DATA = DATA_FOLDER + "dev.tsv"

# to avoid warnings from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate datasets and vocabs
train_data = CRDataset(TRAIN_DATA)
val_data = CRDataset(VAL_DATA)

# create DataLoaders
workers = min(os.cpu_count(), 4)
train_dataloader = DataLoader(
    train_data,
    batch_size=8,
    collate_fn=prepare_batch_transformers,
    num_workers=workers,
    shuffle=True,
)

valid_dataloader = DataLoader(
    val_data,
    batch_size=8,
    collate_fn=prepare_batch_transformers,
    num_workers=workers,
    shuffle=False,
)

if GRID_SEARCH:
    transf_name = "roberta-base"
    # grid search on some hyperparameters for baseline model
    for lstm_layers in [1, 2]:
        for fc_layers in [1, 2]:
            for mentions_size in [150, 300]:
                for wdecay in [0, 1e-5]:

                    tokenizer = tre.Tokenizer(transf_name, add_prefix_space=True)

                    # create model hyperparameters
                    params = ModHParams(
                        use_transformer=transf_name,
                        fine_tune_trans=False,
                        mentions_embedding_dim=mentions_size,
                        lstm_layers=lstm_layers,
                        fc_layers=fc_layers,
                    )

                    # instantiate and train the model
                    crmodel = CRModelLSTM(params, device, model_type="pronoun_output")
                    trainer = Trainer(
                        model=crmodel,
                        loss_function=nn.CrossEntropyLoss(),
                        optimizer=torch.optim.Adam(
                            crmodel.parameters(),
                            lr=0.001,
                            weight_decay=wdecay,
                        ),  # weight decay = L2 regularization
                        device=device,
                        tokenizer=tokenizer,
                    )

                    metrics = trainer.train(
                        train_dataloader,
                        valid_dataloader,
                        70,
                        15,
                        MODEL_FOLDER + "dump.pt",
                    )
                    with open("hw3/stud/grid_search_results.txt", "a") as f:
                        f.write(
                            "MODEL baseline, LSTM_L {}, FC_L {}, MENTIONS_EMB {}, DECAY {}: BEST VALID ACC: {:0.3f}% AT EPOCH {}, TRAINED FOR {} EPOCHS\n".format(
                                lstm_layers,
                                fc_layers,
                                mentions_size,
                                wdecay,
                                max(metrics["valid_acc"]) * 100,
                                np.argmax(metrics["valid_acc"]) + 1,
                                len(metrics["train_history"]),
                            )
                        )

    # grid search on some hyperparameters for probert and corefseq model
    for model_class in ["probert", "corefseq"]:
        for lrate in [1e-3, 1e-4]:
            for ft_lrate in [1e-5, 4e-6]:
                for wdecay in [0, 1e-5]:

                    tokenizer = tre.Tokenizer(transf_name, add_prefix_space=True)
                    if model_class == "probert":
                        tokenizer.huggingface_tokenizer.add_tokens(["<P>", "<A>", "<B>", "Ġ<P>", "Ġ<A>", "Ġ<B>"])

                    # create model hyperparameters
                    params = ModHParams(
                        use_transformer=transf_name,
                        fine_tune_trans=True,
                        tokenizer_len=len(tokenizer),
                    )

                    # instantiate and train the model
                    crmodel = ProBERT(params, device) if model_class == "probert" else CorefSeq(params, device)
                    groups = [
                        {"params": crmodel.classifier.parameters()},
                        {
                            "params": crmodel.word_embedding.parameters(),
                            "lr": ft_lrate,
                            "weight_decay": 0.0,
                        },
                    ]
                    trainer = Trainer(
                        model=crmodel,
                        loss_function=nn.CrossEntropyLoss(),
                        optimizer=torch.optim.Adam(
                            groups,
                            lr=lrate,
                            weight_decay=wdecay,
                        ),  # weight decay = L2 regularization
                        device=device,
                        tokenizer=tokenizer,
                    )

                    metrics = trainer.train(
                        train_dataloader,
                        valid_dataloader,
                        70,
                        15,
                        MODEL_FOLDER + "dump.pt",
                    )
                    with open("hw3/stud/grid_search_results.txt", "a") as f:
                        f.write(
                            "MODEL {}, LR {}, FT_LR {}, DECAY {}: BEST VALID ACC: {:0.3f}% AT EPOCH {}, TRAINED FOR {} EPOCHS\n".format(
                                model_class,
                                lrate,
                                ft_lrate,
                                wdecay,
                                max(metrics["valid_acc"]) * 100,
                                np.argmax(metrics["valid_acc"]) + 1,
                                len(metrics["train_history"]),
                            )
                        )
else:
    # after this coarse-grained hyperparameter tuning, I can find the best candidate model by looking in the grid_search_results.txt file
    # After finding it, I re-train it to save the weights of best model and plot confusion matrix as well the loss during training and validation
    which_model = "probert"  # "corefseq", "baseline"

    transf_name = "roberta-base"
    tokenizer = tre.Tokenizer(transf_name, add_prefix_space=True)
    if which_model == "probert":
        # in probert I add some "mention tokens", also with the special prefix space Ġ used by RoBERTa tokenizer
        tokenizer.huggingface_tokenizer.add_tokens(["<P>", "<A>", "<B>", "Ġ<P>", "Ġ<A>", "Ġ<B>"])

    # create model hyperparameters, note that the default values of ModHParams constructor are already the best-found hyperparameters
    params = ModHParams(
        use_transformer=transf_name,
        fine_tune_trans=True,
        tokenizer_len=len(tokenizer),
    )

    # instantiate the model
    if which_model == "baseline":
        crmodel = CRModelLSTM(params, device, model_type="pronoun_output")
    elif which_model == "probert":
        crmodel = ProBERT(params, device)
    elif which_model == "corefseq":
        crmodel = CorefSeq(params, device)

    # optimizer initialization
    groups = [
        {"params": crmodel.classifier.parameters()},
        {
            "params": crmodel.word_embedding.parameters(),
            "lr": 4e-6,  # 3e-5, 2e-5
            "weight_decay": 0.0,
        },
    ]
    if which_model == "baseline":
        if params.mentions_embedding_dim != 0:  # if using mention tags
            groups.append({"params": crmodel.mentions_embedding.parameters()})
        if params.pos_embedding_dim != 0:  # if using pos
            groups.append({"params": crmodel.pos_embedding.parameters()})
        groups.append({"params": crmodel.lstm.parameters()})
        groups.append({"params": crmodel.fc.parameters()})

    optimizer = (
        torch.optim.Adam(groups, lr=0.0001, weight_decay=1e-5)
        if params.fine_tune_trans
        else torch.optim.Adam(crmodel.parameters(), lr=0.001)
    )  # weight decay = L2 regularization

    # train
    trainer = Trainer(
        model=crmodel, loss_function=nn.CrossEntropyLoss(), optimizer=optimizer, device=device, tokenizer=tokenizer
    )

    metrics = trainer.train(
        train_dataloader,
        valid_dataloader,
        70,
        15,
        MODEL_FOLDER + which_model + ".pt",
    )
    print(
        "BEST ER VALID ACC: {:0.4f}% AT EPOCH {}".format(
            max(metrics["valid_acc"]) * 100,
            np.argmax(metrics["valid_acc"]) + 1,
        )
    )
    trainer.generate_cm("hw3/stud/cm.png")  # confusion matrix
    Trainer.plot_logs(metrics, "hw3/stud/losses.png")  # loss plot
