# let's start with all the imports
# NOTE: part of this code is taken from notebook #8 - Q&A
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict


# just defining utility functions
def prepare_batch_transformers(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """collate_fn for the train and dev DataLoaders if using transformers, applies padding to data and
    takes into account the fact that [CLS] and [SEP] tokens are added from the transformer
    Args:
        batch (List[Dict]): a list of dictionaries, each dict is a sample from the Dataset
    Returns:
        Dict[str,torch.Tensor]: a batch into a dictionary {x:data, ..., y:labels}
    """
    # extract features from batch
    ids = [sample["id"] for sample in batch]  # plain list
    text = [sample["text"] for sample in batch]  # list of lists
    tagged_text = [sample["tagged_text"] for sample in batch]  # list of lists
    mentions = [sample["mentions"] for sample in batch]  # list of tensors
    tagged_mentions = [sample["tagged_mentions"] for sample in batch]  # list of tensors

    labels = None
    if "label" in batch[0]:
        labels = torch.LongTensor([sample["label"] for sample in batch])  # tensor of numbers

    zero_col = torch.zeros((len(batch), 1), dtype=torch.int64)

    pos = None
    if "pos" in batch[0]:  # if using pos tags
        pos = [sample["pos"] for sample in batch]
        pos = pad_sequence([torch.as_tensor(sample) for sample in pos], batch_first=True)
        pos = torch.cat([zero_col, pos, zero_col], dim=1)

    # convert features to tensor and pad them
    mentions = pad_sequence(mentions, batch_first=True)
    # add padding corresponding to [CLS] and [SEP]
    mentions = torch.cat([zero_col, mentions, zero_col], dim=1)

    tagged_mentions = pad_sequence(tagged_mentions, batch_first=True)
    tagged_mentions = torch.cat([zero_col, tagged_mentions, zero_col], dim=1)

    return {
        "id": ids,
        "text": text,
        "tagged_text": tagged_text,
        "pos": pos,  # goes with text
        "mentions": mentions,
        "tagged_mentions": tagged_mentions,
        "labels": labels,
    }
