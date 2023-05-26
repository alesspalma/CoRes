import re
import torch
import transformers_embedder as tre
import os
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from stud.mymodel import ModHParams, ProBERT
from stud.mydataset import CRDataset
from stud.myutils import prepare_batch_transformers
from model import Model


def build_model_123(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2 and 3 of the Coreference resolution pipeline.
            1: Ambiguous pronoun identification.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(True, True)


def build_model_23(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2 and 3 of the Coreference resolution pipeline.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(False, True)


def build_model_3(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements step 3 of the Coreference resolution pipeline.
            3: Coreference resolution
    """
    return StudentModel(device)


class RandomBaseline(Model):
    def __init__(self, predict_pronoun: bool, predict_entities: bool):
        self.pronouns_weights = {
            "his": 904,
            "her": 773,
            "he": 610,
            "she": 555,
            "him": 157,
        }
        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities

    def predict(self, sentences: List[Dict]) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        predictions = []
        for sent in sentences:
            text = sent["text"]
            toks = re.sub("[.,'`()]", " ", text).split(" ")
            if self.predict_pronoun:
                prons = [tok.lower() for tok in toks if tok.lower() in self.pronouns_weights]
                if prons:
                    pron = np.random.choice(prons, 1, self.pronouns_weights)[0]
                    pron_offset = text.lower().index(pron)
                    if self.pred_entities:
                        entity = self.predict_entity(predictions, pron, pron_offset, text, toks)
                    else:
                        entities = [sent["entity_A"], sent["entity_B"]]
                        entity = self.predict_entity(predictions, pron, pron_offset, text, toks, entities)
                    predictions.append(((pron, pron_offset), entity))
                else:
                    predictions.append(((), ()))
            else:
                pron = sent["pron"]
                pron_offset = sent["p_offset"]
                if self.pred_entities:
                    entity = self.predict_entity(predictions, pron, pron_offset, text, toks)
                else:
                    entities = [
                        (sent["entity_A"], sent["offset_A"]),
                        (sent["entity_B"], sent["offset_B"]),
                    ]
                    entity = self.predict_entity(predictions, pron, pron_offset, text, toks, entities)
                predictions.append(((pron, pron_offset), entity))
        return predictions

    def predict_entity(self, predictions, pron, pron_offset, text, toks, entities=None):
        entities = entities if entities is not None else self.predict_entities(entities, toks)
        entity_idx = np.random.choice([0, len(entities) - 1], 1)[0]
        return entities[entity_idx]

    def predict_entities(self, entities, toks):
        offset = 0
        entities = []
        for tok in toks:
            if tok != "" and tok[0].isupper():
                entities.append((tok, offset))
            offset += len(tok) + 1
        return entities


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device: str) -> None:

        # to avoid warnings from the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = torch.device(device)

        self.label_vocab = {0: "A", 1: "B", 2: "NEITHER"}

        # I will instantiate those in the first run of predict method, to allow the server to go up in just 10 seconds
        self.hparams = None
        self.tokenizer = None
        self.model = None

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        # Each prediction is a tuple of two tuples, like: [..., ( ('her', 274), ('Pauline', 418) ), ...]
        # if neither entity A and B are referenced then the tuple is [..., ( ('her', 274), () ), ...]

        if self.model is None:
            # initialize tokenizer and model
            self.tokenizer = tre.Tokenizer("roberta-base", add_prefix_space=True)
            # add mention tokens, also with the special prefix space Ġ used by RoBERTa tokenizer
            self.tokenizer.huggingface_tokenizer.add_tokens(["<P>", "<A>", "<B>", "Ġ<P>", "Ġ<A>", "Ġ<B>"])

            self.hparams = ModHParams(
                use_transformer="roberta-base",
                fine_tune_trans=False,
                tokenizer_len=len(self.tokenizer),
            )

            self.model = ProBERT(self.hparams, self.device).to(self.device)
            self.model.load_state_dict(
                torch.load(
                    "model/probert.pt",
                    map_location=self.device,
                )
            )
            self.model.eval()

        dataset = CRDataset(sentences=tokens)
        dataloader = DataLoader(dataset, batch_size=len(tokens), collate_fn=prepare_batch_transformers, shuffle=False)

        output = []
        with torch.no_grad():
            for batch in dataloader:  # just one batch
                batch["tagged_text"] = self.tokenizer(
                    batch["tagged_text"], padding=True, return_tensors=True, is_split_into_words=True
                ).to(self.device)
                batch["tagged_mentions"] = batch["tagged_mentions"].to(self.device)
                if batch["pos"] is not None:
                    batch["pos"] = batch["pos"].to(self.device)  # if using pos, move to device

                logits = self.model(batch)
                predictions = torch.argmax(logits, -1)

                # update output list
                for idx, predicted in enumerate(predictions):
                    predicted_class = self.label_vocab[predicted.item()]

                    pronoun = (tokens[idx]["pron"], tokens[idx]["p_offset"])  # pronoun assumed always correct
                    if predicted_class == "A":
                        entity = (tokens[idx]["entity_A"], tokens[idx]["offset_A"])
                    elif predicted_class == "B":
                        entity = (tokens[idx]["entity_B"], tokens[idx]["offset_B"])
                    else:
                        entity = ()
                    output.append((pronoun, entity))

        return output
