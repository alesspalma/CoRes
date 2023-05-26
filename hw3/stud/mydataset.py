# let's start with all the imports
# NOTE: part of this code is taken from notebook #6 - POS tagging
from collections import Counter
import torch
import stanza
from torch.utils.data import Dataset
from typing import List, Dict


class CRDataset(Dataset):
    """My Dataset class for the Coreference Resolution (CR) task"""

    def __init__(
        self,
        data_path: str = None,
        sentences: List[Dict] = None,
        use_pos: bool = False,
    ):
        """constructor of this class
        Args:
            data_path (str, optional): path where to load the whole Dataset, if passed it will have priority. Defaults to None.
            sentences (List[List[str]], optional): if Dataset is already loaded assume is a test set, pass sentences here. Defaults to None.
            use_pos (bool, optional): whether to generate the pos tags or not. Defaults to False.
        """
        # since I'm not interested in going back from index to mention tag, and the mentions tagset is fixed, use a plain dictionary and not a Vocab object
        self.mentions2i = {"_": 1, "P": 2, "A": 3, "B": 4}  # leave 0-th index for padding

        self.labels2i = {"A": 0, "B": 1, "NEITHER": 2}

        # since I'm not interested in going back from index to pos tag, and the upos tagset is fixed, use a plain dictionary and not a Vocab object
        self.upos2i = {
            "ADJ": 1,
            "ADP": 2,
            "ADV": 3,
            "AUX": 4,
            "CCONJ": 5,
            "DET": 6,
            "INTJ": 7,
            "NOUN": 8,
            "NUM": 9,
            "PART": 10,
            "PRON": 11,
            "PROPN": 12,
            "PUNCT": 13,
            "SCONJ": 14,
            "SYM": 15,
            "VERB": 16,
            "X": 17,
        }  # leave 0-th index for padding

        self.encoded_samples = False  # just a flag expressing if dataset has been indexed or not
        self.data_samples = None  # list of dictionaries containing each sample of the original dataset

        if data_path:  # if data path is passed, parse_data will override data_samples
            self.data_samples = self.parse_data(path=data_path)
        elif sentences:
            self.data_samples = self.parse_data(sentences=sentences)

        if use_pos:
            # if we want to use pos tags but they are not precomputed, insert into each sample a new list containing the index-encoded pos tags
            # matching indexes of words in data_samples[i]["text"], e.g.: ['he', 'was'] has tags ['PRON', 'AUX'] represented as [11, 4]
            stanza.download(lang="en", processors="tokenize,pos", verbose=False)
            # note that due to this, the test container could be slower: it needs to download these stanza english models each time

            pos_tagger = stanza.Pipeline(
                lang="en",
                processors="tokenize,pos",
                tokenize_pretokenized=True,
                verbose=False,
            )

            for sample in self.data_samples:
                # pos tag each sentence and collect them in the sample dictionary
                doc = pos_tagger([sample["text"]])
                sample["pos"] = [word.upos for word in doc.sentences[0].words]

        self.index_dataset()

    def parse_data(self, path: str = None, sentences: List[Dict] = None) -> List[Dict]:
        """Function took from the evaluate.py file, reads the tsv dataset

        Args:
            path (str): tsv data file path

        Returns:
            List[Dict]: list of samples
        """
        max_len = 0  # just to print max sentence length
        samples: List[Dict] = []
        pron_counter = Counter()
        if path:  # if path to file is given
            with open(path) as f:
                if "train" not in path:
                    # only validation set has the header
                    next(f)
                for line in f:
                    (
                        id,
                        text,
                        pron,
                        p_offset,
                        entity_A,
                        offset_A,
                        is_coref_A,
                        entity_B,
                        offset_B,
                        is_coref_B,
                        _,
                    ) = line.strip().split("\t")
                    pron_counter[pron.lower()] += 1

                    sample = self.enrich_sample(
                        id,
                        text,
                        pron,
                        p_offset,
                        entity_A,
                        offset_A,
                        entity_B,
                        offset_B,
                        is_coref_A=is_coref_A,
                        is_coref_B=is_coref_B,
                    )

                    max_len = max(max_len, len(sample["text"]))
                    samples.append(sample)
        elif sentences:  # if sentences already parsed are given
            for sentence in sentences:
                pron_counter[sentence["pron"].lower()] += 1
                sample = self.enrich_sample(
                    sentence["id"],
                    sentence["text"],
                    sentence["pron"],
                    sentence["p_offset"],
                    sentence["entity_A"],
                    sentence["offset_A"],
                    sentence["entity_B"],
                    sentence["offset_B"],
                )

                max_len = max(max_len, len(sample["text"]))
                samples.append(sample)

        print(pron_counter, "max sentence length is:", max_len)
        return samples

    def enrich_sample(
        self,
        id,
        text,
        pron,
        p_offset,
        entity_A,
        offset_A,
        entity_B,
        offset_B,
        is_coref_A=None,
        is_coref_B=None,
    ) -> Dict:
        """tokenizes sentence and adds mention tags"""
        p_offset = int(p_offset)
        offset_A = int(offset_A)
        offset_B = int(offset_B)

        label = None
        if is_coref_A is not None:
            label = "NEITHER"  # assign ground truth
            if is_coref_A == "TRUE":
                label = "A"
            if is_coref_B == "TRUE":
                label = "B"

        text = text.strip().split(" ")  # tokenize
        mentions = ["_"] * len(text)
        char_counter = 0

        # fill mentions list with token role in the sentence: pronoun, entity A, entity B, none
        for idx, token in enumerate(text):
            if p_offset >= char_counter and p_offset < char_counter + len(token):
                for off in range(len(pron.strip().split(" "))):
                    mentions[idx + off] = "P"
            elif offset_A >= char_counter and offset_A < char_counter + len(token):
                for off in range(len(entity_A.strip().split(" "))):
                    mentions[idx + off] = "A"
            elif offset_B >= char_counter and offset_B < char_counter + len(token):
                for off in range(len(entity_B.strip().split(" "))):
                    mentions[idx + off] = "B"
            char_counter += len(token) + 1  # + 1 to jump the space that we splitted on
        assert len(text) == len(mentions), "text and mentions don't have same length"

        # another approach: now we create the same sentence but with pronoun, entity A and entity B wrapped by <P>, <A> and <B> tags
        # example: "<A> Bob Suter <A> is the uncle of <B> Dehner <B>. <P> His <P> cousin is Minnesota Wild’s captain."
        tagged_text = text.copy()
        tagged_mentions = mentions.copy()

        for value in ["P", "A", "B"]:
            # add opening mention token
            first = tagged_mentions.index(value)
            tagged_mentions.insert(first, "_")
            tagged_text.insert(first, "<" + value + ">")
            # add closing mention token
            last = max(idx for idx, tok in enumerate(tagged_mentions) if tok == value)
            tagged_mentions.insert(last + 1, "_")
            tagged_text.insert(last + 1, "<" + value + ">")
        assert (
            len(tagged_text) == len(tagged_mentions) == len(text) + 6
        ), "tagged_text and tagged_mentions don't have correct length"

        sample = {
            "id": id,
            "text": text,
            "tagged_text": tagged_text,
            "pron": pron,
            "p_offset": p_offset,
            "entity_A": entity_A,
            "offset_A": offset_A,
            "entity_B": entity_B,
            "offset_B": offset_B,
            "mentions": mentions,
            "tagged_mentions": tagged_mentions,
        }

        if label is not None:  # only if it is not test set, add the label
            sample["label"] = label

        return sample

    def index_dataset(self):
        """Indexes mentions, and eventually also labels and pos tags"""

        for sample in self.data_samples:
            # no need to encode words because we will use transformer's tokenizer
            sample["mentions"] = torch.LongTensor([self.mentions2i[token] for token in sample["mentions"]])
            sample["tagged_mentions"] = torch.LongTensor(
                [self.mentions2i[token] for token in sample["tagged_mentions"]]
            )

            if "label" in sample:
                sample["label"] = self.labels2i[sample["label"]]
            if "pos" in sample:  # if using pos tags
                sample["pos"] = torch.LongTensor([self.upos2i[token] for token in sample["pos"]])

        self.encoded_samples = True
        return

    def __len__(self) -> int:
        if self.encoded_samples is False:
            raise RuntimeError("Trying to retrieve length but index_dataset has not been invoked yet!")
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        """returns a dict with idx-th encoded sentence, its pos tags and its list of labels
        Args:
            idx (int): index of sentence to retrieve
        Returns:
            Dict[str,torch.LongTensor]: a dictionary mapping every information of a sample
        """
        if self.encoded_samples is False:
            raise RuntimeError("Trying to retrieve elements but index_dataset has not been invoked yet!")
        return self.data_samples[idx]
