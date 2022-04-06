import collections
import json
import os
import pickle
import shutil
from typing import List, Optional, Tuple, Dict, Callable, Any

import lz4.frame
import marisa_trie
import numpy as np
import torch
from spacy.tokens import Doc
from torch import nn
from tqdm import tqdm

from gnn_lib.data import utils, tokenization
from gnn_lib.data.utils import flatten
from gnn_lib.modules import utils as mod_utils, embedding, encoders
from gnn_lib.utils import common, io, to


WORD_PLACEHOLDER = "[W]"


class Vectorizer:
    def __init__(self, context_length: int, **kwargs: Any) -> None:
        self.context_length = context_length

    def vectorize(self, left_context: str, right_context: str) -> np.array:
        raise NotImplementedError

    def vectorize_batch(self, inputs: List[Tuple[str, str]]) -> np.array:
        return np.stack([self.vectorize(*ipt) for ipt in inputs])

    def to_device(self, device: torch.device) -> None:
        pass

    @property
    def dim(self) -> int:
        raise NotImplementedError


class _BaseWordToVecVectorizer(Vectorizer):
    def __init__(
            self,
            context_length: int,
            mode: str = "center_distance",
            **kwargs: Any
    ) -> None:
        super().__init__(context_length, **kwargs)

        self.total_length = 2 * context_length + 1

        self.mode = mode
        if self.mode == "uniform":
            # every word is equally weighted, ignoring distance from center word
            self.left_context_weights = self.right_context_weights = np.array([1] * context_length)[:, None]
        elif self.mode == "center_distance":
            # words close to center word are weighted more
            self.left_context_weights = np.array([i + 1 for i in range(0, context_length)])[:, None]
            self.right_context_weights = np.array([i for i in range(context_length, 0, -1)])[:, None]
        else:
            raise RuntimeError(f"unknown vectorizer mode {mode}")

    def _words_to_vectors(self, contexts: List[str]) -> np.array:
        raise NotImplementedError

    def vectorize_batch(self, inputs: List[Tuple[str, str]]) -> np.array:
        batch_left_context_words = []
        batch_right_context_words = []
        for left_context, right_context in inputs:
            left_context_words = utils.tokenize_words_regex(left_context)[0]
            batch_left_context_words.append(left_context_words)
            right_context_words = utils.tokenize_words_regex(right_context)[0]
            batch_right_context_words.append(right_context_words)
            assert len(left_context_words) <= self.context_length and len(right_context_words) <= self.context_length

        num_batch_left_context_words = [len(words) for words in batch_left_context_words]
        total_batch_left_context_words = sum(num_batch_left_context_words)
        num_batch_right_context_words = [len(words) for words in batch_right_context_words]
        total_batch_right_context_words = sum(num_batch_right_context_words)

        batch_left_context_vectors, batch_right_context_vectors = mod_utils.split(
            self._words_to_vectors(flatten(batch_left_context_words) + flatten(batch_right_context_words)),
            [total_batch_left_context_words, total_batch_right_context_words]
        )

        batch_left_context_vectors = mod_utils.split(batch_left_context_vectors, num_batch_left_context_words)
        batch_right_context_vectors = mod_utils.split(batch_right_context_vectors, num_batch_right_context_words)

        output_vectors = []
        for left_context, right_context in zip(
                batch_left_context_vectors, batch_right_context_vectors
        ):
            left_context_weights = self.left_context_weights[self.context_length - len(left_context):]
            right_context_weights = self.right_context_weights[:len(right_context)]

            vectors = []
            if len(left_context):
                vectors.append(np.mean(left_context * left_context_weights, axis=0))
            if len(right_context):
                vectors.append(np.mean(right_context * right_context_weights, axis=0))

            if len(vectors):
                vector = np.sum(np.stack(vectors), axis=0) / (
                        np.sum(left_context_weights) + np.sum(right_context_weights)
                )
            else:
                # if there are no context vectors, return zero vector
                vector = np.zeros(self.dim, dtype=float)
            output_vectors.append(vector)

        return np.stack(output_vectors)

    def vectorize(self, left_context: str, right_context: str) -> np.array:
        return self.vectorize_batch([(left_context, right_context)])[0]


class FastTextVectorizer(_BaseWordToVecVectorizer):
    def __init__(
            self,
            context_length: int,
            mode: str = "center_distance",
            model_path: Optional[str] = "cc.en.300.bin",
            **kwargs: Any
    ) -> None:
        super().__init__(context_length, mode)
        import fasttext
        self.ft = fasttext.load_model(model_path)

    def _words_to_vectors(self, words: List[str]) -> np.array:
        return np.stack([self.ft.get_word_vector(word) for word in words])

    @property
    def dim(self) -> int:
        return self.ft.get_dimension()


# class BertVectorizer(Vectorizer):
#     def __init__(self) -> None:
#         super().__init__("", 0)
#         from transformers import BertModel, BertTokenizer, BertConfig
#         self.tok = BertTokenizer.from_pretrained("bert-base-cased")
#         self.device = mod_utils.cuda_or_cpu(force_cpu=True)
#         self.bert: BertModel = BertModel.from_pretrained("bert-base-cased")
#         self.bert.to(self.device)
#         self.bert.eval()
#         self.config: BertConfig = BertConfig.from_pretrained("bert-base-cased")
#
#     def vectorize(self, left_context: str, word: str, right_context: str) -> np.array:
#         tensors = self.tok(left_context + word + right_context, return_tensors="pt").to(self.device)
#         with torch.inference_mode():
#             outputs = self.bert(**tensors).last_hidden_state.cpu().numpy()
#         return np.mean(outputs, axis=1)
#
#     def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
#         input_strs = [l + w + r for w, l, r in inputs]
#         tensors = self.tok(input_strs, padding=True, return_tensors="pt").to(self.device)
#         with torch.inference_mode():
#             outputs = self.bert(**tensors).last_hidden_state.cpu().numpy()
#         vectors = []
#         for i, att_mask in enumerate(tensors["attention_mask"]):
#             vectors.append(outputs[i, :att_mask.sum()].mean(0))
#         return np.stack(vectors)
#
#     def to_device(self, device: torch.device) -> None:
#         self.device = device
#         self.bert.to(self.device)
#
#     @property
#     def dim(self) -> int:
#         return self.config.hidden_size


class StringVectorizer(Vectorizer):
    def vectorize(self, left_context: str, right_context: str) -> np.array:
        return left_context + WORD_PLACEHOLDER + right_context

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        return [self.vectorize(*item) for item in inputs]

    @property
    def dim(self) -> int:
        return -1


class CustomNeuralVectorizerModel(nn.Module):
    @torch.inference_mode()
    def contexts_to_vectors(self, contexts: List[Tuple[str, str]]) -> np.array:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class CustomNeuralVectorizer(Vectorizer):
    def __init__(
            self,
            context_length: int,
            vectorizer_path: Optional[str] = None,
            force_cpu: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(context_length)

        assert vectorizer_path is not None
        self.device = mod_utils.cuda_or_cpu(force_cpu=force_cpu)
        self.model: CustomNeuralVectorizerModel = torch.load(vectorizer_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def vectorize(self, left_context: str, right_context: str) -> np.array:
        return self.vectorize_batch([(left_context, right_context)])[0]

    def vectorize_batch(self, inputs: List[Tuple[str, str]]) -> np.array:
        return self.model.contexts_to_vectors(inputs)

    @property
    def dim(self) -> int:
        return self.model.dim


class CharTransformer(CustomNeuralVectorizerModel):
    def __init__(self) -> None:
        super().__init__()
        self.max_length = 128 - len(WORD_PLACEHOLDER)
        self.max_length_half = self.max_length // 2
        self.hidden_dim = 128
        self.tok = tokenization.CharTokenizer()
        self.pad_token_id = self.tok.token_to_id(tokenization.PAD)

        self.emb = embedding.TensorEmbedding(
            hidden_dim=self.hidden_dim,
            num_embeddings=self.tok.vocab_size,
            max_length=self.max_length,
            cfg=embedding.TensorEmbeddingConfig(dropout=0.1),
            padding_idx=self.pad_token_id
        )
        self.encoder = encoders.Transformer(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
            dropout=0.1
        )

    def forward(self, inputs: List[str]) -> torch.Tensor:
        device = mod_utils.device_from_model(self)
        token_ids = []
        lengths = []
        for i, ipt in enumerate(inputs):
            tokens = self.tok.tokenize(ipt)
            token_ids.append(torch.tensor(tokens, dtype=torch.long))
            lengths.append(len(tokens))

        token_ids_tensor = to(mod_utils.pad(token_ids, float(self.pad_token_id)).long(), device)
        padding_mask = mod_utils.padding_mask(token_ids_tensor, lengths)

        emb = self.emb(token_ids_tensor)
        enc = self.encoder(emb, padding_mask=padding_mask)
        outputs = []
        for vec, length in zip(enc, lengths):
            outputs.append(torch.mean(vec[:length, :], dim=0))
        return torch.stack(outputs)

    @torch.inference_mode()
    def contexts_to_vectors(self, contexts: List[Tuple[str, str]]) -> np.array:
        return self.forward(
            [l[-self.max_length_half:] + WORD_PLACEHOLDER + r[:self.max_length_half] for l, r in contexts]
        ).cpu().numpy()

    @property
    def dim(self) -> int:
        return self.hidden_dim


def get_vectorizer(name: str, context_length: int, **kwargs: Any) -> Vectorizer:
    if name == "ft":
        return FastTextVectorizer(context_length, **kwargs)
    # elif name == "bert":
    #     return BertVectorizer()
    elif name == "string":
        return StringVectorizer(context_length, **kwargs)
    elif name == "custom":
        return CustomNeuralVectorizer(context_length, **kwargs)
    else:
        raise ValueError(f"unknown vectorizer {name}")


class NNIndex:
    @staticmethod
    def get_index(space: str):
        import nmslib
        if space == "leven":
            data_type = nmslib.DataType.OBJECT_AS_STRING
            dist_type = nmslib.DistType.INT
        elif space == "normleven":
            data_type = nmslib.DataType.OBJECT_AS_STRING
            dist_type = nmslib.DistType.FLOAT
        else:
            data_type = nmslib.DataType.DENSE_VECTOR
            dist_type = nmslib.DistType.FLOAT
        return nmslib.init(method="hnsw", space=space, data_type=data_type, dtype=dist_type)

    def __init__(
            self,
            directory: str,
            num_threads: int = len(os.sched_getaffinity(0)),
            ef_search: int = 100,
            **kwargs: Any
    ) -> None:
        self.directory = directory

        self.index_file = os.path.join(self.directory, "index")
        self.lmdb_file = os.path.join(self.directory, "lmdb")

        if (
                not os.path.exists(self.directory)
                or not os.path.exists(self.index_file)
                or not os.path.exists(self.lmdb_file)
        ):
            raise ValueError(f"index directory {self.directory} or index file {self.index_file} or "
                             f"lmdb file {self.lmdb_file} do not exists")

        self.logger = common.get_logger("NN_INDEX")
        self.logger.info("Opening nearest neighbor index")

        self.lmdb = utils.open_lmdb(self.lmdb_file)
        self.txn = self.lmdb.begin()

        self.params = pickle.loads(self.txn.get("params".encode("utf8")))
        self.size = pickle.loads(self.txn.get("num_elements".encode("utf8")))
        self.num_threads = num_threads

        if self.params["vectorizer"] == "custom":
            vectorizer_path = os.path.join(directory, "custom_vectorizer.pt")
            assert os.path.exists(vectorizer_path), "expected a file named custom_vectorizer.pt in the index directory"
        else:
            vectorizer_path = None

        self.vectorizer = get_vectorizer(
            self.params["vectorizer"],
            self.params["context_length"],
            vectorizer_path=vectorizer_path
        )

        self.index = self.get_index(self.params["space"])
        self.index.loadIndex(self.index_file, load_data=self.params["space"] not in {"l2", "cosinesimil"})
        self.index.setQueryTimeParams({"efSearch": ef_search})

    @staticmethod
    def create(
            files: List[str],
            out_directory: str,
            context_length: int,
            vectorizer_name: str,
            dist: str,
            dictionary_file: Optional[str] = None,
            ef_construction: int = 200,
            m: int = 16,
            post: int = 0,
            min_freq: int = 0,
            **kwargs: Any
    ) -> None:
        assert context_length > 0, "context length must be larger than 0"

        logger = common.get_logger("NN_INDEX_CREATION")
        logger.info(f"Creating index with context length {context_length}, "
                    f"{vectorizer_name} vectorizer and {dist} distance at {out_directory}")

        os.makedirs(out_directory, exist_ok=True)
        index_out_file = os.path.join(out_directory, "index")
        lmdb_out_file = os.path.join(out_directory, "lmdb")

        if dist == "euclidean":
            space = "l2"
        elif dist == "cosine":
            space = "cosinesimil"
        elif dist == "edit_distance":
            space = "leven"
            assert vectorizer_name == "string", "for edit_distance distance vectorizer must be string"
        elif dist == "norm_edit_distance":
            space = "normleven"
            assert vectorizer_name == "string", "for norm_edit_distance distance vectorizer must be string"
        else:
            raise ValueError(f"Unknown distance type {dist}")

        if vectorizer_name == "custom":
            assert "vectorizer_path" in kwargs, \
                "when vectorizer is custom, vectorizer_path keyword argument must be specified"
            vectorizer_path = kwargs.pop("vectorizer_path")
            # copy vectorizer from its current location to the output directory, so we easily instantiate the index
            # once its created
            dst_vectorizer_path = os.path.join(out_directory, "custom_vectorizer.pt")
            shutil.copy2(vectorizer_path, dst_vectorizer_path)
            vectorizer_path = dst_vectorizer_path
        else:
            vectorizer_path = None

        vectorizer = get_vectorizer(vectorizer_name, context_length, vectorizer_path=vectorizer_path)
        dictionary = io.dictionary_from_file(dictionary_file) if dictionary_file is not None else None

        index = NNIndex.get_index(space)
        lmdb = utils.open_lmdb(lmdb_out_file, write=True)
        txn = lmdb.begin(write=True)
        txn.drop(lmdb.open_db())

        contexts: Dict[Tuple[str, str], Dict[str, int]] = {}
        context_counts: Dict[Tuple[str, str], int] = {}

        logger.info(f"Processing {len(files):,} files")

        total = 0
        invalid = 0
        for file in tqdm(files, desc="processing files", disable=common.disable_tqdm()):
            with open(file, "r", encoding="utf8") as inf:
                for line in tqdm(inf, total=io.line_count(file), leave=False, disable=common.disable_tqdm()):
                    sequence = json.loads(line)["sequence"]
                    batch = NNIndex.prepare_sequence(sequence, context_length)
                    for left_context, word, right_context in batch:
                        total += 1
                        # if dictionary is given only record stats for words that are in the dictionary
                        if dictionary is not None and word not in dictionary:
                            invalid += 1
                            continue

                        # do not add the same context multiple times to the index, just increase its word counts
                        seen_context = (left_context, right_context) in contexts
                        if not seen_context:
                            contexts[(left_context, right_context)] = {word: 1}
                            context_counts[(left_context, right_context)] = 1
                        else:
                            if word not in contexts[(left_context, right_context)]:
                                contexts[(left_context, right_context)][word] = 1
                            else:
                                contexts[(left_context, right_context)][word] += 1
                            context_counts[(left_context, right_context)] += 1

        contexts = {
            ctx: words
            for ctx, words in contexts.items()
            if context_counts[ctx] >= min_freq
        }

        num_elements = len(contexts)
        logger.info(f"Adding {num_elements:,} elements to index")

        indices = []
        samples = []
        frequencies = []

        def add_batch() -> None:
            nonlocal indices
            nonlocal samples
            nonlocal frequencies

            context_vectors = vectorizer.vectorize_batch([ctx for (ctx, _) in samples])
            index.addDataPointBatch(ids=indices, data=context_vectors)

            for ((left_context, right_context), words_and_frequencies), idx, freq in zip(
                    samples, indices, frequencies
            ):
                data = pickle.dumps({
                    "left_context": left_context,
                    "right_context": right_context,
                    "context_frequency": freq,
                    "word_list": words_and_frequencies,
                })
                data = lz4.frame.compress(data, compression_level=16)
                assert txn.put(f"{idx}".encode("utf8"), data)

            indices = []
            samples = []
            frequencies = []

        for i, (ctx, word_frequencies) in tqdm(
                enumerate(contexts.items()),
                total=len(contexts),
                desc="adding elements to index",
                leave=False,
                disable=common.disable_tqdm()
        ):
            indices.append(i)
            frequencies.append(context_counts[ctx])

            words_and_frequencies = [(w, f) for w, f in word_frequencies.items()]
            # sort descending by frequency
            words_and_frequencies = sorted(words_and_frequencies, key=lambda item: -item[1])

            samples.append((ctx, words_and_frequencies))

            if len(indices) % 64 == 0:
                add_batch()

        if len(indices):
            add_batch()

        if dictionary is not None:
            logger.info(f"{100 * invalid / total:.2f}% of all unique (left_context, word, right_context) items "
                        f"were invalid and thus not added to index")

        logger.info(f"Computing index with {num_elements:,} elements")
        index_time_params = {
            "M": m,
            "indexThreadQty": int(os.getenv("GNN_LIB_INDEX_NUM_THREADS", len(os.sched_getaffinity(0)))),
            "efConstruction": ef_construction,
            "post": post
        }
        index.createIndex(index_time_params, print_progress=True)
        index.saveIndex(index_out_file, save_data=space not in {"l2", "cosinesimil"})

        assert txn.put(
            "params".encode("utf8"),
            pickle.dumps({"context_length": context_length,
                          "space": space,
                          "vectorizer": vectorizer_name})
        )
        assert txn.put("num_elements".encode("utf8"), pickle.dumps(num_elements))
        txn.commit()
        lmdb.close()

        logger.info(f"Saved {num_elements:,} items in the index")

    @staticmethod
    def prepare_sequence(sequence: str, context_length: int) -> List[Tuple[str, str, str]]:
        sequence = utils.clean_sequence(sequence)
        words, whitespaces = utils.tokenize_words_regex(sequence)
        return NNIndex.prepare_words(words, whitespaces, context_length)

    @staticmethod
    def prepare_words(words: List[str], whitespaces: List[bool], context_length: int) -> List[Tuple[str, str, str]]:
        assert len(words) == len(whitespaces)
        whitespaces[-1] = False
        words = [""] * context_length + words + [""] * context_length
        whitespaces = [False] * context_length + whitespaces + [False] * context_length
        outputs = []
        for i in range(context_length, len(words) - context_length):
            word = words[i]
            left_context = utils.de_tokenize_words(
                words[i - context_length:i],
                whitespaces[i - context_length: i]
            ).lstrip().lower()
            right_context = " " * whitespaces[i] + utils.de_tokenize_words(
                words[i + 1: i + context_length + 1],
                whitespaces[i + 1: i + context_length + 1]
            ).rstrip().lower()
            outputs.append((left_context, word, right_context))
        return outputs

    def batch_retrieve(self, batch: List[Tuple[str, str]], n_neighbors: int) -> List[utils.Neighbors]:
        batch = [(l.lstrip().lower(), r.rstrip().lower()) for l, r in batch]
        vectors_batch = self.vectorizer.vectorize_batch(batch)

        neighbors_batch = self.index.knnQueryBatch(
            queries=vectors_batch, k=n_neighbors, num_threads=self.num_threads
        )

        return [self._to_neighbors(neighbors, distances) for neighbors, distances in neighbors_batch]

    def batch_retrieve_from_docs(self, doc_batch: List[Doc], n_neighbors: int) -> List[List[utils.Neighbors]]:
        all_batch = []
        lengths = []
        for doc in doc_batch:
            words = list(t.text for t in doc)
            whitespaces = list(t.whitespace_ == " " for t in doc)
            batch = self.prepare_words(words, whitespaces, self.params["context_length"])
            all_batch.extend([(left_context, right_context) for left_context, _, right_context in batch])
            lengths.append(len(batch))

        neighbors_list = self.batch_retrieve(all_batch, n_neighbors)
        return mod_utils.split(neighbors_list, lengths)

    def _get_item(self, idx: int) -> Dict:
        data = self.txn.get(f"{idx}".encode("utf8"))
        assert data is not None
        data = lz4.frame.decompress(data)
        data = pickle.loads(data)
        return data

    def _to_neighbors(self, neighbor_ids: List[int], distances: List[float]) -> utils.Neighbors:
        # get the data for all neighbors
        data = [self._get_item(neighbor_id) for neighbor_id in neighbor_ids]

        # sort the indices according to distance, then frequency
        indices = sorted(list(range(len(data))), key=lambda idx: (distances[idx], -data[idx]["context_frequency"]))

        # create neighbor object with correct ordering
        word_lists = [data[idx]["word_list"] for idx in indices]
        left_contexts = [data[idx]["left_context"] for idx in indices]
        right_contexts = [data[idx]["right_context"] for idx in indices]
        frequencies = [data[idx]["context_frequency"] for idx in indices]
        distances = [distances[idx] for idx in indices]
        return utils.Neighbors(
            left_contexts=left_contexts,
            right_contexts=right_contexts,
            context_frequencies=frequencies,
            word_lists=word_lists,
            distances=distances
        )

    def retrieve(self, context: Tuple[str, str], n_neighbors: int) -> utils.Neighbors:
        l, r = context
        context_vec = self.vectorizer.vectorize(l.lstrip().lower(), r.rstrip().lower())
        neighbors, distances = self.index.knnQuery(context_vec, k=n_neighbors)
        return self._to_neighbors(neighbors, distances)


def get_neighbor_fn(index: NNIndex, num_neighbors: int) -> Callable[[List[Doc]], List[List[utils.Neighbors]]]:
    def _neigh(docs: List[Doc]) -> List[List[utils.Neighbors]]:
        return index.batch_retrieve_from_docs(docs, num_neighbors)

    return _neigh


class PrefixIndex:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "rb") as inf:
            self.trie, self.dict = pickle.load(inf)

    @staticmethod
    def create(
            dictionary: Dict[str, int],
            save_path: str
    ) -> None:
        trie = marisa_trie.Trie(dictionary.keys())

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as of:
            pickle.dump((trie, dictionary), of)

    def retrieve(self, prefix: str) -> List[Tuple[str, int]]:
        results = []
        keys = self.trie.keys(prefix)
        for key in keys:
            results.append((key, self.dict[key]))
        return sorted(results, key=lambda e: e[1], reverse=True)

    def batch_retrieve(self, prefixes: List[str]) -> List[List[Tuple[str, int]]]:
        return [self.retrieve(prefix) for prefix in prefixes]
