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
from gnn_lib.modules import utils as mod_utils, embedding, encoders
from gnn_lib.utils import common, io, to


class Vectorizer:
    def __init__(self, context_length: int, **kwargs: Any) -> None:
        self.context_length = context_length

    def vectorize(self, left_context: str, word: str, right_context: str) -> np.array:
        raise NotImplementedError

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        vectors = []
        for input_ in inputs:
            vectors.append(self.vectorize(*input_))
        return np.stack(vectors)

    def to_device(self, device: torch.device) -> None:
        pass

    @property
    def dim(self) -> int:
        raise NotImplementedError


class FastTextVectorizer(Vectorizer):
    def __init__(
            self,
            context_length: int,
            model_path: Optional[str] = "cc.en.300.bin",
            mode: str = "center",
            center_weight: float = 3.0,
            use_positional_encodings: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(context_length)
        import fasttext
        self.ft = fasttext.load_model(model_path)

        total_length = 2 * context_length + 1

        # transformer sinusoidal positional embeddings
        self.pe = np.zeros((total_length, self.dim), dtype=float)
        position = np.arange(0, total_length, dtype=float)[:, None]
        div_term = np.exp(
            np.arange(0, self.dim, 2, dtype=float) *
            -(np.log(10000.0) / self.dim)
        )
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

        # always use uniform when context_length == 0
        self.mode = mode if context_length > 0 else "uniform"
        self.center_weight = center_weight
        self.use_pe = use_positional_encodings

    def vectorize(self, left_context: str, word: str, right_context: str) -> np.array:
        left_context_words = utils.tokenize_words_regex(left_context)[0]
        right_context_words = utils.tokenize_words_regex(right_context)[0]
        assert len(left_context_words) == len(right_context_words) == self.context_length

        left_context_vectors = []
        for i, word in enumerate(left_context_words):
            vector = self.ft.get_word_vector(word)
            if self.use_pe:
                vector += self.pe[i]
            left_context_vectors.append(vector)

        word_vector = self.ft.get_word_vector(word)
        if self.use_pe:
            word_vector += self.pe[self.context_length]

        right_context_vectors = []
        for i, word in enumerate(right_context_words):
            vector = self.ft.get_word_vector(word)
            if self.use_pe:
                vector += self.pe[self.context_length + 1 + i]
            left_context_vectors.append(vector)

        all_vectors = []
        if len(left_context_vectors):
            all_vectors.append(np.stack(left_context_vectors).mean(axis=0))
        all_vectors.append(word_vector)
        if len(right_context_vectors):
            all_vectors.append(np.stack(right_context_vectors).mean(axis=0))
        vectors = np.stack(all_vectors)

        if self.mode == "uniform":
            return vectors.mean(axis=0)
        elif self.mode == "center":
            vectors[self.context_length] *= self.center_weight
            return vectors.sum(axis=0) / (2 + self.center_weight)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

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
    def __init__(self, context_length: int, **kwargs: Any) -> None:
        super().__init__(context_length)

    def vectorize(self, left_context: str, word: str, right_context: str) -> np.array:
        return left_context + word + right_context

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        return [self.vectorize(*item) for item in inputs]

    @property
    def dim(self) -> int:
        return -1


class CustomNeuralVectorizerModel(nn.Module):
    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class CustomNeuralVectorizer(Vectorizer):
    def __init__(
            self,
            context_length: int,
            vectorizer_path: Optional[str] = None,
            force_cpu: bool = True
    ) -> None:
        super().__init__(context_length)
        assert vectorizer_path is not None
        self.device = mod_utils.cuda_or_cpu(force_cpu=force_cpu)
        self.model: CustomNeuralVectorizerModel = torch.load(vectorizer_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def vectorize(self, left_context: str, word: str, right_context: str) -> np.array:
        return self.model.vectorize_batch([(left_context, word, right_context)])[0]

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        return self.model.vectorize_batch(inputs)

    @property
    def dim(self) -> int:
        return self.model.dim


class CharTransformer(CustomNeuralVectorizerModel):
    def __init__(self) -> None:
        super().__init__()
        self.max_length = 64
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
            tokens = self.tok.tokenize(ipt[:self.max_length])
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
    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        return self.forward([l + w + r for l, w, r in inputs]).cpu().numpy()

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
            **kwargs: Any
    ) -> None:
        logger = common.get_logger("NN_INDEX_CREATION")

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

        counts = {}

        total = 0
        invalid = 0
        for file in tqdm(files, desc="processing files", disable=common.disable_tqdm()):
            with open(file, "r", encoding="utf8") as inf:
                for line in tqdm(inf, total=io.line_count(file), leave=False, disable=common.disable_tqdm()):
                    sequence = json.loads(line)["sequence"]
                    batch = NNIndex.prepare_sequence(sequence, context_length)
                    for sample in batch:
                        total += 1
                        if dictionary is not None and sample[1] not in dictionary:
                            invalid += 1
                            continue

                        # do not add the same things multiple times to the index, just increase its count
                        if sample in counts:
                            counts[sample] += 1
                            continue

                        counts[sample] = 1

        for i, (sample, freq) in tqdm(
                enumerate(counts.items()),
                total=len(counts),
                desc="adding elements to index",
                leave=False,
                disable=common.disable_tqdm()
        ):
            context_vec = vectorizer.vectorize(*sample)

            index.addDataPoint(id=i, data=context_vec)

            left_context, word, right_context = sample
            data = pickle.dumps({
                "word": word,
                "left_context": left_context,
                "right_context": right_context,
                "frequency": freq
            })
            data = lz4.frame.compress(data, compression_level=16)
            assert txn.put(f"{i}".encode("utf8"), data)

        if dictionary is not None:
            logger.info(f"{100 * invalid / total:.2f}% of all unique (left_context, word, right_context) items "
                        f"were invalid and thus not added to index")

        logger.info("Creating index")
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
        assert txn.put("num_elements".encode("utf8"), pickle.dumps(len(counts)))
        txn.commit()
        lmdb.close()

        logger.info(f"Saved {len(counts)} items in the index")

    @staticmethod
    def prepare_sequence(sequence: str, context_length: int) -> List[Tuple[str, str, str]]:
        sequence = utils.clean_sequence(sequence)
        words, whitespaces = utils.tokenize_words_regex(sequence)
        return NNIndex.prepare_words(words, whitespaces, context_length)

    @staticmethod
    def prepare_words(words: List[str], whitespaces: List[bool], context_length: int) -> List[Tuple[str, str, str]]:
        if context_length <= 0:
            return [("", word, "") for word in words]

        # pad words and whitespaces
        whitespaces[-1] = False
        words = [""] * context_length + words + [""] * context_length
        whitespaces = [False] * context_length + whitespaces + [False] * context_length
        outputs = []
        for i in range(context_length, len(words) - context_length):
            word = words[i]
            left_context = utils.de_tokenize_words(
                words[i - context_length:i],
                whitespaces[i - context_length: i]
            ).lstrip()
            right_context = " " * whitespaces[i] + utils.de_tokenize_words(
                words[i + 1: i + context_length + 1],
                whitespaces[i + 1: i + context_length + 1]
            ).rstrip()
            outputs.append((left_context, word, right_context))
        return outputs

    def batch_retrieve(self, words_batch: List[Tuple[str, str, str]], n_neighbors: int) -> List[utils.Neighbors]:
        vectors_batch = self.vectorizer.vectorize_batch(words_batch)

        neighbors_batch = self.index.knnQueryBatch(
            queries=vectors_batch, k=n_neighbors, num_threads=self.num_threads
        )

        return [self._to_neighbors(neighbors, distances) for neighbors, distances in neighbors_batch]

    def batch_retrieve_from_docs(self, doc_batch: List[Doc], n_neighbors: int) -> List[List[utils.Neighbors]]:
        all_words_batch = []
        lengths = []
        for doc in doc_batch:
            words = list(t.text for t in doc)
            whitespaces = list(t.whitespace_ == " " for t in doc)
            words_batch = self.prepare_words(words, whitespaces, self.params["context_length"])
            all_words_batch.extend(words_batch)
            lengths.append(len(words_batch))

        neighbors_list = self.batch_retrieve(all_words_batch, n_neighbors)
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
        indices = sorted(list(range(len(data))), key=lambda idx: (distances[idx], -data[idx]["frequency"]))

        # create neighbor object with correct ordering
        words = [data[idx]["word"] for idx in indices]
        left_contexts = [data[idx]["left_context"] for idx in indices]
        right_contexts = [data[idx]["right_context"] for idx in indices]
        frequencies = [data[idx]["frequency"] for idx in indices]
        distances = [distances[idx] for idx in indices]
        return utils.Neighbors(
            words=words,
            left_contexts=left_contexts,
            right_contexts=right_contexts,
            frequencies=frequencies,
            distances=distances
        )

    def retrieve(self, words: Tuple[str, str, str], n_neighbors: int) -> utils.Neighbors:
        context_vec = self.vectorizer.vectorize(*words)
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
