import json
import os
import pickle
from typing import List, Optional, Tuple, Any, Dict, Callable

import fasttext
import lz4.frame
import marisa_trie
import nmslib
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from spacy.tokens import Doc

from gnn_lib.data import utils, tokenization
from gnn_lib.modules import encoders, embedding, utils as mod_utils
from gnn_lib.utils import common, io


class Vectorizer:
    def __init__(self, path: str, context_length: int) -> None:
        self.path = path
        self.context_length = context_length
        # TODO: also allow context length > 0
        assert context_length == 0, "only context length 0 supported for now"

    def vectorize(self, word: str, left_context: str, right_context: str) -> np.array:
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
    def __init__(self) -> None:
        super().__init__("", 0)
        self.ft = fasttext.load_model("cc.en.300.bin")

        total_length = 2 * self.context_length + 1
        self.pe = np.zeros((total_length, self.dim), dtype=float)
        position = np.arange(0, total_length, dtype=float)[:, None]
        div_term = np.exp(np.arange(0, self.dim, 2, dtype=float) *
                          -(np.log(10000.0) / self.dim))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

        self.mode = "uniform"
        self.use_pe = False

    def vectorize(self, word: str, left_context: str, right_context: str) -> np.array:
        left_context_vectors = []
        for i, word in enumerate(utils.tokenize_words_regex(left_context)[0]):
            vector = self.ft.get_word_vector(word)
            if self.use_pe:
                vector += self.pe[i]
            left_context_vectors.append(vector)

        word_vector = self.ft.get_word_vector(word)
        if self.use_pe:
            word_vector += self.pe[len(left_context_vectors)]

        right_context_vectors = []
        for i, word in enumerate(utils.tokenize_words_regex(right_context)[0]):
            vector = self.ft.get_word_vector(word)
            if self.use_pe:
                vector += self.pe[len(left_context_vectors) + 1 + i]
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
            weight = 5
            vectors[len(left_context_vectors)] *= weight
            return vectors.sum(axis=0) / max((len(vectors) - 1 + weight), 1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    @property
    def dim(self) -> int:
        return self.ft.get_dimension()


class BertVectorizer(Vectorizer):
    def __init__(self) -> None:
        super().__init__("", 0)
        from transformers import BertModel, BertTokenizer, BertConfig
        self.tok = BertTokenizer.from_pretrained("bert-base-cased")
        self.device = mod_utils.cuda_or_cpu(force_cpu=True)
        self.bert: BertModel = BertModel.from_pretrained("bert-base-cased")
        self.bert.to(self.device)
        self.bert.eval()
        self.config: BertConfig = BertConfig.from_pretrained("bert-base-cased")

    def vectorize(self, word: str, left_context: str, right_context: str) -> np.array:
        tensors = self.tok(left_context + word + right_context, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.bert(**tensors).last_hidden_state.cpu().numpy()
        return np.mean(outputs, axis=1)

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        input_strs = [l + w + r for w, l, r in inputs]
        tensors = self.tok(input_strs, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.bert(**tensors).last_hidden_state.cpu().numpy()
        vectors = []
        for i, att_mask in enumerate(tensors["attention_mask"]):
            vectors.append(outputs[i, :att_mask.sum()].mean(0))
        return np.stack(vectors)

    def to_device(self, device: torch.device) -> None:
        self.device = device
        self.bert.to(self.device)

    @property
    def dim(self) -> int:
        return self.config.hidden_size


class StringVectorizer(Vectorizer):
    def __init__(self) -> None:
        super().__init__("", 0)

    def vectorize(self, word: str, left_context: str, right_context: str) -> np.array:
        return left_context + word + right_context

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        return [l + w + r for w, l, r in inputs]

    @property
    def dim(self) -> int:
        return 128


class CustomNeuralVectorizerModel(nn.Module):
    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class CustomNeuralVectorizer(Vectorizer):
    def __init__(self, path: str) -> None:
        super().__init__(path, 0)
        self.device = mod_utils.cuda_or_cpu(force_cpu=True)
        self.model: CustomNeuralVectorizerModel = torch.load(path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def vectorize(self, word: str, left_context: str, right_context: str) -> np.array:
        return self.model.vectorize_batch([(word, left_context, right_context)])[0]

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        return self.model.vectorize_batch(inputs)

    @property
    def dim(self) -> int:
        return self.model.dim


class CharTransformer(CustomNeuralVectorizerModel):
    def __init__(self) -> None:
        super().__init__()
        self.max_length = 64
        self.hidden_dim = 512
        self.tok = tokenization.CharTokenizer()
        self.emb = embedding.TokenEmbedding(
            self.hidden_dim, self.tok.vocab_size, self.tok.token_to_id(tokenization.PAD))
        self.pos_emb = embedding.SinusoidalPositionalEmbedding(self.hidden_dim)
        self.norm_emb = torch.nn.LayerNorm(self.hidden_dim)
        self.encoder = encoders.Transformer(in_dim=self.hidden_dim,
                                            hidden_dim=self.hidden_dim,
                                            num_layers=6,
                                            dropout=0.1)

    def forward(self, inputs: List[str]) -> torch.Tensor:
        device = mod_utils.device_from_model(self)
        token_ids = []
        positions = []
        lengths = torch.empty(len(inputs), dtype=torch.long, device=device)
        for i, ipt in enumerate(inputs):
            tokens = self.tok.tokenize(ipt)
            token_ids.append(torch.tensor(tokens, dtype=torch.long, device=device))
            positions.append(torch.arange(len(tokens), dtype=torch.long, device=device))
            lengths[i] = len(tokens)

        token_ids_tensor = torch.cat(token_ids)
        positions_tensor = torch.cat(positions)

        emb = self.norm_emb(self.emb(token_ids_tensor) + self.pos_emb(positions_tensor))
        enc = self.encoder(emb, lengths)
        enc_split = mod_utils.split(enc, lengths)
        outputs = []
        for enc in enc_split:
            outputs.append(torch.mean(enc, dim=0))
        outputs = torch.stack(outputs)
        return outputs

    def vectorize_batch(self, inputs: List[Tuple[str, str, str]]) -> np.array:
        with torch.no_grad():
            return self.forward([l + w + r for w, l, r in inputs]).cpu().numpy()

    @property
    def dim(self) -> int:
        return self.hidden_dim


def get_vectorizer(name: str, path: Optional[str] = None) -> Vectorizer:
    if name == "ft":
        return FastTextVectorizer()
    elif name == "bert":
        return BertVectorizer()
    elif name == "string":
        return StringVectorizer()
    elif name == "custom":
        assert path is not None, "need a path for the custom neural vectorizer"
        return CustomNeuralVectorizer(path)
    else:
        raise ValueError(f"Unknown vectorizer {name}")


class NNIndex:
    @staticmethod
    def get_index(space: str):
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

    def __init__(self,
                 directory: str,
                 vectorizer_path: Optional[str] = None,
                 num_threads: int = 8) -> None:
        self.directory = directory
        self.vectorizer_path = vectorizer_path

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

        self.vectorizer = get_vectorizer(self.params["vectorizer"], self.vectorizer_path)

        self.index = self.get_index(self.params["space"])
        self.index.loadIndex(self.index_file, load_data=self.params["space"] not in {"l2", "cosinesimil"})
        self.index.setQueryTimeParams({"efSearch": 100})

    @staticmethod
    def create(
            files: List[str],
            out_directory: str,
            context_length: int,
            vectorizer_name: str,
            dist: str,
            vectorizer_path: Optional[str] = None,
            dictionary_file: Optional[str] = None
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

        vectorizer = get_vectorizer(vectorizer_name, vectorizer_path)
        dictionary = io.dictionary_from_file(dictionary_file) if dictionary_file is not None else None

        index = NNIndex.get_index(space)
        lmdb = utils.open_lmdb(lmdb_out_file, write=True)
        txn = lmdb.begin(write=True)
        txn.drop(lmdb.open_db())

        seen = set()

        idx = 0
        total = 0
        invalid = 0
        for file in tqdm(files, disable=common.disable_tqdm()):
            with open(file, "r", encoding="utf8") as inf:
                for line in tqdm(inf, total=io.line_count(file), leave=False, disable=common.disable_tqdm()):
                    sequence = json.loads(line)["sequence"]
                    batch = NNIndex.prepare_sequence(sequence, context_length)
                    batch_filtered = []
                    for word, left_context, right_context in batch:
                        # do not add the same things multiple times to the index
                        h = hash(left_context + word + right_context)
                        if h in seen:
                            continue
                        total += 1
                        if dictionary is not None and word not in dictionary:
                            invalid += 1
                            continue
                        seen.add(h)
                        batch_filtered.append((word, left_context, right_context))
                    if len(batch_filtered) == 0:
                        continue

                    context_vec = vectorizer.vectorize_batch(batch_filtered)
                    indices = [idx + i for i in range(len(context_vec))]

                    index.addDataPointBatch(context_vec, ids=indices)

                    for i in range(len(context_vec)):
                        word, left_context, right_context = batch_filtered[i]
                        data = pickle.dumps({
                            "word": word,
                            "left_context": left_context,
                            "right_context": right_context
                        })
                        data = lz4.frame.compress(data, compression_level=16)
                        assert txn.put(f"{indices[i]}".encode("utf8"), data)

                    idx += len(indices)

        if dictionary is not None:
            logger.info(f"{100 * invalid / total:.2f}% of all unique (left_context, word, right_context) items "
                        f"were invalid and thus not added to index")

        logger.info("Creating index")
        index_time_params = {
            "M": 32,
            "indexThreadQty": int(os.getenv("GNN_LIB_INDEX_NUM_THREADS", os.cpu_count())),
            "efConstruction": 1000,
            "post": 2
        }
        index.createIndex(index_time_params, print_progress=True)
        index.saveIndex(index_out_file, save_data=space not in {"l2", "cosinesimil"})

        assert txn.put("params".encode("utf8"),
                       pickle.dumps({"context_length": context_length,
                                     "space": space,
                                     "vectorizer": vectorizer_name}))
        assert txn.put("num_sequences".encode("utf8"), pickle.dumps(idx))
        txn.commit()
        lmdb.close()

        logger.info(f"Saved {idx} items in the index")

    @staticmethod
    def prepare_sequence(sequence: str, context_length: int) -> List[Tuple[str, str, str]]:
        sequence = utils.clean_sequence(sequence)
        words, whitespaces = utils.tokenize_words_regex(sequence)
        return NNIndex.prepare_words(words, whitespaces, context_length)

    @staticmethod
    def prepare_words(words: List[str], whitespaces: List[bool], context_length: int) -> List[Tuple[str, str, str]]:
        # pad words and whitespaces
        words = [""] * context_length + words + [""] * context_length
        whitespaces = [False] * context_length + whitespaces + [False] * context_length
        outputs = []
        for i in range(context_length, len(words) - context_length):
            word = words[i]
            left_context = utils.de_tokenize_words(
                words[i - context_length:i],
                whitespaces[i - context_length: i])
            right_context = utils.de_tokenize_words(
                words[i + 1: i + context_length + 1],
                whitespaces[i + 1: i + context_length + 1])
            outputs.append((word, left_context, right_context))
        return outputs

    def batch_retrieve(self, words_batch: List[Tuple[str, str, str]], n_neighbors: int) -> List[utils.NEIGHBORS]:
        vectors_batch = self.vectorizer.vectorize_batch(words_batch)

        neighbors_batch = self.index.knnQueryBatch(
            vectors_batch, k=n_neighbors, num_threads=self.num_threads)

        return [self._to_neighbors(neighbors, distances) for neighbors, distances in neighbors_batch]

    def batch_retrieve_from_docs(self, doc_batch: List[Doc], n_neighbors: int) -> List[List[utils.NEIGHBORS]]:
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

    def _to_neighbors(self, neighbors: List[int], distances: List[float]) -> utils.NEIGHBORS:
        words = []
        left_contexts = []
        right_contexts = []
        for neighbor in neighbors:
            data = self._get_item(neighbor)
            words.append(data["word"])
            left_contexts.append(data["left_context"])
            right_contexts.append(data["right_context"])
        return utils.NEIGHBORS(
            words=words,
            left_contexts=left_contexts,
            right_contexts=right_contexts,
            distances=distances
        )

    def retrieve(self, words: Tuple[str, str, str], n_neighbors: int) -> utils.NEIGHBORS:
        context_vec = self.vectorizer.vectorize(*words)
        neighbors, distances = self.index.knnQuery(context_vec, k=n_neighbors)
        return self._to_neighbors(neighbors, distances)


def get_neighbor_fn(index: NNIndex, num_neighbors: int) -> Callable[[List[Doc]], List[List[utils.NEIGHBORS]]]:
    def neigh(docs: List[Doc]) -> List[List[utils.NEIGHBORS]]:
        return index.batch_retrieve_from_docs(docs, num_neighbors)

    return neigh


class PrefixIndex:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "rb") as inf:
            self.trie, self.dict = pickle.load(inf)

        self.logger = common.get_logger("PREFIX_INDEX")

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
