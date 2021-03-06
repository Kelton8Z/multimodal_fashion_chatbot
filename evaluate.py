import os
from typing import Dict, Tuple, Optional

import clip
import torch
import numpy as np

from jina import Flow, DocumentArray, Document, Executor, requests
from jina.types.arrays.memmap import DocumentArrayMemmap

import os
from typing import Dict, Tuple, Optional

import clip
import torch
import numpy as np

from jina import Flow, DocumentArray, Document, Executor, requests
from jina.types.arrays.memmap import DocumentArrayMemmap

dam = DocumentArrayMemmap('./my-memmap')

class CLIPEncoder(Executor):
    """Encode image into embeddings."""

    def __init__(self, model_name: str = 'RN50x4', *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.set_num_threads(1)
        self.model, self.preprocess = clip.load(model_name, 'cpu')

    @requests(on='/index')
    def encode_index(self, docs: DocumentArray, **kwargs):
        if not docs:
            return
        from PIL import Image
        global dam
        with torch.no_grad():
            for doc in docs:
                image = self.preprocess(Image.open(doc.uri)).unsqueeze(0).to('cpu')
                embed = self.model.encode_image(image)
                doc.embedding = embed.cpu().numpy().flatten()
                dam.append(doc)

    @requests(on=['/search', '/eval'])
    def encode_query(self, docs: DocumentArray, groundtruths: DocumentArray, parameters: Dict, **kwargs):
        # if not docs:
        #     return
        with torch.no_grad():
            for doc in docs:
                # input_torch_tensor = clip.tokenize(doc.content)
                from Multilingual_CLIP.src import multilingual_clip
                model = multilingual_clip.load_model('M-BERT-Distil-40')
                torch.save(model, 'Multilingual_CLIP/src/pytorch_model.bin')
                model = torch.load('Multilingual_CLIP/src/pytorch_model.bin')
                embed = model([doc.content])
                # embed = self.model.encode_text(input_torch_tensor)
                doc.embedding = embed.cpu().numpy().flatten()


class MyIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._docs = DocumentArray()

    # @requests(on='/index')
    # def index(self, docs: 'DocumentArray', **kwargs):
    #     self._docs.extend(docs)

    @requests(on=['/search', '/eval'])
    def search(self, docs: DocumentArray, groundtruths: Optional[DocumentArray], parameters: Dict, **kwargs):
        # if parameters.get('request') == 'eval':
        #     docs = parameters['docs']
        a = np.stack(docs.get_attributes('embedding'))
        # b = np.stack(self._docs.get_attributes('embedding'))
        b = np.stack(dam.get_attributes('embedding'))
        q_emb = _ext_A(_norm(a))
        d_emb = _ext_B(_norm(b))
        dists = _cosine(q_emb, d_emb)
        idx, dist = self._get_sorted_top_k(dists, 5)
        for _q, _ids, _dists in zip(docs, idx, dist):
            for _id, _dist in zip(_ids, _dists):
                # d = Document(self._docs[int(_id)], copy=True, modality='image')
                d = Document(dam[int(_id)], copy=True, modality='image')
                d.evaluations['cosine'] = 1 - _dist
                _q.matches.append(d)
            matched_URIs = [doc.uri for doc in _q.matches]
            print(f'uri of the matches {matched_URIs}')

    @staticmethod
    def _get_sorted_top_k(
            dist: 'np.array', top_k: int
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        if top_k >= dist.shape[1]:
            idx = dist.argsort(axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx, axis=1)
        else:
            idx_ps = dist.argpartition(kth=top_k, axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx_ps, axis=1)
            idx_fs = dist.argsort(axis=1)
            idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
            dist = np.take_along_axis(dist, idx_fs, axis=1)

        return idx, dist


def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim: 2 * dim] = A
    A_ext[:, 2 * dim:] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim: 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2


class MyEvaluator(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_at = 50
        self.num_docs = 0
        self.total_precision = 0

    @property
    def avg_precision(self):
        return self.total_precision / self.num_docs

    @requests(on=['/search', '/eval'])
    def evaluate(self, docs: 'DocumentArray', groundtruths, **kwargs):
        print(f' gt type {type(groundtruths)}')
        for doc, groundtruth in zip(docs, groundtruths):
            self.num_docs += 1
            actual = [match.tags['id'] for match in doc.matches]
            actual_URIs = [match.uri for match in actual]
            desired = groundtruth  # pseudo_match
            # precision_score = doc.evaluations.add()
            if desired.uri in actual_URIs:
                self.total_precision += 1  # self._precision(actual, desired)
            doc.evaluations['cosine'] = self.avg_precision
        print(f' ************* {self.avg_precision} *************')


def print_resp(resp):
    print(resp)


class MyEvaluator(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_at = 50
        self.num_docs = 0
        self.total_precision = 0

    @property
    def avg_precision(self):
        return self.total_precision / self.num_docs

    @requests(on=['/search', '/eval'])
    def evaluate(self, docs: 'DocumentArray', groundtruths, **kwargs):
        print(f' gt type {type(groundtruths)}')
        for doc, groundtruth in zip(docs, groundtruths):
            self.num_docs += 1
            actual = [match.tags['id'] for match in doc.matches]
            actual_URIs = [match.uri for match in actual]
            desired = groundtruth  # pseudo_match
            # precision_score = doc.evaluations.add()
            if desired.uri in actual_URIs:
                self.total_precision += 1  # self._precision(actual, desired)
            doc.evaluations['cosine'] = self.avg_precision
        print(f' ************* {self.avg_precision} *************')

def print_resp(resp):
    print(resp)

f = Flow(cors=True, restful=True).add(uses=CLIPEncoder).add(uses=MyIndexer).add(uses=MyEvaluator)

image_files = list(os.walk('data'))
# [('./', ['stamps', 't_shirts'], []),
# ('./stamps', [], ['sunflower.jpg', '9.jpg', '14.jpg', 'stamp1.jpeg']),
# ('./t_shirts', [], ['shirt1.jpeg', 'shirt3.png', 'shirt2.png'])]
# '''
stamps = list(map(lambda x: image_files[1][0]+'/'+x, image_files[1][2]))[:18]
assert(len(stamps)==18)
docs = DocumentArray([Document(uri=img) for img in stamps])

with f:
    f.index(inputs=docs, request_size=12, show_progress=True)

    qa_pairs = [('????????????????????????????????????', 'white sunflower with a piece removed'), ('5????????????', '5 pokemons'),
                ('??????????????????', 'Intact sunflower'), ('????????????????????????????????????', 'a taller plant with similar but larger flowers'),
                ('?????????????????????', '4 birds on a tree branch with a setting sun'),
                ('3?????????????????????', '3 men waving their arms'), ('????????????????????????', 'a ghetto warrior holding a gun'),
                ('?????????????????????', 'hands up alien'),
                ('??????????????????????????????????????????', 'a cat astronaut standing on a planet and holding a flag'),
                ('??????????????? ?????????', 'Look at me and a cartoon face'),
                ('???????????????', 'naked baby with spiky hair'), ('?????????????????????A???', 'Rick and Morty Doraemon'),
                ('????????????A???', 'Superman Doraemon'), ('???????????????', 'flamingo couple'), ('?????????', 'mickey mouse'),
                ('???????????????', 'Jamaica Basketball'), ('?????????????????????', 'fierce dog with microphone'),
                ('????????????', 'black and white flower'), ('????????????', 'hand drawn pig'), ('??????????????????', 'panda on skateboard'),
                ('??????????????????', 'cow surrounded by flowers'), ('???????????????????????????', 'man squating and praying'),
                ('NASA?????????????????????', 'astronaut riding rocket with NASA logo'),
                ('???????????????????????????', 'boy in uniforms cheering'), ('???????????????????????????', 'ghetto warrior with arms crossed'),
                ('UFO', 'I want to believe aliens with UFO'), ('?????????', 'donald ducks'),
                ('?????????????????????', 'monkey in uniform'), ('???????????????A???', 'Flash Doraemon'),
                ('??????????????????????????????????????????', 'owl with glasses hat and tie'), ('?????????????????????', 'skating robot'),
                ('??????A???', 'Doraemon'), ('????????????', 'space launch'), ('?????????????????????', 'skating astronaut'),
                ('????????????????????????', 'Angel above earth'), ('?????????????????????', 'Big rat holding small rat'),
                ('9????????????', '9 pokemons'),
                ('??????????????????', 'Pig wearing red shoes'), ('???????????????', 'Roaring tiger tattoo'),
                ('???????????????', 'Vicious Bros Interview')]

    docs = DocumentArray([Document(content=pair[0]) for pair in qa_pairs])
    groundtruths = DocumentArray([Document(content=pair[1]) for pair in qa_pairs])

    f.post(
        '/eval',
        on_done=print_resp,
        inputs=docs,
        groundtruths=groundtruths,
        parameters={'request': 'eval', 'top_k': 5},
        request_size=12,
        show_progress=True
    )