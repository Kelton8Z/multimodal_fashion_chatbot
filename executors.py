from typing import Tuple, Dict, Optional

import cv2
import json
import clip
import torch
import numpy as np

from jina import Executor, DocumentArray, requests, Document
from jina.types.arrays.memmap import DocumentArrayMemmap

class CLIPEncoder(Executor):
    """Encode image into embeddings."""

    def __init__(self, model_name: str = 'ViT-B/32', *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.set_num_threads(1)
        self.model, self.preprocess = clip.load(model_name, 'cpu')


    @requests(on='/index')
    def encode_index(self, docs: DocumentArray, **kwargs):
        if not docs:
            return
        from PIL import Image
        dam = DocumentArrayMemmap('./my-memmap')
        with torch.no_grad():
            for doc in docs:
                image = self.preprocess(Image.open(doc.uri)).unsqueeze(0).to('cpu')
                embed = self.model.encode_image(image)
                doc.embedding = embed.cpu().numpy().flatten()
                dam.extend([doc])

    @requests(on=['/search', '/eval'])
    def encode_query(self, docs: Optional[DocumentArray], groundtruths: Optional[DocumentArray], parameters: Dict, **kwargs):
        # if not docs:
        #     return
        with torch.no_grad():
            for doc in docs:
                # input_torch_tensor = clip.tokenize(doc.content)
                from Multilingual_CLIP.src import multilingual_clip
                model = multilingual_clip.load_model('M-BERT-Distil-40')
                embed = model([doc.content])
                # embed = self.model.encode_text(input_torch_tensor)
                doc.embedding = embed.cpu().numpy().flatten()

        # if parameters.get('request') == 'eval':
        #     docs = parameters['docs']
        # import requests
        # with torch.no_grad():
        #     for doc in docs:
        #         try:
        #             r = requests.get('https://api-free.deepl.com/v2/translate', params={'auth_key': 'de001190-f3e2-156d-a384-5ff98ecf5f8e:fx', 'text': doc.content, 'target_lang': 'ZH'}, timeout = 5, verify=True)
        #             translated_text = r.json()['translations'][0]['text']
        #             input_torch_tensor = clip.tokenize(translated_text)
        #             doc.embedding = self.model.encode_text(input_torch_tensor).cpu().numpy().flatten()
        #         except requests.exceptions.RequestException as e:
        #             print(e)


# class WudaoEncoder(Executor):
#     def __init__(self, model_name: str = 'ViT-B/32', *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         torch.set_num_threads(1)
#         #self.model, self.preprocess = clip.load(model_name, 'cpu')
#
#
#     @requests(on='/index')
#     def encode_index(self, docs: DocumentArray, **kwargs):
#         if not docs:
#             return
#         from PIL import Image
#         with torch.no_grad():
#             for doc in docs:
#                 image = self.preprocess(Image.open(doc.uri)).unsqueeze(0).to('cpu')
#                 embed = self.model.encode_image(image)
#                 doc.embedding = embed.cpu().numpy().flatten()
#
#
#     @requests(on=['/search', '/eval'])
#     def encode_query(self, docs: DocumentArray, **kwargs):
#         if not docs:
#             return
#         with torch.no_grad():
#             for doc in docs:
#                 r = requests.get('https://api-free.deepl.com/v2/translate',
#                                  params={'auth_key': 'de001190-f3e2-156d-a384-5ff98ecf5f8e:fx', 'text': doc.content,
#                                          'target_lang': 'ZH'})
#                 translated_text = r.json()['translations'][0]['text']
#                 input_torch_tensor = clip.tokenize(translated_text)
#                 embed = self.model.encode_text(input_torch_tensor)
#                 doc.embedding = embed.cpu().numpy().flatten()

# data = {
#     'json':'',
#     'baaiApiToken': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJDTElFTlRfSUQiOiIxMzk4MjI3NDI0MDgwMDY0NTEyIiwiZXhwIjoxNjI0NzkwMzgwfQ.K30lx1L-wC0wGLqdDnuHH0CwqkzJaXXXXXXXX',
# }
# files = {'file':open('data/stamps/angel_above_earth.jpg', 'rb')}
# r = requests.post('https://wudaoai.cn/model-api/api/v1/verifyApi', data=data,files=files)
# print(r.json())

# store the embeddings with
class MyIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._docs = DocumentArray()

    @requests(on='/index')
    def index(self, docs: 'DocumentArray', **kwargs):
        self._docs.extend(docs)

    # ????????????????????????top 4 img?????????
    @requests(on=['/search', '/eval'])
    def search(self, docs: Optional[DocumentArray], groundtruths: Optional[DocumentArray], parameters: Dict, **kwargs):
        # if parameters.get('request') == 'eval':
        #     docs = parameters['docs']
        a = np.stack(docs.get_attributes('embedding'))
        encoder = CLIPEncoder()

        encoder.encode_query(self._docs, groundtruths, parameters)
        print(f' self docs {self._docs}')
        b = np.stack(self._docs.get_attributes('embedding'))
        q_emb = _ext_A(_norm(a))
        d_emb = _ext_B(_norm(b))
        dists = _cosine(q_emb, d_emb)
        idx, dist = self._get_sorted_top_k(dists, 5)
        for _q, _ids, _dists in zip(docs, idx, dist):
            for _id, _dist in zip(_ids, _dists):
                d = Document(self._docs[int(_id)], copy=True, modality='image')
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


    # ?????????????????????????????????????????????????????????????????????
    @requests(on='/generate')
    def imgGeneration(self, template_path, pattern_path, **kwargs):
        def img_resize(image):
            height, width = image.shape[0], image.shape[1]
            # ?????????????????????????????????
            width_new = 500
            height_new = 500
            # ???????????????????????????
            if width / height >= width_new / height_new:
                img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
            else:
                img_new = cv2.resize(image, (int(width * height_new / height), height_new))
            return img_new

        template = cv2.imread(template_path)
        pattern = cv2.imread(pattern_path)
        pattern = img_resize(pattern)
        src_mask = 255 * np.ones(pattern.shape, pattern.dtype)
        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if (pattern[i][j] == pattern[0][0]).all():
                    src_mask[i][j] = [0, 0, 0]
        center = (template.shape[1] // 2, template.shape[0] // 2)
        output = cv2.seamlessClone(pattern, template, src_mask, center, cv2.NORMAL_CLONE)
        img_path = 'test.png'
        cv2.imwrite(img_path, output)
        return img_path


# class MyEncoder(Executor):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         np.random.seed(1337)
#         # generate a random orthogonal matrix
#         H = np.random.rand(784, 64)
#         u, s, vh = np.linalg.svd(H, full_matrices=False)
#         self.oth_mat = u @ vh
#
#     @requests
#     def encode(self, docs: 'DocumentArray', **kwargs):
#         # reduce dimension to 50 by random orthogonal projection
#         content = np.stack(docs.get_attributes('content'))
#         embeds = (content.reshape([-1, 784]) / 255) @ self.oth_mat
#         for doc, embed in zip(docs, embeds):
#             doc.embedding = embed
#             doc.convert_image_blob_to_uri(width=28, height=28)
#             doc.pop('blob')
#
#
def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim : 2 * dim] = A
    A_ext[:, 2 * dim :] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim : 2 * dim] = -2.0 * B.T
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
        # self.total_recall = 0

    @property
    def avg_precision(self):
        return self.total_precision / self.num_docs

    # @property
    # def avg_recall(self):
    #     return self.total_recall / self.num_docs

    def _precision(self, actual, desired):
        if self.eval_at == 0:
            return 0.0
        actual_at_k = actual[: self.eval_at] if self.eval_at else actual
        ret = len(set(actual_at_k).intersection(set(desired)))
        sub = len(actual_at_k)
        return ret / sub if sub != 0 else 0.0

    # def _recall(self, actual, desired):
    #     if self.eval_at == 0:
    #         return 0.0
    #     actual_at_k = actual[: self.eval_at] if self.eval_at else actual
    #     ret = len(set(actual_at_k).intersection(set(desired)))
    #     return ret / len(desired)

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
                self.total_precision += 1 # self._precision(actual, desired)
            # self.total_recall += self._recall(actual, desired)
            doc.evaluations['cosine'] = self.avg_precision
            # recall_score = doc.evaluations.add()
            # recall_score['cosine'] = self.avg_recall
            # doc.evaluations.append(recall_score)
        print(f' ************* {self.avg_precision} *************')