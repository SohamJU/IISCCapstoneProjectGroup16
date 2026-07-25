import numpy as np

from src.embeddings.embedder import ProductEmbedder


class DummyModel:
    def __init__(self) -> None:
        self.started_with = None
        self.stopped = False

    def start_multi_process_pool(self, target_devices=None):
        self.started_with = target_devices
        return {"processes": [1, 2]}

    def encode_multi_process(self, texts, pool, batch_size, normalize_embeddings):
        return [[0.1, 0.2] for _ in texts]

    def stop_multi_process_pool(self, pool) -> None:
        self.stopped = True


def test_embed_texts_multiprocess_uses_two_cpu_workers(monkeypatch):
    embedder = ProductEmbedder(use_cpu=True)
    dummy_model = DummyModel()
    monkeypatch.setattr(embedder, "_model", dummy_model, raising=False)

    result = embedder.embed_texts_multiprocess(["hello", "world"])

    assert dummy_model.started_with == ["cpu", "cpu"]
    assert dummy_model.stopped is True
    assert result.shape == (2, 2)
    assert np.array_equal(result[0], np.array([0.1, 0.2], dtype=np.float32))
