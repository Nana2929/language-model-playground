"""Setup fixtures for testing :py:class:`lmp.model._lstm_2000.LSTM2000`."""

import pytest
import torch

from lmp.model._lstm_2000 import LSTM2000
from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def lstm_2000(d_blk: int, d_emb: int, n_blk: int, tknzr: BaseTknzr) -> LSTM2000:
  """:py:class:`lmp.model.LSTM2000` instance."""
  return LSTM2000(d_blk=d_blk, d_emb=d_emb, n_blk=n_blk, tknzr=tknzr)


@pytest.fixture
def batch_tkids(lstm_2000: LSTM2000) -> torch.Tensor:
  """Batch of token ids."""
  # Shape: (2, 4).
  return torch.randint(low=0, high=lstm_2000.emb.num_embeddings, size=(2, 4))


@pytest.fixture
def batch_cur_tkids(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Batch of input token ids."""
  # Shape: (2, 3).
  return batch_tkids[..., :-1]


@pytest.fixture
def batch_next_tkids(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Batch of target token ids."""
  # Shape: (2, 3).
  return batch_tkids[..., 1:]
