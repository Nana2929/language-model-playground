"""Setup fixtures for testing :py:mod:`lmp.util.tknzr`."""

import os

import pytest

import lmp


@pytest.fixture(params=[False, True])
def is_uncased(request) -> bool:
  """Respect cases if set to ``False``."""
  return request.param


@pytest.fixture(params=[1, 128])
def max_seq_len(request) -> int:
  """Maximum length constraint."""
  return request.param


@pytest.fixture(params=[-1, 10000])
def max_vocab(request) -> int:
  """Maximum vocabulary size."""
  return request.param


@pytest.fixture(params=[0, 10])
def min_count(request) -> int:
  """Minimum token occurrence counts."""
  return request.param


@pytest.fixture
def tknzr_file_path(exp_name: str, request) -> str:
  """Clean up tokenizer file."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, 'tknzr.pkl')

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_file_path
