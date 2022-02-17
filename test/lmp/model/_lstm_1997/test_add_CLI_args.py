"""Test parser arguments.

Test target:
- :py:meth:`lmp.model._lstm_1997.LSTM1997.add_CLI_args`.
"""

import argparse

from lmp.dset import ALL_DSETS
from lmp.model._elman_net import LSTM1997


def test_arguments(
  batch_size: int,
  beta1: float,
  beta2: float,
  ckpt_step: int,
  d_blk: int,
  d_emb: int,
  eps: float,
  exp_name: str,
  log_step: int,
  lr: float,
  max_norm: float,
  max_seq_len: int,
  n_blk: int,
  n_epoch: int,
  seed: int,
  tknzr_exp_name: str,
  wd: float,
) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  LSTM1997.add_CLI_args(parser=parser)
  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      args = parser.parse_args(
        [
          '--batch_size',
          str(batch_size),
          '--beta1',
          str(beta1),
          '--beta2',
          str(beta2),
          '--ckpt_step',
          str(ckpt_step),
          '--d_blk',
          str(d_blk),
          '--d_emb',
          str(d_emb),
          '--dset_name',
          dset_type.dset_name,
          '--eps',
          str(eps),
          '--exp_name',
          exp_name,
          '--log_step',
          str(log_step),
          '--lr',
          str(lr),
          '--max_norm',
          str(max_norm),
          '--max_seq_len',
          str(max_seq_len),
          '--n_blk',
          str(n_blk),
          '--n_epoch',
          str(n_epoch),
          '--seed',
          str(seed),
          '--tknzr_exp_name',
          tknzr_exp_name,
          '--ver',
          ver,
          '--wd',
          str(wd),
        ]
      )
      assert args.batch_size == batch_size
      assert args.beta1 == beta1
      assert args.beta2 == beta2
      assert args.ckpt_step == ckpt_step
      assert args.d_blk == d_blk
      assert args.d_emb == d_emb
      assert args.dset_name == dset_type.dset_name
      assert args.eps == eps
      assert args.exp_name == exp_name
      assert args.log_step == log_step
      assert args.lr == lr
      assert args.max_norm == max_norm
      assert args.max_seq_len == max_seq_len
      assert args.n_blk == n_blk
      assert args.n_epoch == n_epoch
      assert args.seed == seed
      assert args.tknzr_exp_name == tknzr_exp_name
      assert args.ver == ver
      assert args.wd == wd
