"""Top-K inference method."""

import argparse
from typing import Any, ClassVar, List

import torch

import lmp.util.validate
from lmp.infer._base import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr._base import EOS_TKID, PAD_TKID, BaseTknzr


class TopKInfer(BaseInfer):
  """Top-K inference method.

  For each inference step, this method pick the token id with the **top-K highest probability** from token id
  probability distribution, and use that token id as the next token id prediction result.  It is a non-greedy algorithm
  since the best prediction (which correspond to the highest probability) is not guaranteed to be chosen.  In exchange
  it has higher diversity on generation results compare to :py:class:`lmp.infer.Top1Infer`.

  Parameters
  ----------
  k: int
    Number of token ids to be sampled.
  max_seq_len: str
    Maximum length constraint on generated token list.  One can use larger contraint compare to training.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI name of top-k inference method is ``top-K``.
  k: int
    Number of token ids to be sampled.

  See Also
  --------
  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  lmp.infer.Top1Infer
    Top-1 inference method.
  lmp.script.gen_txt
    Use pre-trained language model checkpoint to generate continual text of given text segment.
  """

  infer_name: ClassVar[str] = 'top-K'

  def __init__(self, k: int, max_seq_len: int, **kwargs: Any):
    super().__init__(max_seq_len=max_seq_len)
    # `k` validation.
    lmp.util.validate.raise_if_not_instance(val=k, val_name='k', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, k], val_names=['1', 'k'])
    self.k = k

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add top-K inference method constructor parameters to CLI arguments parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

    Returns
    -------
    None

    See Also
    --------
    :doc:`lmp.script.gen_txt </script/gen_txt>`
      Use pre-trained language model checkpoint to generate continual text of given text segment.

    Examples
    --------
    >>> import argparse
    >>> from lmp.infer import TopKInfer
    >>> parser = argparse.ArgumentParser()
    >>> TopKInfer.infer_parser(parser)
    >>> args = parser.parse_args(['--k', '10'])
    >>> assert args.k == 10
    """
    super().add_CLI_args(parser=parser)

    # Required arguments.
    group = parser.add_argument_group('top-K inference method arguments')
    group.add_argument(
      '--k',
      help='Number of token ids to be sampled.',
      required=True,
      type=int,
    )

  @torch.no_grad()
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Top-K inference algorithm is structured as follow:

    #. Encode input text as 1 sample batch.
    #. Remove token ids after ``[eos]`` since model is not trained to predict tokens after seeing ``[eos]``.
    #. Loop over conditioned token ids to generate conditioned hidden states.
    #. Loop to generate token ids.  In each iteration, generated token id was choosed so that it is one of the top-K
       highest probabilities from next token id prediction probability distribution.  Generating loop will stop early
       if ``[eos]`` is generated, otherwise generating loop only stop when maximum length constraint enforced by
       ``self.max_seq_len`` is violated.
    #. Decode generated token ids into text and return.

    Parameters
    ----------
    model: lmp.model.BaseModel
      Pre-trained language model which will be used to generate text.
    tknzr: lmp.tknzr.BaseTknzr
      Pre-trained tokenizer which perform text encoding and decoding.
    txt: str
      Text segment which the generation process is conditioned on.

    Returns
    -------
    str
      Generated text.
    """
    # Get model running device.
    device = next(model.parameters()).device

    # Encode as 1 sample batch.  We convert token ids to tensor and move tensor to the same running device as model.
    # shape: (1, max_seq_len).
    batch_cur_tkids = torch.LongTensor(tknzr.batch_enc(batch_txt=[txt], max_seq_len=self.max_seq_len)).to(device)

    # Remove token ids after `[eos]` since model is not trained to predict tokens after seeing `[eos]`.
    mask = (batch_cur_tkids == EOS_TKID) | (batch_cur_tkids == PAD_TKID)
    seq_len = batch_cur_tkids.size(1) - mask.sum()
    batch_cur_tkids = batch_cur_tkids[:, :seq_len]

    # Loop over conditioned token ids to generate conditioned hidden states.
    batch_prev_states = None
    for i in range(seq_len - 1):
      _, batch_prev_states = model.pred(batch_cur_tkids=batch_cur_tkids[:, i], batch_prev_states=batch_prev_states)

    # Calculate how many token at most can be generated.
    out_seq_len = self.max_seq_len - seq_len + 1

    # Generate token ids.
    batch_cur_tkids = batch_cur_tkids[:, -1]
    gen_tkids: List[int] = []
    for _ in range(out_seq_len):
      # Get next token id prediction probability distribution.
      # shape: (1, vocab_size)
      batch_next_tkids_pd, batch_prev_states = model.pred(
        batch_cur_tkids=batch_cur_tkids,
        batch_prev_states=batch_prev_states,
      )

      # Get top-K highest probabilities from next token id prediction probability distribution.
      # shape: (1, k).
      batch_next_tkids_topk_p, batch_next_tkids_topk = batch_next_tkids_pd.topk(k=self.k, dim=-1)

      # Use the top-K highest probabilities to construct multinomial distribution.  Then sample token id from
      # multinomial distribution as the next token id prediction result.
      # `batch_next_tkids_topk_sample` shape: (1, 1).
      batch_next_tkids_topk_sample = torch.multinomial(batch_next_tkids_topk_p, num_samples=1)

      # Use sampled result to fetch next token id prediction.
      # shape: (1).
      batch_next_tkids = torch.gather(
        input=batch_next_tkids_topk,
        dim=1,
        index=batch_next_tkids_topk_sample,
      ).squeeze(1)
      gen_tkid = int(batch_next_tkids.item())
      gen_tkids.append(gen_tkid)

      # Update input token ids.
      batch_cur_tkids = batch_next_tkids

      # If the prediction token id is `[eos]`, then stop generation immediately.
      if gen_tkid == EOS_TKID:
        break

    # Output generated text.
    return tknzr.batch_dec(batch_tkids=[gen_tkids], rm_sp_tks=True)[0]
