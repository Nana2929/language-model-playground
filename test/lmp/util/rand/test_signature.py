"""Test :py:mod:`lmp.util.rand` signatures."""

import inspect
from inspect import Parameter, Signature

import lmp.util.rand


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.rand, 'set_seed')
  assert inspect.isfunction(lmp.util.rand.set_seed)
  assert inspect.signature(lmp.util.rand.set_seed) == Signature(
    parameters=[
      Parameter(
        name='seed',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
    ],
    return_annotation=None,
  )
