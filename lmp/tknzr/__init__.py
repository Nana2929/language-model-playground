"""Tokenizer module.

Attributes
----------
ALL_TKNZRS: list[lmp.tknzr.BaseTknzr]
  All available tokenizers.
TKNZR_OPTS: typing.Final[dict[str, lmp.tknzr.BaseTknzr]]
  Mapping tokenizer name ``tknzr_name`` to tokenizer class.

Examples
--------
Get :py:class:`lmp.tknzr.CharTknzr` by its name.

>>> from lmp.tknzr import CharTknzr, TKNZR_OPTS
>>> CharTknzr.tknzr_name in TKNZR_OPTS
True
>>> TKNZR_OPTS[CharTknzr.tknzr_name] == CharTknzr
True
"""

from typing import Dict, Final, List, Type

from lmp.tknzr._base import BaseTknzr
from lmp.tknzr._char import CharTknzr
from lmp.tknzr._ws import WsTknzr

ALL_TKNZRS: Final[List[Type[BaseTknzr]]] = [
  CharTknzr,
  WsTknzr,
]
TKNZR_OPTS: Final[Dict[str, Type[BaseTknzr]]] = {t.tknzr_name: t for t in ALL_TKNZRS}
