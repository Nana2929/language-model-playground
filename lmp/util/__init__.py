r"""Helper functions collection.

All utilities files in this module must re-import in this very file. This help
to avoid unnecessary import structure (for example, we prefer using
`lmp.util._load_model` over `lmp.util._model.load_model`).

All submodules which provide loading utilites should provide two interface, one
for directly passing parameter and one for using configuration object (for
example, `lmp.util._load_model` and `lmp.util._load_model_by_config`).

Usage:
    import lmp.util

    dataset = lmp.util.load_dataset_by_config(...)
    tokenizer = lmp.util.train_tokenizer_by_config(...)
"""
