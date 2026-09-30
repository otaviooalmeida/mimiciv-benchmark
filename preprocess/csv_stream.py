"""CSV chunk reader for versioned MIMIC source schemas."""

import pandas as pd


def iter_csv_chunks(path, columns, *, chunksize, dtype=None, optional_columns=()):
    """Read selected columns in bounded chunks, filling absent optional fields.

    The header-only probe catches schema mismatches early without materializing any
    event rows. Missing optional columns are added as nulls to each chunk.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    columns = list(columns)
    optional = set(optional_columns)
    unexpected_optional = optional.difference(columns)
    if unexpected_optional:
        raise ValueError(
            "optional columns must be requested columns: {}".format(
                sorted(unexpected_optional)
            )
        )

    header = set(pd.read_csv(path, nrows=0).columns)
    missing_required = [
        column for column in columns if column not in header and column not in optional
    ]
    if missing_required:
        raise ValueError(
            "{} is missing required columns: {}".format(path, missing_required)
        )

    present_columns = [column for column in columns if column in header]
    missing_optional = [column for column in columns if column not in header]
    for chunk in pd.read_csv(
        path, usecols=present_columns, chunksize=chunksize,
        low_memory=False, dtype=dtype,
    ):
        for column in missing_optional:
            chunk[column] = pd.NA
        yield chunk.reindex(columns=columns)
