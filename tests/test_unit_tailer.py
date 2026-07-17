# -*- coding: utf-8 -*-
"""L1 unit — iter_tail_csv streaming semantics: no loss, from_start, partial lines,
repeated headers.  Deterministic (pre-written files + small max_idle, no threads)."""
import os
import realtime as RT

HEADER = "Dst Port,Protocol,Flow Duration"


def _write(tmp_path, body):
    p = os.path.join(str(tmp_path), "live.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    return p


def _count(gen):
    return sum(len(df) for df in gen)


def test_from_start_yields_all_rows(tmp_path):
    p = _write(tmp_path, HEADER + "\n" + "\n".join(f"{80+i},6,{i}" for i in range(5)) + "\n")
    n = _count(RT.iter_tail_csv(p, poll=0.02, from_start=True, max_idle=0.1))
    assert n == 5


def test_from_end_skips_existing(tmp_path):
    p = _write(tmp_path, HEADER + "\n" + "\n".join(f"{80+i},6,{i}" for i in range(5)) + "\n")
    n = _count(RT.iter_tail_csv(p, poll=0.02, from_start=False, max_idle=0.1))
    assert n == 0                       # pre-existing rows are not re-reported


def test_repeated_header_is_skipped(tmp_path):
    body = HEADER + "\n" + "80,6,1\n" + HEADER + "\n" + "81,6,2\n"
    p = _write(tmp_path, body)
    n = _count(RT.iter_tail_csv(p, poll=0.02, from_start=True, max_idle=0.1))
    assert n == 2                       # duplicated header row not counted as data


def test_partial_line_is_buffered(tmp_path):
    # last line has no trailing newline -> writer mid-flush -> must not be yielded
    body = HEADER + "\n" + "80,6,1\n" + "81,6,2\n" + "82,6"   # partial
    p = _write(tmp_path, body)
    n = _count(RT.iter_tail_csv(p, poll=0.02, from_start=True, max_idle=0.1))
    assert n == 2


def test_columns_match_header(tmp_path):
    p = _write(tmp_path, HEADER + "\n80,6,1\n")
    chunks = list(RT.iter_tail_csv(p, poll=0.02, from_start=True, max_idle=0.1))
    assert list(chunks[0].columns) == HEADER.split(",")
    assert int(chunks[0].iloc[0]["Dst Port"]) == 80
