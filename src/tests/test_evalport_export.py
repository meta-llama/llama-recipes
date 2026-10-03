# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from openeval.validate import validate_suite

EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2]
    / "end-to-end-use-cases/benchmarks/evals_synthetic_data"
)
SCRIPT = EXAMPLE_DIR / "to_evalport.py"
spec = importlib.util.spec_from_file_location("to_evalport", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_export_checked_in_dataset():
    suite = exporter.build_suite(EXAMPLE_DIR / "generated_data", "test-judge")
    result = validate_suite(suite)
    assert result.valid, result.errors
    paths = sorted(
        (EXAMPLE_DIR / "generated_data").glob("data_*.json"),
        key=exporter._case_sort_key,
    )
    assert len(suite["test_cases"]) == len(paths) == 12
    assert [case["id"] for case in suite["test_cases"]] == [
        f"hallucination_eval_{index}" for index in range(12)
    ]
    for case, path in zip(suite["test_cases"], paths):
        original = json.loads(path.read_text(encoding="utf-8"))
        assert case["input"] == original["context"]
        assert case["expected_output"] == original["report"]
        assert case["graders"] == [suite["graders"][0]["id"]]
    assert suite["graders"][0]["params"]["model"] == "test-judge"
    assert suite["graders"][0]["params"]["schema"] == exporter.JUDGE_OUTPUT_SCHEMA


def test_case_ids_remain_stable_when_files_are_added(tmp_path):
    pair = {"context": "Résumé: علی\n", "report": "Report\n"}
    (tmp_path / "data_2.json").write_text(json.dumps(pair), encoding="utf-8")
    original = exporter.build_suite(tmp_path, "test-judge")["test_cases"][0]
    (tmp_path / "data_10.json").write_text(json.dumps(pair), encoding="utf-8")
    cases = exporter.build_suite(tmp_path, "test-judge")["test_cases"]
    assert original in cases
    assert original["id"] == "hallucination_eval_2"
    assert original["input"] == pair["context"]


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        "{}",
        '{"context": 1, "report": "r"}',
        '{"context": "c", "report": null}',
        '{"context": " ", "report": "r"}',
    ],
)
def test_invalid_input_does_not_replace_existing_output(tmp_path, contents):
    (tmp_path / "data_0.json").write_text(contents, encoding="utf-8")
    output = tmp_path / "suite.json"
    output.write_text("existing output", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--judge-model",
            "test-judge",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "data_0.json" in result.stderr
    assert output.read_text(encoding="utf-8") == "existing output"


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_empty_dataset_is_rejected(tmp_path, missing):
    with pytest.raises(ValueError, match="directory does not exist|No data_"):
        exporter.build_suite(
            tmp_path / "missing" if missing else tmp_path, "test-judge"
        )


def test_empty_judge_model_is_rejected():
    with pytest.raises(ValueError, match="model identifier"):
        exporter.build_suite(EXAMPLE_DIR / "generated_data", " ")


def test_cli_defaults_work_outside_example_directory(tmp_path):
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--judge-model", "test-judge"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Exported 12 test cases" in result.stdout
        output = (tmp_path / "evals.evalport.json").read_bytes()
        assert validate_suite(json.loads(output)).valid
        if _ == 0:
            first_output = output
        else:
            assert output == first_output


def test_output_cannot_replace_input(tmp_path):
    source = tmp_path / "data_0.json"
    contents = '{"context": "c", "report": "r"}'
    source.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(tmp_path),
            "--output",
            str(source),
            "--judge-model",
            "test-judge",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "overwrite an input" in result.stderr
    assert source.read_text(encoding="utf-8") == contents
