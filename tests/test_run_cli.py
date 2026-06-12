import sys
from pathlib import Path

from avlm_or import run


def test_baseline_cli_dispatch(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_record_originals(output_root, device, reference_labels_path):
        calls.append((output_root, device, reference_labels_path))

    monkeypatch.setattr(run, "record_originals", fake_record_originals)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "baseline",
            "--output-root",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    run.main()

    assert calls == [(str(tmp_path), "cpu", None)]


def test_text_cli_passes_template_options(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run_text_batch(*arguments):
        calls.append(arguments)

    monkeypatch.setattr(run, "run_text_batch", fake_run_text_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "text",
            "--text-method",
            "reverse_greedy",
            "--backend",
            "efficient",
            "--text-font-sizes",
            "10,12",
            "--text-angles=-45,0,45",
            "--output-root",
            str(tmp_path),
        ],
    )

    run.main()

    assert calls
    assert calls[0][-2:] == ((10, 12), (-45, 0, 45))
