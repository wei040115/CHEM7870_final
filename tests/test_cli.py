from pathlib import Path

from diffusivity.__main__ import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_no_json_skips_json_export(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    main([
        '--no-plots',
        '--no-uncertainty',
        '--no-json',
        '--out-dir', str(tmp_path),
        '--prefix', 'skip_json',
    ])

    out = tmp_path / 'skip_json'
    assert not (out / 'skip_json_results.json').exists()
    assert (out / 'skip_json_results.csv').exists()


def test_no_csv_skips_csv_export(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    main([
        '--no-plots',
        '--no-uncertainty',
        '--no-csv',
        '--out-dir', str(tmp_path),
        '--prefix', 'skip_csv',
    ])

    out = tmp_path / 'skip_csv'
    assert (out / 'skip_csv_results.json').exists()
    assert not (out / 'skip_csv_results.csv').exists()
