"""databricks/tools/sync_landing.py: only files missing from the volume are uploaded, nothing is overwritten."""
import importlib.util
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / 'databricks' / 'tools' / 'sync_landing.py'
spec = importlib.util.spec_from_file_location('sync_landing', SCRIPT)
sync_landing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_landing)

VOLUME = '/Volumes/pulse/bronze/landing'


class NotFound(Exception):
    """Named like the SDK's error, which is how the script recognises it."""


class FakeFiles:
    """The Files API over a dict of path -> bytes."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.uploads = []

    def list_directory_contents(self, directory):
        prefix = directory.rstrip('/') + '/'
        children = {p[len(prefix):].split('/')[0] for p in self.files if p.startswith(prefix)}
        if not children:
            raise NotFound(directory)
        for child in sorted(children):
            path = prefix + child
            is_dir = not any(p == path for p in self.files)
            yield SimpleNamespace(path=path + ('/' if is_dir else ''), is_directory=is_dir, name=child)

    def create_directory(self, path):
        assert PurePosixPath(path).is_absolute()

    def upload(self, path, contents, overwrite=None):
        assert overwrite is False
        assert path not in self.files, 'must never replace a landed file'
        self.files[path] = contents.read()
        self.uploads.append(path)


def write(root, relative, data=b'x'):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_uploads_only_missing_files_oldest_date_first(tmp_path):
    for relative in ('fuelhh/ingest_date=2026-11-07/part-b.parquet', 'fuelhh/ingest_date=2026-11-06/part-a.parquet',
                     'pv_live/ingest_date=2026-11-06/part-c.parquet', 'pv_live/ingest_date=2026-11-06/notes.txt'):
        write(tmp_path, relative)
    files = FakeFiles({f'{VOLUME}/fuelhh/ingest_date=2026-11-06/part-a.parquet': b'x'})
    client = SimpleNamespace(files=files)

    uploaded = sync_landing.sync(client, tmp_path, VOLUME)

    # The file already there and the non-Parquet file are skipped
    assert uploaded == ['pv_live/ingest_date=2026-11-06/part-c.parquet', 'fuelhh/ingest_date=2026-11-07/part-b.parquet']
    assert files.uploads == [f'{VOLUME}/{p}' for p in uploaded]
    # A second run finds nothing to do
    assert sync_landing.sync(client, tmp_path, VOLUME) == []


def test_empty_volume_gets_everything(tmp_path):
    write(tmp_path, 'fuelhh/ingest_date=2026-11-06/part-a.parquet', b'parquet bytes')
    files = FakeFiles()
    sync_landing.sync(SimpleNamespace(files=files), tmp_path, VOLUME)
    assert files.files == {f'{VOLUME}/fuelhh/ingest_date=2026-11-06/part-a.parquet': b'parquet bytes'}


def test_dry_run_uploads_nothing(tmp_path, capsys):
    write(tmp_path, 'fuelhh/ingest_date=2026-11-06/part-a.parquet')
    files = FakeFiles()
    assert sync_landing.sync(SimpleNamespace(files=files), tmp_path, VOLUME, dry_run=True) == [
        'fuelhh/ingest_date=2026-11-06/part-a.parquet']
    assert files.uploads == [] and 'would upload' in capsys.readouterr().out


def test_other_errors_are_not_hidden(tmp_path):
    class Broken:
        def list_directory_contents(self, directory):
            raise PermissionError('no access to the volume')

    with pytest.raises(PermissionError):
        sync_landing.volume_files(SimpleNamespace(files=Broken()), VOLUME)
