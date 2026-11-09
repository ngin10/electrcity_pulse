"""Copy new landing files into the Databricks volume bronze reads on Free Edition.

    python databricks/tools/sync_landing.py              # from data/landing (the zero-cost local landing zone)
    python databricks/tools/sync_landing.py --source gs://databricks_1/electricity_gen_by_fuel/landing
    python databricks/tools/sync_landing.py --dry-run    # only list what would be copied

Free Edition runs on Databricks' own infrastructure, so it can't read your machine or your bucket. Run this
from the repo root, after ingestion and before the Databricks job:
1. For a gs:// source only: `gcloud storage rsync` mirrors it to a local folder, downloading only files not
   there yet. It uses your own gcloud login (the consumer's service account can only create, not read).
2. Every local file not yet in the volume is uploaded, with the same path.

Landing files are append-only and uniquely named, so a file already in the volume is never sent again and
nothing is overwritten. Auto Loader then loads each new file exactly once.

Needs: the gcloud CLI, `pip install -r databricks/tools/requirements.txt`, and `databricks auth login`.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

# The ingestion code's default LANDING_PATH: landing locally costs nothing
DEFAULT_SOURCE = 'data/landing'
DEFAULT_MIRROR = 'data/landing_mirror'          # inside data/, which git ignores
DEFAULT_VOLUME = '/Volumes/pulse/bronze/landing'


def mirror_gcs(source, mirror):
    """Bring the local mirror up to date with the bucket; only missing or changed files are downloaded."""
    gcloud = shutil.which('gcloud')
    if gcloud is None:
        sys.exit('gcloud not found: install the Google Cloud CLI, or pass --source with a local folder')
    Path(mirror).mkdir(parents=True, exist_ok=True)
    subprocess.run([gcloud, 'storage', 'rsync', '--recursive', source, str(mirror)], check=True)


def local_files(root):
    """Paths of the .parquet files under root, relative to it, with forward slashes: e.g.
    fuelhh/ingest_date=2026-11-06/part-ab12.parquet."""
    root = Path(root)
    return {path.relative_to(root).as_posix() for path in root.rglob('*.parquet')}


def volume_files(client, volume):
    """Paths of every file already in the volume, relative to it. An empty or missing volume gives none."""
    found, pending = set(), [volume]
    while pending:
        directory = pending.pop()
        try:
            entries = list(client.files.list_directory_contents(directory))
        except Exception as error:
            # databricks.sdk.errors.NotFound, matched by name so this module imports without the SDK
            if type(error).__name__ != 'NotFound':
                raise
            continue
        for entry in entries:
            if entry.is_directory:
                pending.append(entry.path.rstrip('/'))
            else:
                found.add(str(PurePosixPath(entry.path).relative_to(volume)))
    return found


def files_to_upload(local, remote):
    """The local files the volume doesn't have yet, oldest landing date first (dates sort as text)."""
    return sorted(local - remote, key=lambda path: (path.split('/')[1:2], path))


def sync(client, mirror, volume, dry_run=False):
    """Upload every mirrored file missing from the volume; returns the paths uploaded (or that would be)."""
    missing = files_to_upload(local_files(mirror), volume_files(client, volume))
    for count, relative in enumerate(missing, 1):
        target = f'{volume}/{relative}'
        if dry_run:
            print(f'would upload {target}')
            continue
        client.files.create_directory(str(PurePosixPath(target).parent))
        with open(Path(mirror) / relative, 'rb') as contents:
            # overwrite=False: if the file somehow exists already, fail rather than replace landed data
            client.files.upload(target, contents, overwrite=False)
        if count % 100 == 0 or count == len(missing):
            print(f'uploaded {count}/{len(missing)}')
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', default=DEFAULT_SOURCE,
                        help='the landing zone: a local folder, or a gs:// path (mirrored with gcloud first)')
    parser.add_argument('--mirror', default=DEFAULT_MIRROR, help='local copy of a gs:// source')
    parser.add_argument('--volume', default=DEFAULT_VOLUME, help='the volume bronze reads')
    parser.add_argument('--profile', help='Databricks CLI profile (default: the default profile)')
    parser.add_argument('--dry-run', action='store_true', help='list what would be uploaded, upload nothing')
    args = parser.parse_args(argv)

    if args.source.startswith('gs://'):
        mirror_gcs(args.source, args.mirror)
        local_root = args.mirror
    else:
        local_root = args.source
        if not Path(local_root).is_dir():
            sys.exit(f'{local_root} not found: run from the repo root, or pass --source')

    from databricks.sdk import WorkspaceClient
    client = WorkspaceClient(profile=args.profile)
    uploaded = sync(client, local_root, args.volume.rstrip('/'), dry_run=args.dry_run)
    print(f'{len(uploaded)} new file(s) {"to upload" if args.dry_run else "uploaded"} to {args.volume}')


if __name__ == '__main__':
    main()
