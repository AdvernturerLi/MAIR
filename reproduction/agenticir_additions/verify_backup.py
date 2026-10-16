"""Verify downloaded backup and expose final images without long Windows paths."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    root = args.archive.parent
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == args.sha256, 'Downloaded archive hash mismatch'
    with tarfile.open(args.archive) as archive:
        manifest = json.loads(archive.extractfile('BACKUP_MANIFEST.json').read())
        expected = manifest['file_sha256']
        members = archive.getmembers()
        assert len(members) == len(expected) + 1
        assert len({m.name for m in members}) == len(members)
        for member in members:
            parts = PurePosixPath(member.name).parts
            assert member.isfile() and not member.issym() and not member.islnk()
            assert parts and not PurePosixPath(member.name).is_absolute()
            assert '..' not in parts and ':' not in member.name and '\\' not in member.name
            assert member.name == 'BACKUP_MANIFEST.json' or member.name in expected
        # Read in original archive order. Keep lengthy raw trace paths in the
        # verified archive; extracting them would exceed Windows MAX_PATH.
        for member in members:
            if member.name == 'BACKUP_MANIFEST.json':
                continue
            checksum = hashlib.sha256()
            expose = '/' not in member.name or member.name.startswith('restored_images/')
            target = root.joinpath(*PurePosixPath(member.name).parts)
            output = None
            if expose:
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open('wb')
            try:
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        checksum.update(block)
                        if output is not None:
                            output.write(block)
            finally:
                if output is not None:
                    output.close()
            assert checksum.hexdigest() == expected[member.name], member.name
    rows = json.loads((root / 'completed_results.json').read_text())
    images = list((root / 'restored_images').glob('*.png'))
    assert len(rows) == len(images) == manifest['completed_images']
    assert set(manifest['sample_ids']) == {r['sample_id'] for r in rows}
    for row in rows:
        assert row['status'] == 'success'
        image = root / 'restored_images' / (row['sample_id'] + '.png')
        assert cv2.imread(str(image)) is not None, str(image)
        row['server_output'] = row['output']
        row['output'] = 'restored_images/' + row['sample_id'] + '.png'
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / 'BACKUP_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    verification = {'status': 'verified', 'completed_images': len(images),
                    'verified_files_in_archive': len(expected), 'archive_sha256': args.sha256,
                    'all_final_images_readable': True,
                    'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                    'raw_logs_and_intermediates': 'Fully verified inside archive; long paths retained in archive'}
    temporary = root / 'LOCAL_VERIFICATION.json.tmp'
    temporary.write_text(json.dumps(verification, indent=2))
    temporary.replace(root / 'LOCAL_VERIFICATION.json')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
