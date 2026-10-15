"""Resolve an immutable CI source commit and validate an explicitly selected release.

Uses only the standard library and Git, without importing or executing the
selected release. Manual runs can therefore build older tags using current
release tooling, and validation-only runs never authorize publication.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from pathlib import Path


def git(repository, *arguments):
    return subprocess.run(
        ['git', *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def prepare_release(repository, event_name, event_ref, event_sha, release_tag='', publish=False):
    """Return the source commit, release tag, and publication authorization."""
    if event_name not in {'push', 'pull_request', 'workflow_dispatch'}:
        raise ValueError(f'Unsupported release event: {event_name}')
    manual = event_name == 'workflow_dispatch'
    if not manual and (release_tag or publish):
        raise ValueError('Release inputs are only supported for workflow_dispatch')
    if manual and publish and not release_tag:
        raise ValueError('Publishing manually requires an explicit release_tag')

    commit = git(repository, 'rev-parse', '--verify', '--end-of-options', f'{event_sha}^{{commit}}')
    tag = release_tag if manual else ''
    if event_name == 'push' and event_ref.startswith('refs/tags/'):
        tag = event_ref.removeprefix('refs/tags/')

    if tag:
        if not tag.startswith('v'):
            raise ValueError('Release tags must start with v (for example, v0.2.0)')
        git(repository, 'check-ref-format', f'refs/tags/{tag}')
        if manual:
            # Fully qualify the tag: a branch with the same name must never be
            # released accidentally. Fetching FETCH_HEAD does not move the tag.
            git(repository, 'fetch', '--no-tags', '--depth=1', 'origin', f'refs/tags/{tag}')
            commit = git(repository, 'rev-parse', '--verify', 'FETCH_HEAD^{commit}')

        source = git(repository, 'show', f'{commit}:batbot/_version.py')
        version = None
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__version__'
                for target in node.targets
            ):
                version = ast.literal_eval(node.value)
        if not isinstance(version, str) or tag != f'v{version}':
            raise ValueError(f'Release tag {tag!r} does not match package version {version!r}')

    return {
        'commit': commit,
        'release_tag': tag,
        'publish': str(bool(tag) and (not manual or publish)).lower(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        result = prepare_release(
            args.repository,
            os.environ['GITHUB_EVENT_NAME'],
            os.environ['GITHUB_REF'],
            os.environ['GITHUB_SHA'],
            release_tag=os.environ.get('RELEASE_TAG', ''),
            publish=os.environ.get('PUBLISH', 'false') == 'true',
        )
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f'Release preparation failed: {exc}\n{exc.stderr or ""}')
    except (KeyError, ValueError, SyntaxError) as exc:
        parser.exit(1, f'Release preparation failed: {exc}\n')
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf8') as stream:
        for name, value in result.items():
            stream.write(f'{name}={value}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
