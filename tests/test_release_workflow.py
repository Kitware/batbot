"""Exercise release selection with real local Git remotes, without publishing."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_release import git, prepare_release

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_release.py'


@pytest.fixture
def repository(tmp_path, monkeypatch):
    # Local fixtures must not inherit contributors' signing keys or Git hooks.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    origin = tmp_path / 'origin.git'
    origin.mkdir()
    git(origin, 'init', '--bare')
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '--initial-branch=main')
    git(source, 'config', 'user.name', 'Release Test')
    git(source, 'config', 'user.email', 'release@example.invalid')
    package = source / 'batbot'
    package.mkdir()
    version = package / '_version.py'
    version.write_text("__version__ = '0.2.0'\n", encoding='utf8')
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'Old release without current release tooling')
    old = git(source, 'rev-parse', 'HEAD')
    git(source, 'tag', '--annotate', 'v0.2.0', '--message', 'Annotated release')
    version.write_text("__version__ = '0.3.0'\n", encoding='utf8')
    git(source, 'commit', '-am', 'New main branch version')
    current = git(source, 'rev-parse', 'HEAD')
    git(source, 'tag', 'v0.3.0')
    # A branch named like the old tag deliberately points at the NEW version.
    git(source, 'branch', 'v0.2.0')
    git(source, 'remote', 'add', 'origin', str(origin))
    git(source, 'push', 'origin', 'refs/heads/main', 'refs/heads/v0.2.0', '--tags')
    checkout = tmp_path / 'checkout 空間'
    git(tmp_path, 'clone', '--no-tags', '--branch=main', str(origin), str(checkout))
    return SimpleNamespace(root=checkout, origin=origin, source=source, old=old, current=current)


@pytest.mark.parametrize(
    'event,ref',
    [
        ('push', 'refs/heads/main'),
        ('pull_request', 'refs/pull/44/merge'),
        ('workflow_dispatch', 'refs/heads/main'),
    ],
)
def test_branch_and_pr_validation_never_authorize_publication(repository, event, ref):
    result = prepare_release(repository.root, event, ref, repository.current)
    assert result == {'commit': repository.current, 'release_tag': '', 'publish': 'false'}


def test_tag_push_authorizes_only_the_matching_version(repository):
    result = prepare_release(repository.root, 'push', 'refs/tags/v0.2.0', repository.old)
    assert result == {'commit': repository.old, 'release_tag': 'v0.2.0', 'publish': 'true'}
    with pytest.raises(ValueError, match='does not match'):
        prepare_release(repository.root, 'push', 'refs/tags/v0.2.0', repository.current)


@pytest.mark.parametrize('tag,sha_name', [('v0.2.0', 'old'), ('v0.3.0', 'current')])
@pytest.mark.parametrize('publish', [False, True])
def test_dispatch_fetches_exact_annotated_or_lightweight_tag(repository, tag, sha_name, publish):
    assert not (repository.root / 'scripts/prepare_release.py').exists()
    result = prepare_release(
        repository.root,
        'workflow_dispatch',
        'refs/heads/main',
        repository.current,
        release_tag=tag,
        publish=publish,
    )
    assert result == {
        'commit': getattr(repository, sha_name),
        'release_tag': tag,
        'publish': str(publish).lower(),
    }
    assert git(repository.root, 'rev-parse', 'HEAD') == repository.current


def test_dispatch_cannot_publish_an_unselected_branch(repository):
    with pytest.raises(ValueError, match='explicit release_tag'):
        prepare_release(
            repository.root,
            'workflow_dispatch',
            'refs/heads/main',
            repository.current,
            publish=True,
        )


@pytest.mark.parametrize('event', ['push', 'pull_request', 'schedule'])
def test_other_events_cannot_supply_manual_publication_inputs(repository, event):
    with pytest.raises(ValueError):
        prepare_release(
            repository.root,
            event,
            'refs/heads/main',
            repository.current,
            release_tag='v0.2.0',
            publish=True,
        )


@pytest.mark.parametrize('tag', ['main', 'refs/tags/v0.2.0', 'v0.2.0\npublish=true', 'v../0.2.0'])
def test_invalid_tags_fail_before_any_fetch(repository, tag):
    fetch_head = repository.root / '.git/FETCH_HEAD'
    assert not fetch_head.exists()
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        prepare_release(
            repository.root,
            'workflow_dispatch',
            'refs/heads/main',
            repository.current,
            release_tag=tag,
            publish=True,
        )
    assert not fetch_head.exists()


def test_a_branch_with_no_matching_tag_cannot_be_published(repository):
    git(repository.origin, 'branch', 'v0.9.0', repository.current)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_release(
            repository.root,
            'workflow_dispatch',
            'refs/heads/main',
            repository.current,
            release_tag='v0.9.0',
            publish=True,
        )


def test_existing_but_mismatched_tag_is_rejected(repository):
    git(repository.origin, 'tag', 'v0.9.0', repository.old)
    with pytest.raises(ValueError, match='does not match package version'):
        prepare_release(
            repository.root,
            'workflow_dispatch',
            'refs/heads/main',
            repository.current,
            release_tag='v0.9.0',
            publish=True,
        )


@pytest.mark.parametrize('assignment', ["__version__ = '0.4.0'", "__version__ = str('0.4.0')"])
def test_version_is_parsed_without_executing_release_code(repository, tmp_path, assignment):
    sentinel = tmp_path / 'must-not-be-created'
    (repository.source / 'batbot/_version.py').write_text(
        f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n{assignment}\n',
        encoding='utf8',
    )
    git(repository.source, 'commit', '-am', 'Version with executable side effects')
    git(repository.source, 'push', 'origin', 'HEAD:refs/tags/v0.4.0')
    arguments = (
        repository.root,
        'workflow_dispatch',
        'refs/heads/main',
        repository.current,
        'v0.4.0',
        True,
    )
    if assignment.endswith("str('0.4.0')"):
        with pytest.raises(ValueError):
            prepare_release(*arguments)
    else:
        result = prepare_release(*arguments)
        assert result['commit'] == git(repository.source, 'rev-parse', 'HEAD')
        assert result['publish'] == 'true'
    assert not sentinel.exists()


@pytest.mark.parametrize('publish', ['false', 'true'])
def test_workflow_entry_point_emits_immutable_source_and_preserves_outputs(
    repository, tmp_path, publish
):
    output = tmp_path / 'github-output'
    output.write_text('previous=value\n', encoding='utf8')
    environment = {
        **os.environ,
        'GITHUB_EVENT_NAME': 'workflow_dispatch',
        'GITHUB_REF': 'refs/heads/main',
        'GITHUB_SHA': repository.current,
        'RELEASE_TAG': 'v0.2.0',
        'PUBLISH': publish,
        'GITHUB_OUTPUT': str(output),
    }
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=repository.root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        'commit': repository.old,
        'release_tag': 'v0.2.0',
        'publish': publish,
    }
    assert output.read_text(encoding='utf8') == (
        f'previous=value\ncommit={repository.old}\nrelease_tag=v0.2.0\npublish={publish}\n'
    )


@pytest.mark.parametrize(
    'tag,message',
    [('', 'explicit release_tag'), ('v9.9.9', "couldn't find remote ref refs/tags/v9.9.9")],
)
def test_failed_preparation_emits_no_publication_authorization(repository, tmp_path, tag, message):
    output = tmp_path / 'github-output'
    environment = {
        **os.environ,
        'GITHUB_EVENT_NAME': 'workflow_dispatch',
        'GITHUB_REF': 'refs/heads/main',
        'GITHUB_SHA': repository.current,
        'RELEASE_TAG': tag,
        'PUBLISH': 'true',
        'GITHUB_OUTPUT': str(output),
    }
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=repository.root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert message in completed.stderr
    assert not output.exists()
