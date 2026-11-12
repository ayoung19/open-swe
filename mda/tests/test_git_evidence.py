import asyncio
import base64
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from deepagents.backends.local_shell import LocalShellBackend
from rt.git_evidence import GitCheckout, GitSnapshot, TurnGit
from rt.sandbox_script import SandboxScript, SandboxScriptError
from runtime_api.operations import PullRequestHead


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True)


def _checkout(root: Path, name: str, remote: str, branch: str = "work") -> Path:
    path = root / name
    _git("init", "-b", branch, str(path))
    for key, value in [("user.name", "Test"), ("user.email", "test@example.com")]:
        _git("-C", str(path), "config", key, value)
    _git("-C", str(path), "remote", "add", "origin", remote)
    (path / "file").write_text("initial")
    _git("-C", str(path), "add", "file")
    _git("-C", str(path), "commit", "-m", "initial")
    return path


@pytest.fixture
def sandbox() -> LocalShellBackend:
    return LocalShellBackend(virtual_mode=False)


async def test_the_work_order_names_the_turns_repository_and_a_pull_request_its_branch(
    tmp_path: Path, sandbox: LocalShellBackend
) -> None:
    actual = _checkout(tmp_path, "actual", "git@github.com:org/actual.git", "local")
    later = _checkout(tmp_path, "later", "https://github.com/org/later.git")
    git = TurnGit()
    git.start(sandbox, str(tmp_path))

    # A tool changes one checkout; the next model call fingerprints it.
    await git.changing()
    (actual / "file").write_text("changed")
    git.changed(10)
    git.settle()
    await git.metadata()
    # Then another, and a pull request is opened for the first one's work.
    await git.changing()
    _git("-C", str(later), "checkout", "-b", "later")
    git.changed(20)
    git.found_pull_requests([PullRequestHead(repo="org/actual", branch="Exact/PR-head")], 30)

    assert await git.metadata() == {
        "open_swe_git_targets": [
            {"repo": "org/actual", "branch": "Exact/PR-head"},
            {"repo": "org/later", "branch": "later"},
        ],
        "ls_git_repo": "org/actual",
        "ls_git_branch": "Exact/PR-head",
    }


async def test_a_turn_that_changed_nothing_waits_for_no_fingerprint(
    tmp_path: Path, sandbox: LocalShellBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def never(*args: object, **kwargs: object) -> dict[str, object]:
        started.set()
        await asyncio.Event().wait()
        return {}

    monkeypatch.setattr(SandboxScript, "run", never)
    git = TurnGit()
    git.start(sandbox, str(tmp_path))
    await started.wait()

    assert await asyncio.wait_for(git.metadata(), 1) == {}


async def test_a_sandbox_the_script_cannot_read_leaves_only_the_pull_requests(
    tmp_path: Path, sandbox: LocalShellBackend
) -> None:
    git = TurnGit()
    git.start(sandbox, str(tmp_path / "missing"))
    await git.changing()
    git.changed(1)
    git.found_pull_requests([PullRequestHead(repo="org/actual", branch="Exact/head")], 2)

    assert await git.metadata() == {
        "open_swe_git_targets": [{"repo": "org/actual", "branch": "Exact/head"}],
        "ls_git_repo": "org/actual",
        "ls_git_branch": "Exact/head",
    }


async def test_the_script_sees_more_edits_to_an_already_dirty_checkout(
    tmp_path: Path, sandbox: LocalShellBackend
) -> None:
    for name in ("a-no-origin", "b-unborn"):
        _git("init", str(tmp_path / name))
    _git("-C", str(tmp_path / "b-unborn"), "remote", "add", "origin", "https://github.com/o/u.git")
    repo = _checkout(tmp_path, "repo", "git@github.com:org/actual.git")
    script = SandboxScript("git_trace.py")

    async def fingerprint() -> GitSnapshot:
        return GitSnapshot.model_validate(await script.run(sandbox, {"root": str(tmp_path)}))

    (repo / "file").write_text("dirty")
    first = (await fingerprint()).repos
    (repo / "file").write_text("different dirty content")
    second = (await fingerprint()).repos

    assert list(first) == list(second) == [str(repo)]
    assert first[str(repo)].repo == "org/actual"
    assert first[str(repo)].branch == "work"
    assert first[str(repo)].dirty != second[str(repo)].dirty

    broken = tmp_path / "broken"
    shutil.copytree(repo, broken)
    (broken / ".git" / "index").write_bytes(b"corrupt")
    before = await fingerprint()
    assert list(before.repos) == [str(repo)]
    assert list(before.skipped) == [str(broken)]
    recovered = GitSnapshot(repos={str(broken): GitCheckout(repo="org/actual", branch="work")})
    assert recovered.changes_since(before) == []
    with pytest.raises(SandboxScriptError, match="Could not inspect workspace git state"):
        await script.run(sandbox, {"root": str(tmp_path / "missing")})


async def test_fingerprints_leave_a_stale_index_untouched_and_allow_fast_forward_pulls(
    tmp_path: Path, sandbox: LocalShellBackend
) -> None:
    upstream = _checkout(tmp_path, "upstream", "https://github.com/org/actual.git")
    repo = tmp_path / "repo"
    _git("clone", str(upstream), str(repo))
    _git("-C", str(repo), "remote", "set-url", "origin", "https://github.com/org/actual.git")
    os.utime(repo / "file", (1, 1))
    index = repo / ".git" / "index"
    before = (index.read_bytes(), index.stat().st_mtime_ns)
    script = SandboxScript("git_trace.py")
    await script.run(sandbox, {"root": str(repo)})
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before

    for number in range(5):
        (upstream / "new").write_text(str(number))
        _git("-C", str(upstream), "add", "new")
        _git("-C", str(upstream), "commit", "-m", "advance")
        os.utime(repo / "file", (number + 1, number + 1))
        _, pull = await asyncio.gather(
            script.run(sandbox, {"root": str(repo)}),
            sandbox.aexecute(
                f"git -C {shlex.quote(str(repo))} pull --ff-only {shlex.quote(str(upstream))} work"
            ),
        )
        assert pull.exit_code == 0, pull.output
        assert (repo / "new").read_text() == str(number)
        assert not (repo / ".git" / "index.lock").exists()


async def test_staged_unstaged_and_untracked_content_remain_distinguishable(
    tmp_path: Path, sandbox: LocalShellBackend
) -> None:
    repo = _checkout(tmp_path, "repo", "https://github.com/org/actual.git")
    script = SandboxScript("git_trace.py")
    fingerprints: list[object] = []
    for content in (b"staged", b"initial", b"unstaged", b"more unstaged", b"\0binary", b"\0edited"):
        (repo / "file").write_bytes(content)
        if content == b"staged":
            _git("-C", str(repo), "add", "file")
        fingerprints.append(await script.run(sandbox, {"root": str(repo)}))
    (repo / "untracked").write_text("new")
    fingerprints.append(await script.run(sandbox, {"root": str(repo)}))
    assert all(left != right for left, right in zip(fingerprints, fingerprints[1:], strict=False))


async def test_tools_and_fingerprints_exclude_each_other_in_both_orders(
    tmp_path: Path, sandbox: LocalShellBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready, scanning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def scan(*args: object, **kwargs: object) -> dict[str, object]:
        scanning.set()
        await finish.wait()
        scanning.clear()
        return {"repos": {}}

    async def setup() -> None:
        await ready.wait()

    monkeypatch.setattr(SandboxScript, "run", scan)
    git = TurnGit()
    git.start(sandbox, str(tmp_path), setup)
    tool = asyncio.create_task(git.changing())
    await asyncio.sleep(0)
    assert not tool.done()
    assert not scanning.is_set()
    ready.set()
    await scanning.wait()
    assert not tool.done()
    finish.set()
    await tool
    git.changed(1)

    finish.clear()
    git.settle()
    await scanning.wait()
    tool = asyncio.create_task(git.changing())
    await asyncio.sleep(0)
    assert not tool.done()
    finish.set()
    await tool
    # Another parallel tool may finish while execute is still running.
    await git.changing()
    git.changed(2)
    finish.clear()
    git.settle()
    await asyncio.sleep(0)
    assert not scanning.is_set()
    git.changed(3)
    git.settle()
    await scanning.wait()
    finish.set()
    await git.metadata()


async def test_scan_deadline_reaps_git_before_a_tool_can_proceed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _checkout(tmp_path, "repo", "https://github.com/org/actual.git")
    pid_file = tmp_path / "pids"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command = bin_dir / "git"
    command.write_text(f"#!/bin/sh\necho $$ >> {shlex.quote(str(pid_file))}\nexec sleep 60\n")
    command.chmod(0o755)
    sandbox = LocalShellBackend(virtual_mode=False, env={"PATH": f"{bin_dir}:{os.environ['PATH']}"})
    script = SandboxScript("git_trace.py")
    source = (
        (Path(__file__).parents[1] / "rt/sandbox_scripts/git_trace.py")
        .read_text()
        .replace("DEADLINE_SECONDS = 20", "DEADLINE_SECONDS = 0.2")
    )
    script.__dict__["_source"] = shlex.quote(base64.b64encode(source.encode()).decode())
    monkeypatch.setattr("rt.git_evidence._SCRIPT", script)
    git = TurnGit()
    git.start(sandbox, str(tmp_path))
    await asyncio.wait_for(git.changing(), 5)
    assert pid_file.exists()
    for pid in pid_file.read_text().splitlines():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid), 0)
    git.changed(1)
