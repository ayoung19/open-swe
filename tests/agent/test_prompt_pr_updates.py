import pytest

from agent.prompt import construct_system_prompt


@pytest.mark.parametrize("source", ["dashboard", "slack", "github", "linear", "desktop"])
def test_pr_update_recovery_renders_as_a_single_ordered_workflow(source: str) -> None:
    prompt = construct_system_prompt(working_dir="/workspace", source=source)

    edit_index = prompt.index("with `gh pr edit`")
    fallback_index = prompt.index("`repository.pullRequest.projectCards`")
    patch_index = prompt.index("`gh api --method PATCH repos/<owner>/<repo>/pulls/<number>`")
    verify_index = prompt.index("verify the requested metadata on the same PR")

    assert edit_index < fallback_index < patch_index < verify_index
    assert prompt.count("`repository.pullRequest.projectCards`") == 1
    assert "Do not make a second REST attempt" in prompt
    assert "A 401, 403, 404, permission, or access failure is genuine" in prompt


def test_desktop_pr_creation_override_preserves_existing_pr_recovery() -> None:
    hosted_prompt = construct_system_prompt(working_dir="/workspace")
    desktop_prompt = construct_system_prompt(working_dir="/workspace", source="desktop")

    assert "For desktop runs, open new PRs with `gh pr create`" not in hosted_prompt
    assert desktop_prompt.index("`repository.pullRequest.projectCards`") < desktop_prompt.index(
        "For desktop runs, open new PRs with `gh pr create`"
    )
    assert "overrides only the hosted new-PR creation rules" in desktop_prompt
    assert "the existing-PR update rules still apply" in desktop_prompt


def test_pr_body_fallback_serializes_the_body_file_without_shell_interpolation() -> None:
    prompt = construct_system_prompt(working_dir="/workspace")
    fallback = prompt.split("`repository.pullRequest.projectCards`", maxsplit=1)[1].split(
        "After that PATCH succeeds", maxsplit=1
    )[0]

    assert "`jq -Rs '{body: .}' < /tmp/pr-body.md" in fallback
    assert "--input -`" in fallback
    assert "without shell interpolation" in fallback
    assert "never use `POST`" in fallback
