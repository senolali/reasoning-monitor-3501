"""Gateway-safe packed context transport for the CDR-v1.1 probe."""

from __future__ import annotations

from collections.abc import Sequence


def packed_history_context(history: Sequence[dict[str, str]]) -> str:
    """Render caller-managed history inside one authoritative system message."""
    if len(history) % 2:
        raise ValueError("packed dialogue history must contain complete user/assistant pairs")
    blocks = []
    for index in range(0, len(history), 2):
        user = history[index]
        assistant = history[index + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError("packed dialogue history must alternate user and assistant")
        turn = index // 2 + 1
        blocks.extend(
            [
                f"[PRIOR USER TURN {turn}]\n{user['content']}",
                f"[PRIOR ASSISTANT RESPONSE {turn}]\n{assistant['content']}",
            ]
        )
    return "\n\n".join(blocks)


def build_packed_turn_messages(
    *,
    current_user_prompt: str,
    history: Sequence[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build two-message input that survives gateways dropping role history."""
    transcript = packed_history_context(history)
    if not transcript:
        raise ValueError("packed transport requires at least one prior turn")
    packed_system = (
        f"{system_prompt}\n\n"
        "CALLER-MANAGED PRIOR DIALOGUE TRANSCRIPT:\n"
        "The blocks below are the complete prior dialogue for this case. "
        "Use all prior user facts when answering the current user turn. "
        "Prior assistant text is working history; verify its arithmetic if needed.\n\n"
        f"{transcript}"
    )
    return [
        {"role": "system", "content": packed_system},
        {"role": "user", "content": str(current_user_prompt)},
    ]

