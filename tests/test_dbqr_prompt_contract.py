from reasoning_monitor.dbqr_json_plan import parse_plan_response
from reasoning_monitor.dbqr_prompt_contract import (
    PROMPT_CONTRACT_VERSION,
    SYSTEM_PROMPT,
    prompt_contract_sha256,
)


def test_prompt_contract_teaches_the_strict_plan_envelope():
    assert PROMPT_CONTRACT_VERSION == "DBQR-PROMPT-v1.1"
    assert "DBQR-PLAN-v1" in SYSTEM_PROMPT
    assert "FINAL_PLAN_JSON: " in SYSTEM_PROMPT
    assert '"steps"' in SYSTEM_PROMPT
    assert '"op":"query"' in SYSTEM_PROMPT
    assert "Do not use Markdown" in SYSTEM_PROMPT
    assert len(prompt_contract_sha256()) == 64


def test_prompt_contract_example_is_accepted_by_the_parser():
    response = SYSTEM_PROMPT.split("VALID FORMAT EXAMPLE", maxsplit=1)[1]
    response = response.split("\n", maxsplit=1)[1].strip()
    reasoning, plan = parse_plan_response(response)
    assert len(reasoning) == 2
    assert plan["version"] == "DBQR-PLAN-v1"
    assert plan["steps"][0] == {"id": "s1", "op": "query", "table": "table_1"}
