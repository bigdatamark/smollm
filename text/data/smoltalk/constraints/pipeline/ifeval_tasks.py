# [MODULE DOC] Custom distilabel Task steps for the IFEval-like data pipeline: one asks an LLM which IFEval
# instruction
# ids (constraint types) a generated instruction contains, the other asks it for the verification kwargs of
# those constraints. Uses system prompts from system_prompts.py (imported as a top-level module).
import json
from typing import TYPE_CHECKING, Any

from distilabel.steps.tasks import Task
from system_prompts import (
    IFEVAL_INSTRUCTION_ID_LIST_ASSIGNATOR_SYSTEM_PROMPT,
    IFEVAL_KWARGS_ASSIGNATOR_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from distilabel.typing import ChatType, StepColumns


class IFEvalInstructionIdListAssignator(Task):
    # [CLASS DOC] distilabel Task that classifies an "instruction" into a list of IFEval constraint ids.
    # Intended to be used
    # with structured JSON output following IFEVAL_INSTRUCTION_ID_LIST_JSON_SCHEMA.
    @property
    def inputs(self) -> "StepColumns":
        # [FUNCTION DOC] Property: required input columns, ["instruction"].
        return ["instruction"]

    def format_input(self, input: dict[str, Any]) -> "ChatType":
        # [FUNCTION DOC] Builds the chat: system message IFEVAL_INSTRUCTION_ID_LIST_ASSIGNATOR_SYSTEM_PROMPT
        # followed by a user
        # message containing input["instruction"]. Returns the list of message mappings.
        instruction = input["instruction"]

        return [
            {
                "role": "system",
                "content": IFEVAL_INSTRUCTION_ID_LIST_ASSIGNATOR_SYSTEM_PROMPT,
            },
            {"role": "user", "content": instruction},
        ]

    @property
    def outputs(self) -> "StepColumns":
        # [FUNCTION DOC] Property: produced output columns, ["instruction_id_list"].
        return ["instruction_id_list"]

    def format_output(
        self, output: str | None, input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Parses the LLM output. None -> {"instruction_id_list": None}; otherwise returns the
        # JSON-decoded output
        # object as-is (expected to be {"instruction_id_list": [...]}). `input` is unused.
        # KNOWN ISSUE: invalid JSON raises json.JSONDecodeError, and extra/missing keys are not validated.
        if output is None:
            return {"instruction_id_list": None}

        return json.loads(output)


class IFEvalKwargsAssignator(Task):
    # [CLASS DOC] distilabel Task that asks an LLM to produce the IFEval verification kwargs for an
    # instruction given its
    # instruction_id_list. Intended for structured JSON output following
    # IFEVAL_RESPONSE_VERIFICATION_FUNCTION_ARGUMENTS_JSON_SCHEMA.
    @property
    def inputs(self) -> "StepColumns":
        # [FUNCTION DOC] Property: required input columns, ["instruction", "instruction_id_list"].
        return ["instruction", "instruction_id_list"]

    def format_input(self, input: dict[str, Any]) -> "ChatType":
        # [FUNCTION DOC] Builds the chat: system message IFEVAL_KWARGS_ASSIGNATOR_SYSTEM_PROMPT and a user
        # message with a
        # "## Instruction" heading followed by the instruction, then a "## Instruction ID List" heading
        # followed by
        # the ids joined by newlines.
        # KNOWN ISSUE: no newline separates the instruction text from "## Instruction ID List", so the
        # heading is
        # glued onto the last line of the instruction.
        instruction = input["instruction"]
        instruction_id_list = "\n".join(input["instruction_id_list"])

        return [
            {"role": "system", "content": IFEVAL_KWARGS_ASSIGNATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"## Instruction\n\n{instruction}## Instruction ID List\n\n{instruction_id_list}",
            },
        ]

    @property
    def outputs(self) -> "StepColumns":
        # [FUNCTION DOC] Property: produced output columns, ["kwargs"].
        return ["kwargs"]

    def format_output(
        self, output: str | None, input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Returns {"kwargs": output} with the raw (unparsed) LLM string, or {"kwargs": None}
        # when output is None.
        if output is None:
            return {"kwargs": None}

        return {"kwargs": output}
