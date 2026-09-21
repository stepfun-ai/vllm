# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager


@pytest.fixture
def tokenizer():
    return make_mock_tokenizer(
        {"<think>": 256, "</think>": 257, "<tool_call>": 258, "</tool_call>": 259},
        special_tokens=[],
    )


@pytest.mark.parametrize("body", ["\n\n", "\nanswer\n\n"])
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_step5_tool_whitespace_and_arguments(tokenizer, mock_request, body, chunk_size):
    mock_request.tools = [
        ChatCompletionToolsParam(
            function={
                "name": "convert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flag": {"type": "boolean"},
                        "empty": {"type": ["string", "null"]},
                        "text": {"type": "string"},
                    },
                },
            }
        )
    ]
    call = (
        "<tool_call>\n<function=convert>\n"
        "<parameter=flag>\nTrue\n</parameter>\n"
        "<parameter=empty>\nNone\n</parameter>\n"
        "<parameter=text>\nNone\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    text = body + call + call
    parser_cls = ToolParserManager.get_tool_parser("step5")
    parser = parser_cls(tokenizer, tools=mock_request.tools)
    result = parser.extract_tool_calls(text, mock_request)
    assert result.content == body
    assert len(result.tool_calls) == 2
    expected = {"flag": True, "empty": None, "text": "None"}
    for tool in result.tool_calls:
        assert tool.function.name == "convert"
        assert json.loads(tool.function.arguments) == expected

    parser = parser_cls(tokenizer, tools=mock_request.tools)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    deltas = simulate_tool_streaming(parser, mock_request, chunks)
    deltas.append((parser.finish_streaming(), text))
    assert collect_content(deltas) == body
    arguments = {0: "", 1: ""}
    for delta, _ in deltas:
        if delta:
            for tool in delta.tool_calls or []:
                if tool.function:
                    arguments[tool.index] += tool.function.arguments or ""
    assert [json.loads(value) for value in arguments.values()] == [expected, expected]


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_step5_reasoning_preserves_bare_boundary_whitespace(
    tokenizer, mock_request, enable_thinking
):
    parser_cls = ReasoningParserManager.get_reasoning_parser("step5")
    parser = parser_cls(
        tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking}
    )
    expected = ("\nreason\n", "\nanswer\n")
    assert (
        parser.extract_reasoning("\nreason\n</think>\nanswer\n", mock_request)
        == expected
    )
    assert (
        simulate_reasoning_streaming(
            parser,
            ["\nreason\n", "</think>", "\nanswer\n"],
            [(), (257,), ()],
        )
        == expected
    )
