import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src import handler


def _tool_use_response(tool_name, tool_input, tool_use_id="tool1"):
    return {
        "stopReason": "tool_use",
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
    }


def _final_text_response(text):
    return {
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
    }


def _lambda_invoke_result(status_code, body_dict):
    payload_bytes = json.dumps({"statusCode": status_code, "body": json.dumps(body_dict)}).encode()
    mock_payload = MagicMock()
    mock_payload.read.return_value = payload_bytes
    return {"Payload": mock_payload}


class TestInvokeToolLambda:
    @patch("src.handler.boto3.client")
    def test_success_returns_parsed_body(self, mock_boto_client):
        mock_lambda = MagicMock()
        mock_lambda.invoke.return_value = _lambda_invoke_result(200, {"temperature_f": 70})
        mock_boto_client.return_value = mock_lambda

        result = handler._invoke_tool_lambda(
            "current-weather-lambda", {"latitude": 1, "longitude": 2}
        )

        assert result == {"temperature_f": 70}
        mock_lambda.invoke.assert_called_once()
        assert mock_lambda.invoke.call_args.kwargs["FunctionName"] == "current-weather-lambda"

    @patch("src.handler.boto3.client")
    def test_non_200_returns_error_dict(self, mock_boto_client):
        mock_lambda = MagicMock()
        mock_lambda.invoke.return_value = _lambda_invoke_result(400, {"error": "bad input"})
        mock_boto_client.return_value = mock_lambda

        result = handler._invoke_tool_lambda("current-weather-lambda", {})

        assert result == {"error": "bad input"}

    @patch("src.handler.boto3.client")
    def test_function_error_returns_error_dict(self, mock_boto_client):
        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'{"errorMessage": "boom"}'
        mock_lambda.invoke.return_value = {"Payload": mock_payload, "FunctionError": "Unhandled"}
        mock_boto_client.return_value = mock_lambda

        result = handler._invoke_tool_lambda("current-weather-lambda", {})

        assert "error" in result
        assert "crashed" in result["error"]

    @patch("src.handler.boto3.client")
    def test_invoke_exception_returns_error_dict(self, mock_boto_client):
        mock_lambda = MagicMock()
        mock_lambda.invoke.side_effect = Exception("network blip")
        mock_boto_client.return_value = mock_lambda

        result = handler._invoke_tool_lambda("current-weather-lambda", {})

        assert "error" in result
        assert "Failed to invoke" in result["error"]


class TestOrchestratorLoop:
    def setup_method(self):
        os.environ["ORCHESTRATOR_INSTRUCTIONS"] = "test instructions"

    def teardown_method(self):
        os.environ.pop("ORCHESTRATOR_INSTRUCTIONS", None)

    @patch("src.handler._invoke_tool_lambda")
    @patch("src.handler.boto3.client")
    def test_single_tool_call_then_final_answer(self, mock_boto_client, mock_invoke_tool):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [
            _tool_use_response("get_current_weather", {"latitude": 40.7, "longitude": -74.0}),
            _final_text_response("It's 70F and sunny right now."),
        ]
        mock_boto_client.return_value = mock_bedrock
        mock_invoke_tool.return_value = {"temperature_f": 70}

        result = handler.run_orchestrator_loop("What's the weather?", 40.7, -74.0)

        assert result == "It's 70F and sunny right now."
        assert mock_bedrock.converse.call_count == 2
        mock_invoke_tool.assert_called_once_with(
            "current-weather-lambda", {"latitude": 40.7, "longitude": -74.0}
        )

    @patch("src.handler._invoke_tool_lambda")
    @patch("src.handler.boto3.client")
    def test_calls_outfit_suggestion_only_when_model_asks(self, mock_boto_client, mock_invoke_tool):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [
            _tool_use_response("get_current_weather", {"latitude": 1, "longitude": 2}),
            _tool_use_response(
                "get_outfit_suggestion",
                {"inputText": "38F and windy"},
                tool_use_id="tool2",
            ),
            _final_text_response("Wear a warm coat -- it's 38F and windy."),
        ]
        mock_boto_client.return_value = mock_bedrock
        mock_invoke_tool.side_effect = [
            {"temperature_f": 38},
            {"recommendation": "Wear a warm coat."},
        ]

        result = handler.run_orchestrator_loop("What should I wear?", 1, 2)

        assert "warm coat" in result
        assert mock_invoke_tool.call_count == 2
        second_call = mock_invoke_tool.call_args_list[1]
        assert second_call.args[0] == "outfit-suggestion-agent"
        assert second_call.args[1] == {"inputText": "38F and windy"}

    @patch("src.handler._invoke_tool_lambda")
    @patch("src.handler.boto3.client")
    def test_tool_error_does_not_crash_the_loop(self, mock_boto_client, mock_invoke_tool):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [
            _tool_use_response("get_current_weather", {"latitude": 1, "longitude": 2}),
            _final_text_response("I couldn't get current conditions, but here's what I know."),
        ]
        mock_boto_client.return_value = mock_bedrock
        mock_invoke_tool.return_value = {"error": "upstream failure"}

        result = handler.run_orchestrator_loop("weather?", 1, 2)

        assert "couldn't get current conditions" in result
        # Confirm the error was actually passed back to the model as a
        # toolResult rather than silently dropped. NOTE: `messages` is one
        # mutable list reused across the whole loop, and call_args_list
        # captures it by reference, not a snapshot -- by now it holds the
        # loop's *final* state (4 entries: initial user text, the tool_use
        # turn, the toolResult turn, and the final assistant text appended
        # last), so the toolResult we want is at index -2, not -1.
        captured_messages = mock_bedrock.converse.call_args_list[1].kwargs["messages"]
        tool_result_content = captured_messages[-2]["content"][0]["toolResult"]["content"]
        assert tool_result_content[0]["json"] == {"error": "upstream failure"}

    @patch("src.handler._invoke_tool_lambda")
    @patch("src.handler.boto3.client")
    def test_unknown_tool_name_handled_gracefully(self, mock_boto_client, mock_invoke_tool):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [
            _tool_use_response("some_nonexistent_tool", {}),
            _final_text_response("Final answer anyway."),
        ]
        mock_boto_client.return_value = mock_bedrock

        result = handler.run_orchestrator_loop("weather?", 1, 2)

        assert result == "Final answer anyway."
        mock_invoke_tool.assert_not_called()

    @patch("src.handler._invoke_tool_lambda")
    @patch("src.handler.boto3.client")
    def test_exceeds_max_iterations_raises(self, mock_boto_client, mock_invoke_tool):
        mock_bedrock = MagicMock()
        # Always wants a tool, never finishes.
        mock_bedrock.converse.side_effect = [
            _tool_use_response("get_current_weather", {"latitude": 1, "longitude": 2})
        ] * (handler.MAX_ITERATIONS + 2)
        mock_boto_client.return_value = mock_bedrock
        mock_invoke_tool.return_value = {"temperature_f": 70}

        with pytest.raises(handler.OrchestratorError):
            handler.run_orchestrator_loop("weather?", 1, 2)

        assert mock_bedrock.converse.call_count == handler.MAX_ITERATIONS

    @patch("src.handler.boto3.client")
    def test_bedrock_call_failure_raises_orchestrator_error(self, mock_boto_client):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = Exception("throttled")
        mock_boto_client.return_value = mock_bedrock

        with pytest.raises(handler.OrchestratorError):
            handler.run_orchestrator_loop("weather?", 1, 2)


class TestPayloadExtraction:
    def test_direct_invoke_event_passed_through(self):
        event = {"question": "weather?", "latitude": 1, "longitude": 2}
        assert handler._extract_payload(event) == event

    def test_function_url_event_parses_json_body(self):
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps({"question": "weather?", "latitude": 1, "longitude": 2}),
        }
        result = handler._extract_payload(event)
        assert result == {"question": "weather?", "latitude": 1, "longitude": 2}

    def test_function_url_event_decodes_base64_body(self):
        inner = json.dumps({"question": "weather?", "latitude": 1, "longitude": 2})
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": base64.b64encode(inner.encode()).decode(),
            "isBase64Encoded": True,
        }
        result = handler._extract_payload(event)
        assert result == {"question": "weather?", "latitude": 1, "longitude": 2}


class TestLambdaHandler:
    @patch("src.handler.run_orchestrator_loop")
    def test_success_returns_200(self, mock_loop):
        mock_loop.return_value = "It's sunny."
        event = {"question": "weather?", "latitude": 1, "longitude": 2}

        result = handler.lambda_handler(event, None)

        assert result["statusCode"] == 200
        assert json.loads(result["body"])["answer"] == "It's sunny."

    def test_missing_question_returns_400(self):
        result = handler.lambda_handler({"latitude": 1, "longitude": 2}, None)
        assert result["statusCode"] == 400

    def test_missing_coordinates_returns_400(self):
        result = handler.lambda_handler({"question": "weather?"}, None)
        assert result["statusCode"] == 400

    def test_non_numeric_coordinates_returns_400(self):
        event = {"question": "weather?", "latitude": "not-a-number", "longitude": 2}
        result = handler.lambda_handler(event, None)
        assert result["statusCode"] == 400

    @patch("src.handler.run_orchestrator_loop")
    def test_orchestrator_error_returns_502(self, mock_loop):
        mock_loop.side_effect = handler.OrchestratorError("boom")
        event = {"question": "weather?", "latitude": 1, "longitude": 2}

        result = handler.lambda_handler(event, None)

        assert result["statusCode"] == 502
