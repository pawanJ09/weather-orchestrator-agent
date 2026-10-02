"""
weather-orchestrator-agent (Style 2: Classic Bedrock Converse API)

The real agentic loop in this architecture. Unlike the other three Lambdas
(each a single call, no iteration), this one genuinely needs to decide,
turn by turn, which of three tools to call based on what it's learned so
far -- current-weather-lambda, forecast-weather-lambda, and
outfit-suggestion-agent, all invoked directly via boto3 lambda.invoke()
(no Gateway, no MCP, no Action Groups).

The loop is driven by Bedrock Converse's tool-use contract: a response's
stopReason comes back "tool_use" when the model wants to call something,
with a toolUse block naming which tool and with what input. We execute
that tool, append the result as a toolResult block, and call converse()
again -- repeating until stopReason is "end_turn" (the model is done) or
MAX_ITERATIONS is hit (a safety cap so a misbehaving model can't loop
until Lambda's own timeout kills the function uninformatively).

get_outfit_suggestion is a genuine *optional* tool here: the model decides
whether the question calls for it (see config/instructions.txt) -- it is
not called automatically just because it exists.

Invoked directly (intended to sit behind a Lambda Function URL per Style
2's architecture -- see README for why that part isn't automated in
deploy.yml) or via a direct boto3 lambda.invoke() with a flat JSON event,
same as a manual/CI smoke test.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import boto3

# Same inference-profile caveat as outfit-suggestion-agent: confirm this
# is still correct for your account/region, and keep
# iam/execution-role-policy.json's ARNs in sync if you change it.
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

MAX_TOKENS = 500
TEMPERATURE = 0.3
MAX_ITERATIONS = 6

# Overridable via env vars if you ever rename the sibling functions;
# defaults match the actual deployed names in this project.
CURRENT_WEATHER_FUNCTION = os.environ.get("CURRENT_WEATHER_FUNCTION_NAME", "current-weather-lambda")
FORECAST_FUNCTION = os.environ.get("FORECAST_FUNCTION_NAME", "forecast-weather-lambda")
OUTFIT_SUGGESTION_FUNCTION = os.environ.get(
    "OUTFIT_SUGGESTION_FUNCTION_NAME", "outfit-suggestion-agent"
)

TOOL_NAME_TO_FUNCTION = {
    "get_current_weather": CURRENT_WEATHER_FUNCTION,
    "get_forecast": FORECAST_FUNCTION,
    "get_outfit_suggestion": OUTFIT_SUGGESTION_FUNCTION,
}

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_current_weather",
                "description": (
                    "Get current weather conditions (temperature, humidity, "
                    "precipitation, wind) for a location."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_forecast",
                "description": (
                    "Get a 10-day daily forecast (high/low temps, precipitation "
                    "chance, wind) for a location."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_outfit_suggestion",
                "description": (
                    "Get a short clothing recommendation given a plain-text "
                    "summary of weather conditions. Only call this if the user "
                    "is asking what to wear or for similar clothing advice -- "
                    "not for plain weather lookups."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "inputText": {
                                "type": "string",
                                "description": (
                                    "A plain-text summary of the relevant weather "
                                    "conditions to base the recommendation on."
                                ),
                            }
                        },
                        "required": ["inputText"],
                    }
                },
            }
        },
    ]
}


class OrchestratorError(Exception):
    """Raised when the loop can't produce a final answer."""


def _load_instructions() -> str:
    """Same env-var-first, local-file-fallback pattern as outfit-suggestion-agent."""
    env_value = os.environ.get("ORCHESTRATOR_INSTRUCTIONS")
    if env_value:
        return env_value

    local_path = Path(__file__).resolve().parent.parent / "config" / "instructions.txt"
    if local_path.exists():
        return local_path.read_text()

    raise OrchestratorError(
        "No ORCHESTRATOR_INSTRUCTIONS env var set and config/instructions.txt "
        "not found locally. In Lambda, this should always be set by the "
        "deploy workflow."
    )


def _invoke_tool_lambda(function_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Act: actually call one of the three sibling Lambdas. Every one of them
    returns {"statusCode": ..., "body": "<json string>"} -- this unwraps
    that into either the parsed body (success) or an {"error": ...} dict
    the model can see and react to, rather than raising and killing the
    whole orchestrator run over one failed tool call.
    """
    client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    try:
        response = client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 -- any boto3/botocore failure here
        # means "couldn't even reach the tool," which the model should see
        # as a tool error, not a crash of the whole orchestrator.
        return {"error": f"Failed to invoke {function_name}: {exc}"}

    raw_payload = response["Payload"].read()

    if response.get("FunctionError"):
        return {
            "error": f"{function_name} crashed: {raw_payload.decode('utf-8', errors='replace')}"
        }

    try:
        parsed = json.loads(raw_payload)
        body = json.loads(parsed.get("body", "{}"))
    except (json.JSONDecodeError, AttributeError) as exc:
        return {"error": f"{function_name} returned an unparseable response: {exc}"}

    if parsed.get("statusCode") != 200:
        return {
            "error": body.get(
                "error", f"{function_name} returned status {parsed.get('statusCode')}"
            )
        }

    return body


def _extract_final_text(message: dict[str, Any]) -> str:
    for block in message.get("content", []):
        if "text" in block:
            return block["text"]
    raise OrchestratorError("Final model response had no text content")


def run_orchestrator_loop(question: str, latitude: float, longitude: float) -> str:
    """The actual Reason -> Act -> Observe -> repeat loop."""
    instructions = _load_instructions()
    bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    initial_text = (
        f"The user is at latitude {latitude}, longitude {longitude}. They ask: {question}"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": initial_text}]}]

    for _ in range(MAX_ITERATIONS):
        try:
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=messages,
                system=[{"text": instructions}],
                toolConfig=TOOL_CONFIG,
                inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
            )
        except Exception as exc:  # noqa: BLE001 -- see fetch_outfit_recommendation
            raise OrchestratorError(f"Bedrock Converse call failed: {exc}") from exc

        stop_reason = response.get("stopReason")
        assistant_message = response["output"]["message"]
        messages.append(assistant_message)

        if stop_reason != "tool_use":
            return _extract_final_text(assistant_message)

        # Act: execute every tool_use block in this turn, Observe: feed
        # each result back as a toolResult in the next user message.
        tool_result_blocks = []
        for block in assistant_message.get("content", []):
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            tool_name = tool_use["name"]
            tool_input = tool_use.get("input", {})

            function_name = TOOL_NAME_TO_FUNCTION.get(tool_name)
            if function_name is None:
                result: dict[str, Any] = {"error": f"Unknown tool requested: {tool_name}"}
            else:
                result = _invoke_tool_lambda(function_name, tool_input)

            tool_result_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result}],
                    }
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

    raise OrchestratorError(f"Exceeded {MAX_ITERATIONS} tool-use iterations without a final answer")


def _is_function_url_event(event: dict[str, Any]) -> bool:
    return "requestContext" in event and "http" in event.get("requestContext", {})


def _extract_payload(event: dict[str, Any]) -> dict[str, Any]:
    """
    Direct boto3 lambda.invoke() sends the payload as the event itself.
    A Lambda Function URL instead wraps it in an API-Gateway-style event
    with the real JSON body as a string (optionally base64-encoded) under
    event["body"]. Handling both means this works identically whether
    it's smoke-tested via `aws lambda invoke` or called through a Function
    URL someone sets up later (see README).
    """
    if not _is_function_url_event(event):
        return event

    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def _plain_response(http_status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": http_status, "body": json.dumps(body)}


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    payload = _extract_payload(event)

    question = payload.get("question")
    if not question or not isinstance(question, str):
        return _plain_response(400, {"error": "Missing or invalid 'question'"})

    try:
        latitude = float(payload.get("latitude"))
        longitude = float(payload.get("longitude"))
    except TypeError, ValueError:
        return _plain_response(400, {"error": "Missing or invalid 'latitude'/'longitude'"})

    try:
        answer = run_orchestrator_loop(question, latitude, longitude)
    except OrchestratorError as exc:
        return _plain_response(502, {"error": str(exc)})

    return _plain_response(200, {"answer": answer})
