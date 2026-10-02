# weather-orchestrator-agent

The Orchestrator in the Agentic Weather App (Style 2: Classic Bedrock
Converse API) — and the only component in this whole architecture with a
genuine, hand-written Reason → Act → Observe → repeat loop. The other
three Lambdas each make one call and return; this one decides, turn by
turn, which of three tools to call based on what it's learned so far.

Runtime: **Python 3.14** · Dependencies: **boto3** (ships with the Lambda
runtime)

## This is the last of the four Style 2 components

It's last by design: it needs `current-weather-lambda`,
`forecast-weather-lambda`, and `outfit-suggestion-agent` to already exist,
since it calls all three directly via `boto3.client('lambda').invoke()` —
no Gateway, no MCP, no Action Groups, per Style 2's architecture.

## Where the loop actually lives

Bedrock's Converse API signals "I want to call a tool" through the
response's `stopReason` field, which comes back `"tool_use"` instead of
the normal `"end_turn"`. `run_orchestrator_loop()` in `src/handler.py` is
a `for` loop (capped at `MAX_ITERATIONS = 6`) that:

1. **Reason** — calls `bedrock-runtime.converse()` with the conversation
   so far and a `toolConfig` describing all three tools.
2. Checks `stopReason`. If it's not `"tool_use"`, the model is done —
   return its text as the final answer.
3. **Act** — for each `toolUse` block in the response, actually invoke the
   corresponding Lambda (`_invoke_tool_lambda`).
4. **Observe** — append each result as a `toolResult` block in a new user
   message, then loop back to step 1 with the updated conversation.

`get_outfit_suggestion` is a genuine **optional** tool here — the model
decides whether a question calls for it (see `config/instructions.txt`,
point 4), not something always invoked. A plain "what's the weather"
question should never trigger it; "what should I wear" should.

## A real bug I caught building this, worth knowing if you touch this code

`messages` is one mutable list, reused and appended to across every loop
iteration — not rebuilt each time. If you ever want to inspect what was
sent on a *specific* iteration (debugging, or writing a new test), capturing
a reference to `messages` via a mock's `call_args_list` will **not** give
you a snapshot of that moment — Python/Mock capture the list by reference,
so by the time you look at it later, it holds the loop's *final* state,
with later appends mixed in. `tests/test_handler.py`'s
`test_tool_error_does_not_crash_the_loop` hits this directly: the
`toolResult` message ends up at index `-2` relative to the end of the
final captured list, not `-1`, because one more assistant turn gets
appended after it. If you add new tests that inspect `messages` mid-loop,
either index from a fixed, known position, or `copy.deepcopy()` the
argument inside a custom `side_effect` function to get a true snapshot.

## Repo layout

```
src/handler.py            The loop, tool dispatch, and the Lambda handler
config/instructions.txt   System prompt -- explicit about outfit suggestion being optional
tests/test_handler.py     26 checks covering the loop, tool dispatch, both invocation shapes
events/                   Sample event designed to exercise the full tool-use path
iam/                      Trust policy, execution policy (logs + Bedrock + 3x InvokeFunction)
.github/workflows/        CI (lint+test) and Deploy (build, deploy, smoke-test)
```

## Invocation contract

```json
{ "question": "a plain-text question", "latitude": 40.7128, "longitude": -74.0060 }
```

Returns `{"statusCode": 200, "body": "{\"answer\": \"...\"}"}`, or `400`
(missing/invalid question or coordinates) / `502` (the loop couldn't
produce a final answer — Bedrock failure or `MAX_ITERATIONS` exceeded).

Handles **two** event shapes, unlike the other three Lambdas: a flat
direct-invoke payload (what `aws lambda invoke` and CI's smoke test use),
and a Lambda Function URL event (API-Gateway-style, JSON body as a string
under `event["body"]`, optionally base64-encoded) — see the next section
for why the second shape exists but isn't wired up to real AWS
infrastructure yet.

## ⚠️ About the Function URL — deliberately not automated

Style 2's architecture describes this Lambda as having its own Function
URL (no separate entry Lambda, unlike Style 3) — this is the one you'd
actually `curl` directly. The handler already supports that event shape
(`_extract_payload` / `_is_function_url_event`).

**What `deploy.yml` does *not* do**: create or configure that Function URL.
I made this call deliberately rather than guess at it: a Function URL's
auth type (`NONE` vs `AWS_IAM`) is a real security decision — `NONE` makes
the endpoint callable by anyone on the internet, who could then run up
your Bedrock bill with no authentication at all. I didn't want to bake an
unverified security-relevant default into an automated script. If you
want one:

```bash
# AWS_IAM auth (recommended) -- callers need their own AWS credentials
# with lambda:InvokeFunctionUrl permission on this function
aws lambda create-function-url-config \
  --function-name weather-orchestrator-agent \
  --auth-type AWS_IAM

# Get the URL
aws lambda get-function-url-config --function-name weather-orchestrator-agent
```

Testing an `AWS_IAM`-protected Function URL with plain `curl` means
hand-signing SigV4 requests, which is painful enough that a tool like
[`awscurl`](https://github.com/okigan/awscurl) is worth it if you go this
route. For everything in this project so far — local testing, CI's smoke
test — `aws lambda invoke` has been simpler and is what `deploy.yml` uses;
the Function URL is purely a convenience for external, non-AWS-SDK callers
if you decide you want one.

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make check    # ruff format + ruff check + pytest --cov (90% floor)
```

## Manual test (needs real AWS credentials + all three sibling Lambdas deployed)

```bash
python3 -c "
from src.handler import lambda_handler
import json
print(lambda_handler(json.load(open('events/sample_event_direct.json')), None))
"
```

This makes real calls — Bedrock, plus all three sibling Lambdas via
`lambda:InvokeFunction` — so it needs valid credentials with every
permission listed in `iam/execution-role-policy.json`, and all three
other components already deployed and working.

## ⚠️ Before you deploy

Same inference-profile situation as `outfit-suggestion-agent`: `MODEL_ID`
is pinned to `us.anthropic.claude-haiku-4-5-20251001-v1:0`. If this
doesn't match what's actually available in your account/region, you'll
see the same `ValidationException` that repo's README documents — the fix
has two parts (the model ID in `handler.py`, and the matching ARNs in
`iam/execution-role-policy.json`), not just one.

## AWS setup: automatic, including the role

Standard create-or-update pattern (no legacy-role wrinkle here — this is
a brand-new function and role, nothing to correct):

1. Creates `weather-orchestrator-agent-role` if missing.
2. Creates or updates `weather-orchestrator-agent` (code + the
   `ORCHESTRATOR_INSTRUCTIONS` env var, built the same comma/quote/newline
   -safe way as `outfit-suggestion-agent`'s deploy script).
3. Smoke-tests it live with a question designed to exercise the full
   tool-use path (current weather + outfit suggestion both relevant).

`--timeout 60` and `--memory-size 256` are higher than the other three
Lambdas' — this one can make up to `MAX_ITERATIONS` sequential Bedrock
calls plus nested Lambda invocations (one of which, `outfit-suggestion-agent`,
makes its own Bedrock call), so it needs real headroom.

## GitHub Actions setup

### Required repo secret

| Secret | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | ARN of an IAM role GitHub assumes via OIDC |

If reusing the same deploy role across all four repos (as discussed
earlier in this project), it needs its permissions policy extended with a
**new** statement for this function's resources — the existing statements
for the other three repos' functions don't cover this one:

```json
{
  "Sid": "OrchestratorRoleBootstrap",
  "Effect": "Allow",
  "Action": ["iam:GetRole", "iam:CreateRole", "iam:PutRolePolicy"],
  "Resource": "arn:aws:iam::<ACCOUNT_ID>:role/weather-orchestrator-agent-role"
},
{
  "Sid": "OrchestratorPassRole",
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": "arn:aws:iam::<ACCOUNT_ID>:role/weather-orchestrator-agent-role",
  "Condition": { "StringEquals": { "iam:PassedToService": "lambda.amazonaws.com" } }
},
{
  "Sid": "OrchestratorLambdaDeployAndInvoke",
  "Effect": "Allow",
  "Action": [
    "lambda:GetFunction",
    "lambda:CreateFunction",
    "lambda:UpdateFunctionCode",
    "lambda:UpdateFunctionConfiguration",
    "lambda:InvokeFunction"
  ],
  "Resource": "arn:aws:lambda:<REGION>:<ACCOUNT_ID>:function:weather-orchestrator-agent"
}
```

## The whole architecture, end to end

With this component deployed, Style 2 is complete:

```
You --(aws lambda invoke / Function URL)--> weather-orchestrator-agent
                                                   |
                      +----------------------------+----------------------------+
                      |                             |                            |
            current-weather-lambda       forecast-weather-lambda       outfit-suggestion-agent
            (deterministic, Open-Meteo)  (deterministic, Open-Meteo)   (reasoning, Bedrock Converse)
```

Four repos, four independent CI/CD pipelines, one real agentic loop at
the center deciding which of the other three to call and when.
