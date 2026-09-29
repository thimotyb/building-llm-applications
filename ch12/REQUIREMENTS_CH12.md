# Chapter 12 Requirements — Multi-Provider LLM Applications with LiteLLM

Status: working plan / scratchpad

Target chapter: `ch12`

Prerequisite: the complete multi-provider application developed through `ch11`

## 1. Purpose

Chapter 12 will address the complexity that emerges when an application must
support several LLM providers directly. Chapters 4 through 11 progressively
introduce provider selection, RAG, tool calling, routing, guardrails, memory,
and MCP integration. By chapter 11, the application supports OpenAI, Ollama,
Gemini, and DeepSeek, but it also contains provider-specific constructors,
parameters, compatibility adaptations, and validation branches.

The chapter will introduce LiteLLM as an abstraction and gateway layer that
provides a common OpenAI-compatible interface across providers. The goal is to
reduce provider-specific code in the application without pretending that all
models have identical capabilities.

The central teaching message will be:

> A unified API removes most provider-specific plumbing, but it does not make
> model capabilities or behavior identical.

## 2. Learning objectives

After completing the chapter, students should be able to:

1. Explain why a growing set of provider branches becomes difficult to
   maintain.
2. Distinguish API portability from functional portability.
3. Call several providers through the LiteLLM Python SDK using one request and
   response format.
4. Configure and run the LiteLLM Proxy as a local LLM gateway.
5. Connect LangChain and LangGraph to the gateway through one
   OpenAI-compatible client.
6. Keep chat-model selection independent from embedding-model selection.
7. Define a minimum capability contract for all supported chat models.
8. Test text generation, structured output, tool calling, and RAG across
   providers without comparing exact natural-language responses.
9. Configure controlled retries and fallbacks.
10. Recognize the model-specific behavior that still requires explicit policy
    or validation.

## 3. Scope

### 3.1 Providers

The first implementation should cover the providers already introduced in the
course:

- OpenAI;
- Gemini;
- Ollama;
- DeepSeek.

The architecture should allow another LiteLLM-supported provider to be added
through configuration, without adding a new branch to the application layer.

### 3.2 Features included

- LiteLLM Python SDK fundamentals;
- LiteLLM Proxy running locally;
- provider aliases in a LiteLLM YAML configuration;
- one LangChain chat-model construction path;
- one LangChain embedding-model construction path;
- plain text responses;
- structured output through a common-denominator strategy;
- tool calling;
- streaming, if it works consistently with the selected model versions;
- retries and explicit fallbacks;
- normalized error handling;
- basic latency, token-usage, and cost reporting;
- reuse of the precomputed Gemini Chroma vector store;
- automated provider contract tests.

### 3.3 Non-goals for the first version

- production deployment of a shared, multi-tenant LiteLLM service;
- enterprise authentication, billing, or administrative dashboards;
- replacing LangSmith with LiteLLM observability;
- automatic selection of the “best” model based on output quality;
- benchmarking every model supported by LiteLLM;
- forcing unsupported capabilities to appear equivalent;
- rewriting chapters 4–11 to depend on LiteLLM.

The earlier chapters must remain direct examples of provider-specific
integration. Chapter 12 should build on that experience and demonstrate the
architectural improvement.

## 4. Current problem to expose

The chapter should begin with a short review of the chapter 11 implementation.
The current factory must:

- import multiple provider-specific LangChain packages;
- select a different class for each provider;
- map different API key names and model names;
- handle provider-specific base URLs;
- handle differences in structured output;
- suppress or translate parameters that are not supported everywhere;
- select a separate embedding provider when the chat provider has no
  embeddings API;
- maintain provider-specific vector-store paths.

This implementation is valuable because it makes the differences visible, but
it becomes expensive to extend and test as more providers are added.

## 5. Proposed chapter progression

### 5.1 Section 12.1 — The multi-provider maintenance problem

Use the chapter 11 factory as the baseline.

Tasks:

- identify provider-specific branches;
- classify differences as credentials, transport, parameters, capabilities,
  or response behavior;
- distinguish chat providers from embedding providers;
- explain why adding another provider affects code, configuration, and tests.

Expected outcome: students understand the problem before seeing the
abstraction.

### 5.2 Section 12.2 — Unified calls with the LiteLLM SDK

Introduce a minimal `litellm.completion()` example that can call all four
providers by changing only the model identifier.

Example model identifiers should follow the LiteLLM provider-prefix format,
subject to verification against the installed LiteLLM version:

```text
openai/<openai-model>
gemini/<gemini-model>
ollama/<ollama-model>
deepseek/<deepseek-model>
```

The example must demonstrate:

- one messages format;
- one response extraction path;
- normalized usage metadata;
- normalized exception handling;
- provider selection through configuration rather than Python conditionals.

The SDK example is instructional. It is not yet the final integration with the
LangGraph application.

### 5.3 Section 12.3 — Capability contract

Before introducing the gateway, define the minimum functionality that the
course application expects from a chat model.

Initial contract:

1. return a non-empty text response;
2. produce data conforming to a small schema;
3. emit a valid function/tool call;
4. accept a tool result and complete the conversation;
5. preserve message ordering in multi-turn conversations;
6. expose sufficient usage metadata for diagnostics.

Optional capabilities must be reported separately:

- native JSON Schema structured output;
- forced tool selection;
- parallel tool calls;
- reasoning content;
- Responses API support;
- multimodal inputs;
- streaming usage metadata.

The application should use a common-denominator implementation when a feature
is required across all providers. For structured output, function calling is
the initial preferred strategy because DeepSeek does not currently accept the
same `json_schema` request used by OpenAI.

Unsupported parameters should not be silently discarded by default. LiteLLM's
capability inspection should be used to make differences visible. A global
`drop_params=True` setting should be avoided in the teaching implementation
because it can hide a loss of behavior.

### 5.4 Section 12.4 — LiteLLM Proxy as an LLM gateway

Run LiteLLM Proxy locally and configure provider-specific deployments in one
YAML file.

Use explicit logical aliases for classroom diagnostics:

```text
course-openai
course-gemini
course-ollama
course-deepseek
course-gemini-embedding
```

The application should select an alias through environment variables:

```dotenv
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_API_KEY=<LOCAL_GATEWAY_KEY>
LITELLM_CHAT_MODEL=course-deepseek
LITELLM_EMBEDDING_MODEL=course-gemini-embedding
VECTORSTORE_ID=gemini
```

Provider API keys remain in the project-root `.env` file and are referenced by
the LiteLLM configuration. They must never be written directly into the YAML
file or committed.

The first gateway configuration should favor explicit selection rather than
automatic load balancing. Students should be able to tell which provider is
being called.

### 5.5 Section 12.5 — Migrate the final application

Create a self-contained chapter 12 version of the final chapter 11
application. Preserve its application behavior while replacing the
provider-specific model factory.

The chat factory should have one construction path, conceptually equivalent
to:

```python
ChatOpenAI(
    model=settings.litellm_chat_model,
    base_url=settings.litellm_base_url,
    api_key=settings.litellm_api_key,
)
```

The embeddings factory should also use the gateway's OpenAI-compatible
embeddings endpoint, while remaining logically independent from the chat
model.

The migrated application must retain:

- the chapter 11 tools;
- RAG over travel information;
- LangGraph routing;
- guardrails;
- memory/checkpoint behavior;
- MCP integration where applicable;
- existing response-normalization helpers where LangChain can return content
  blocks rather than a plain string.

The chapter should show that application nodes no longer need to know whether
the selected model is OpenAI, Gemini, Ollama, or DeepSeek.

### 5.6 Section 12.6 — Embeddings and persistent vector stores

LiteLLM unifies access to embedding endpoints, but embeddings generated by
different models are not interchangeable.

Requirements:

- keep chat-model and embedding-model settings separate;
- keep the persistent-store identity separate from the chat provider;
- use Gemini embeddings by default for the chapter 12 example;
- reuse `ch11/vectorstore_db/gemini` or install the existing
  `vectorstore_gemini.zip` archive;
- do not rebuild embeddings when the compatible Gemini store is already
  available;
- fail clearly if the selected vector store was created with an incompatible
  embedding model.

`VECTORSTORE_ID=gemini` should make the reuse explicit even if the LiteLLM
embedding alias is named `course-gemini-embedding`.

### 5.7 Section 12.7 — Routing, retries, and fallbacks

After explicit provider selection works, introduce one controlled fallback
example.

Suggested scenario:

```text
primary chat model: DeepSeek
fallback chat model: Gemini
```

The example should explain:

- which errors trigger a retry;
- which errors trigger a fallback;
- which errors must be returned immediately;
- why a fallback can change cost, latency, output style, and capability;
- why a fallback must be visible in logs and result metadata.

Do not present fallback as a substitute for validation. The final response
should record the provider and deployment that actually handled the request.

### 5.8 Section 12.8 — Operational trade-offs

Conclude with the benefits and costs of the gateway architecture.

Benefits:

- less provider-specific application code;
- one request and response format;
- centralized credentials and endpoints;
- normalized exceptions;
- centralized retry and fallback policies;
- easier provider additions;
- consistent cost, usage, and latency observation;
- the option to move from an in-process SDK to a shared gateway later.

Trade-offs:

- another dependency and process to run;
- an additional failure point;
- possible version drift between LiteLLM and provider APIs;
- some provider-specific features may lag or require pass-through parameters;
- normalized syntax does not guarantee normalized quality;
- fallbacks can change application behavior;
- the gateway configuration becomes production-critical.

## 6. Recommended architecture

The recommended final architecture is:

```text
LangGraph application
        |
        | OpenAI-compatible chat and embedding requests
        v
LiteLLM Proxy
        |
        +--> OpenAI
        +--> Gemini
        +--> Ollama
        +--> DeepSeek

Chroma vector store
        |
        +--> identified by the embedding model, not the chat provider
```

The LiteLLM Python SDK should be taught first, but the Proxy should be the final
chapter architecture because it keeps provider routing outside the LangGraph
application and preserves the existing LangChain interfaces.

## 7. Proposed files

```text
ch12/
├── REQUIREMENTS_CH12.md
├── README.md
├── requirements.txt
├── .env_example
├── litellm_config.yaml
├── env_config.py
├── llm_gateway.py
├── main_01_sdk.py
├── main_02_gateway.py
├── main_03_travel_assistant.py
└── tests/
    ├── test_gateway_config.py
    ├── test_provider_contract.py
    ├── test_vectorstore_compatibility.py
    └── test_travel_assistant.py
```

Possible supporting scripts:

```text
ch12/scripts/start_litellm.sh
ch12/scripts/smoke_test_gateway.py
```

The final names should follow the naming style selected for chapter 12 and
should not force changes to the earlier chapter structure.

## 8. Configuration requirements

### 8.1 Project-root environment

Continue using the project-root `.env` as the source of secrets and local
configuration. Expected variables include:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=

GEMINI_API_KEY=
GEMINI_MODEL=
GEMINI_EMBEDDING_MODEL=

DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=
DEEPSEEK_THINKING=disabled

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=
OLLAMA_EMBEDDING_MODEL=

LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_API_KEY=
LITELLM_MASTER_KEY=
LITELLM_CHAT_MODEL=course-deepseek
LITELLM_EMBEDDING_MODEL=course-gemini-embedding
VECTORSTORE_ID=gemini
```

The exact authentication variables required by the local Proxy must be
validated against the pinned LiteLLM version before implementation.

### 8.2 LiteLLM YAML configuration

The YAML configuration must:

- reference secrets through environment variables;
- define one explicit alias per provider;
- define a Gemini embedding alias;
- keep provider-specific parameters out of application code;
- document any DeepSeek thinking-mode configuration;
- avoid automatic routing until explicit-provider tests pass;
- add fallback configuration only in the routing lesson;
- contain no real credentials.

## 9. Provider capability policy

The chapter must distinguish three categories:

### 9.1 Required common capabilities

- plain text chat;
- function/tool calling;
- schema-valid structured data through function calling;
- multi-turn tool-result handling.

### 9.2 Optional capabilities

- native JSON Schema response format;
- Responses API;
- reasoning output;
- parallel tool calling;
- multimodal input;
- forced tool choice;
- provider-side web search.

### 9.3 Provider-specific configuration

Provider-specific options may appear in `litellm_config.yaml`, but they should
not appear as branches in LangGraph nodes or business logic.

If a feature cannot be represented safely through the common contract, the
application must either:

1. disable it for the shared example;
2. expose it as an explicitly optional feature; or
3. fail with a clear capability error.

## 10. Test strategy

### 10.1 Static tests

- validate YAML syntax;
- validate required environment-variable names without printing secrets;
- compile all Python files;
- run `git diff --check`;
- check that no credential is present in tracked files;
- verify that application modules contain no provider-selection branches;
- verify that chat and embedding model identifiers are independent.

### 10.2 Offline unit tests

- mock the LiteLLM gateway responses;
- test normalized response extraction;
- test normalized exception handling;
- test provider and deployment metadata recording;
- test fallback metadata;
- test vector-store identity selection;
- test behavior when a capability is unavailable;
- test the application without requiring network access.

### 10.3 Live provider contract tests

Run the same tests for every configured and available provider:

1. text response is non-empty;
2. structured output validates against a Pydantic model;
3. a tool call contains the expected name and valid arguments;
4. a tool result can be returned to the model;
5. the final response is non-empty;
6. usage metadata is present when supported;
7. actual provider/deployment metadata is recorded;
8. errors use the expected normalized exception hierarchy.

Tests must not assert exact prose. They should assert structure, required
values, and behavior.

Providers with missing credentials or an unavailable local Ollama service
should be explicitly skipped with a readable reason, not reported as passed.

### 10.4 RAG compatibility test

- load the precomputed Gemini vector store;
- confirm that no embedding rebuild is triggered;
- run a similarity search;
- use the retrieved context with each available chat provider;
- validate that the final answer is grounded in the retrieved context;
- record which chat provider handled the request.

### 10.5 Routing and fallback test

- configure a controlled primary failure;
- verify that the configured fallback is used;
- verify that the final metadata identifies the fallback provider;
- verify that authentication errors are not accidentally hidden when policy
  requires them to fail immediately;
- verify retry limits and timeouts.

## 11. Acceptance criteria

Chapter 12 is complete when:

1. all four providers can be selected through configuration;
2. the application uses one chat-model construction path;
3. the application uses one embedding-model construction path;
4. no LangGraph node branches on provider name;
5. text, structured output, and tool-call contract tests pass for every
   provider that advertises the required capabilities;
6. incompatible optional capabilities are reported clearly;
7. the Gemini vector store is reused without recalculation;
8. the final travel assistant retains the relevant chapter 11 behavior;
9. one explicit fallback scenario is demonstrated and observable;
10. no secrets are committed;
11. setup and execution instructions are reproducible on Linux/WSL;
12. the chapter explains both the benefits and the limitations of LiteLLM.

## 12. Implementation sequence

Recommended order:

1. pin and install LiteLLM in an isolated `ch12` environment;
2. verify current model identifiers for all four providers;
3. implement and test the SDK-only example;
4. implement the provider capability probe;
5. write the first provider contract tests;
6. create the Proxy YAML configuration;
7. test each explicit gateway alias independently;
8. implement the single LangChain chat factory;
9. implement the single embeddings factory;
10. verify reuse of the Gemini vector store;
11. migrate the selected final chapter 11 application;
12. add routing and fallback only after direct aliases pass;
13. run offline and live tests;
14. document setup, limitations, and troubleshooting;
15. review the chapter against the progression from ch04 through ch11.

## 13. Decisions to make before implementation

- Select the exact chapter 11 application variant to migrate as the final
  chapter 12 example, likely the most complete routing/guardrail variant.
- Decide whether the Proxy will run through the LiteLLM CLI or Docker for the
  classroom default. The CLI is likely simpler for the first version.
- Select and pin a LiteLLM version after a four-provider proof of concept.
- Confirm the current LiteLLM model identifiers for the configured OpenAI,
  Gemini, Ollama, and DeepSeek models.
- Decide whether streaming is part of the required contract or an optional
  extension.
- Decide which failures are eligible for cross-provider fallback.
- Decide how provider and fallback metadata will be exposed to the user and
  logs.
- Decide whether the LiteLLM Proxy should be introduced as a separate terminal
  process or started by a repository helper script.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LiteLLM/provider API drift | Pin versions and keep live contract tests. |
| Silent parameter loss | Do not enable global `drop_params`; inspect capabilities. |
| Different structured-output support | Use function calling as the shared baseline. |
| Tool-call behavior differs | Test a complete tool round trip per provider. |
| DeepSeek reasoning affects tool history | Disable it in the common example or test it separately. |
| Wrong embedding store is loaded | Key persistence by `VECTORSTORE_ID` and embedding identity. |
| Fallback changes behavior invisibly | Return and log actual provider/deployment metadata. |
| Gateway becomes a hidden dependency | Include health checks and clear startup diagnostics. |
| Local Ollama is unavailable | Skip live Ollama tests explicitly and explain startup requirements. |
| Secrets leak into YAML or logs | Reference environment variables and run a secret scan. |

## 15. Source references

- LiteLLM overview and SDK/Proxy comparison:
  <https://docs.litellm.ai/>
- Completion parameters and provider capability inspection:
  <https://docs.litellm.ai/docs/completion/input>
- Embeddings:
  <https://docs.litellm.ai/docs/embedding/supported_embedding>
- Router, retries, load balancing, and fallback:
  <https://docs.litellm.ai/docs/routing>
- DeepSeek provider integration:
  <https://docs.litellm.ai/docs/providers/deepseek>
- Ollama provider integration:
  <https://docs.litellm.ai/docs/providers/ollama>

These references must be rechecked when implementation begins because LiteLLM
and provider APIs evolve frequently.
