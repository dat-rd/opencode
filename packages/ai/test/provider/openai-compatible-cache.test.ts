import { describe, expect } from "bun:test"
import { Effect } from "effect"
import { LLM, Message, ToolResultPart, ToolDefinition } from "../../src/index.js"
import { compileRequest } from "../../src/route/client.js"
import * as OpenAICompatibleChat from "../../src/protocols/openai-compatible-chat.js"
import * as OpenRouter from "../../src/providers/openrouter.js"
import { it } from "../lib/effect.js"

const route = OpenAICompatibleChat.route.with({
  provider: "gateway",
  endpoint: { baseURL: "https://api.gateway.test/v1" },
})

// A gateway that speaks the OpenAI chat shape in front of Anthropic expects the cache marker
// where Anthropic's own translator looks for it. On a tool message that is the message itself;
// a marker inside the tool message's content array is not a position the translator reads, and
// the request is billed as uncached input from then on.
const anthropicGateway = route.model({ id: "claude-sonnet-5", compatibility: { cacheFormat: "anthropic" } })

const toolResult = () =>
  Message.tool(
    ToolResultPart.make({
      id: "call_1",
      name: "read",
      result: { type: "content", value: [{ type: "text", text: "file contents" }] },
    }),
  )

const compile = (model: typeof anthropicGateway) =>
  compileRequest(
    LLM.request({
      model,
      system: "You are concise.",
      messages: [
        Message.user("Read the file"),
        Message.make({
          id: "assistant_1",
          role: "assistant",
          content: [{ type: "tool-call", id: "call_1", name: "read", input: { path: "a.ts" } }],
        }),
        toolResult(),
      ],
      tools: [ToolDefinition.make({ name: "read", description: "Read a file", inputSchema: {} })],
    }),
  )

describe("openai-compatible chat cache markers", () => {
  it.effect("marks a tool message at the message level when the format is anthropic", () =>
    Effect.gen(function* () {
      const prepared = yield* compile(anthropicGateway)
      const messages = prepared.body.messages ?? []
      const tool = messages.find((message) => message.role === "tool")

      expect(tool).toBeDefined()
      // The marker has to sit on the tool message, not inside its content array.
      expect(tool).toMatchObject({ cache_control: { type: "ephemeral" } })
      expect(JSON.stringify(tool?.content)).not.toContain("cache_control")
    }),
  )

  it.effect("places no marker at all when no format is declared", () =>
    Effect.gen(function* () {
      const prepared = yield* compile(route.model({ id: "gpt-5" }))

      // OpenAI caches implicitly, so a chat request that declares no format carries no marker.
      expect(JSON.stringify(prepared.body)).not.toContain("cache_control")
    }),
  )

  it.effect("leaves a route that documents the content-part position alone", () =>
    Effect.gen(function* () {
      // OpenRouter documents the content-part position for its own contract, so the marker
      // stays there even when the model declares the anthropic format.
      const prepared = yield* compile(
        OpenRouter.route.model({ id: "anthropic/claude-sonnet-5", compatibility: { cacheFormat: "anthropic" } }),
      )
      const messages = prepared.body.messages ?? []
      const tool = messages.find((message) => message.role === "tool")

      expect(tool).toBeDefined()
      expect(tool).not.toHaveProperty("cache_control")
    }),
  )
})
