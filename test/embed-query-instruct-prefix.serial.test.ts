/**
 * Fork (qwen3-mrl-support) — Qwen3 query-side instruction prefix.
 *
 * Qwen3-Embedding is trained with an `Instruct: {task}\nQuery:{query}`
 * template on the QUERY side only; documents are embedded raw. These tests
 * pin the gateway's applyQueryInstruct seam at the only observable layer —
 * the outbound HTTP body's `input` array (same real-SDK + mocked-fetch
 * harness as embed-input-type-wire.serial.test.ts):
 *
 *   - embedQuery against a qwen3-embedding model prepends the template;
 *     the existing Matryoshka `dimensions` behavior is untouched.
 *   - embed() (document/index side) against the same model stays raw.
 *   - Non-qwen3 models never see the template, query side included.
 *   - GBRAIN_QUERY_INSTRUCT overrides the task sentence; empty string
 *     disables the prefix entirely (kill switch).
 *   - The prefix is applied BEFORE the MAX_CHARS cap, so the template
 *     survives truncation of an oversized query.
 *
 * Serial: mocks globalThis.fetch and mutates process.env.
 */

import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { configureGateway, embed, embedQuery, resetGateway } from '../src/core/ai/gateway.ts';

const KO_DEFAULT_TASK = '주어진 질문에 답이 되는 사내 문서를 검색하라';

type FetchHandler = (url: string, init: RequestInit) => Promise<Response>;
let fetchHandler: FetchHandler | null = null;
const origFetch = globalThis.fetch;
const origInstructEnv = process.env.GBRAIN_QUERY_INSTRUCT;

beforeEach(() => {
  fetchHandler = null;
  delete process.env.GBRAIN_QUERY_INSTRUCT;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    if (!fetchHandler) {
      throw new Error('fetch called but no handler installed');
    }
    return fetchHandler(typeof url === 'string' ? url : url.toString(), init ?? {});
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = origFetch;
  if (origInstructEnv === undefined) {
    delete process.env.GBRAIN_QUERY_INSTRUCT;
  } else {
    process.env.GBRAIN_QUERY_INSTRUCT = origInstructEnv;
  }
  resetGateway();
});

/** OpenAI-shaped /v1/embeddings response (Ollama's compat endpoint shape). */
function openAIShapedResponse(dims: number, count: number): Response {
  const vec = Array.from({ length: dims }, () => 0.1);
  return new Response(
    JSON.stringify({
      data: Array.from({ length: count }, (_, i) => ({ object: 'embedding', index: i, embedding: vec })),
      usage: { prompt_tokens: 3, total_tokens: 3 },
    }),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  );
}

function configureOllamaQwen3() {
  configureGateway({
    embedding_model: 'ollama:qwen3-embedding:8b-fp16',
    embedding_dimensions: 1536,
    env: {},
  });
}

function captureBody(dims: number) {
  let captured: any = null;
  fetchHandler = async (_url, init) => {
    captured = JSON.parse(init.body as string);
    return openAIShapedResponse(dims, Array.isArray(captured.input) ? captured.input.length : 1);
  };
  return () => captured;
}

describe('qwen3-embedding — query side gets the Instruct template', () => {
  test('embedQuery prepends the default Korean task sentence', async () => {
    configureOllamaQwen3();
    const body = captureBody(1536);

    await embedQuery('조달청 쇼핑몰에 제품 등록하려면?');
    expect(body().input[0]).toBe(`Instruct: ${KO_DEFAULT_TASK}\nQuery:조달청 쇼핑몰에 제품 등록하려면?`);
    // Matryoshka dim threading (the branch we share the model match with)
    // must be unaffected by the text rewrite.
    expect(body().dimensions).toBe(1536);
  });

  test('embed (document/index side) stays raw for the same model', async () => {
    configureOllamaQwen3();
    const body = captureBody(1536);

    await embed(['이 문서는 색인용 본문이다']);
    expect(body().input[0]).toBe('이 문서는 색인용 본문이다');
  });

  test('bare model id (no :tag) also matches', async () => {
    configureGateway({
      embedding_model: 'ollama:qwen3-embedding',
      embedding_dimensions: 1536,
      env: {},
    });
    const body = captureBody(1536);

    await embedQuery('hello');
    expect(body().input[0]).toBe(`Instruct: ${KO_DEFAULT_TASK}\nQuery:hello`);
  });
});

describe('scoping — non-qwen3 models never see the template', () => {
  test('asymmetric non-qwen3 model (zembed-1 on ollama): query text stays raw', async () => {
    configureGateway({
      embedding_model: 'ollama:zembed-1',
      embedding_dimensions: 1280,
      env: {},
    });
    const body = captureBody(1280);

    await embedQuery('what does foo bar do?');
    expect(body().input[0]).toBe('what does foo bar do?');
    // The wire-level asymmetry signal for such models is untouched.
    expect(body().input_type).toBe('query');
  });

  test('qwen3-lookalike prefix without the family separator does not match', async () => {
    configureGateway({
      embedding_model: 'ollama:qwen3-embedding-custom',
      embedding_dimensions: 768,
      env: {},
    });
    const body = captureBody(768);

    await embedQuery('hello');
    expect(body().input[0]).toBe('hello');
  });
});

describe('GBRAIN_QUERY_INSTRUCT override', () => {
  test('custom task sentence replaces the default', async () => {
    process.env.GBRAIN_QUERY_INSTRUCT =
      'Given a web search query, retrieve relevant passages that answer the query';
    configureOllamaQwen3();
    const body = captureBody(1536);

    await embedQuery('hello');
    expect(body().input[0]).toBe(
      'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:hello',
    );
  });

  test('empty string disables the prefix (kill switch)', async () => {
    process.env.GBRAIN_QUERY_INSTRUCT = '';
    configureOllamaQwen3();
    const body = captureBody(1536);

    await embedQuery('hello');
    expect(body().input[0]).toBe('hello');
  });
});

describe('interaction with the MAX_CHARS cap', () => {
  test('template survives truncation of an oversized query', async () => {
    configureOllamaQwen3();
    const body = captureBody(1536);

    await embedQuery('가'.repeat(9000));
    const sent: string = body().input[0];
    expect(sent.length).toBe(8000);
    expect(sent.startsWith(`Instruct: ${KO_DEFAULT_TASK}\nQuery:`)).toBe(true);
  });
});
