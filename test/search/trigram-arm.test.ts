/**
 * Opt-in trigram recall arm (`search.trigram_arm`).
 *
 * The gap: the keyword and title arms both match whole FTS lexemes. Under an
 * FTS config that cannot segment or stem CJK ('english' over Korean), a chunk
 * that only ever writes a name with a particle attached ("인터엠디는") or
 * fused into a compound ("카카오헬스케어") holds a lexeme the base-form query
 * token never equals — and the D2 AND→OR relaxation cannot help, because it
 * relaxes at that same lexeme grain. On PGLite the CJK ILIKE fallback has the
 * same blind spot from the other side: it matches the query as one literal
 * substring, so a multi-token query never matches at all.
 *
 * Under test:
 *   1. extractTrigramTokens — the pure query-side tokenizer (operand list).
 *   2. engine.searchTrigram — reaches the surface form searchKeyword misses,
 *      with the gap reproduced in the same test rather than assumed.
 *   3. The knob gate — off (every bundle default) contributes nothing through
 *      hybridSearch; on fuses the arm's candidates into the blend.
 *
 * Hermetic PGLite. The gateway is pinned with an EMPTY env so embedding is
 * deterministically unavailable — hybridSearch takes the keyword(+title
 * +trigram) no-embed path with zero network, regardless of host API keys.
 *
 * Fixture text is invented (a fictional vendor log); no real corpus content.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from 'bun:test';
import { PGLiteEngine } from '../../src/core/pglite-engine.ts';
import { resetPgliteState } from '../helpers/reset-pglite.ts';
import { hybridSearch } from '../../src/core/search/hybrid.ts';
import { configureGateway } from '../../src/core/ai/gateway.ts';
import {
  extractTrigramTokens,
  TRIGRAM_MAX_TOKENS,
} from '../../src/core/search/trigram.ts';

let engine: PGLiteEngine;

const DIM = 1536;

beforeAll(async () => {
  // Pin 1536-d (matches the preload schema default) with an EMPTY env so
  // isAvailable('embedding') is false → hybridSearch never embeds.
  configureGateway({
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: DIM,
    env: {},
  });
  engine = new PGLiteEngine();
  await engine.connect({}); // in-memory
  await engine.initSchema();
});

afterAll(async () => {
  await engine.disconnect();
  // Restore the preload-equivalent gateway for sibling files in this shard.
  configureGateway({
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: DIM,
    env: { ...process.env },
  });
});

beforeEach(async () => {
  await resetPgliteState(engine);
});

/**
 * A page whose body mentions two invented vendors ONLY in inflected /
 * compounded form, and whose TITLE shares no token with the query — so the
 * title arm cannot rescue it either. This is the exact shape the arm exists
 * for.
 */
const VENDOR_BODY =
  '인터엠디는 좋은 회사다. 카카오헬스케어와 협업 방안을 논의했다.';

async function seedVendorLog(): Promise<void> {
  await engine.putPage('notes/vendor-log', {
    type: 'note',
    title: 'Quarterly Vendor Log',
    compiled_truth: VENDOR_BODY,
  });
  await engine.upsertChunks('notes/vendor-log', [
    {
      chunk_index: 0,
      chunk_text: VENDOR_BODY,
      chunk_source: 'compiled_truth',
    },
  ]);
}

describe('extractTrigramTokens — pure', () => {
  test('a 2-character Hangul token survives the CJK floor', () => {
    expect(extractTrigramTokens('회의')).toEqual(['회의']);
  });

  test('a 1-character Hangul token is below even the CJK floor', () => {
    expect(extractTrigramTokens('회의 록')).toEqual(['회의']);
  });

  test('a 3-character ASCII token is dropped; a 4-character one is kept', () => {
    // 'the' is exactly the class the higher Latin floor exists to exclude:
    // word_similarity has no IDF, so a stopword operand would match a large
    // fraction of any English corpus at the default threshold.
    expect(extractTrigramTokens('the notion')).toEqual(['notion']);
    expect(extractTrigramTokens('the sync')).toEqual(['sync']);
  });

  test('wrapping quotes, backticks, brackets and trailing punctuation are trimmed', () => {
    expect(extractTrigramTokens('"카카오헬스케어" (인터엠디),')).toEqual([
      '카카오헬스케어',
      '인터엠디',
    ]);
    expect(extractTrigramTokens('`notion` [webhook]?')).toEqual([
      'webhook',
      'notion',
    ]);
  });

  test('interior punctuation is preserved — only the token edges are trimmed', () => {
    expect(extractTrigramTokens('notion-sync.')).toEqual(['notion-sync']);
  });

  test('duplicates collapse, case-insensitively (pg_trgm lowercases both sides)', () => {
    expect(extractTrigramTokens('notion notion Notion')).toEqual(['notion']);
  });

  test('tokens come back longest-first and capped at TRIGRAM_MAX_TOKENS', () => {
    const query = 'aaaaaaaaaa bbbbbbbbb cccccccc ddddddd eeeeee fffff gggg hhhhhhhhhhh iiiiiiiiiiii';
    const tokens = extractTrigramTokens(query);
    expect(tokens.length).toBe(TRIGRAM_MAX_TOKENS);
    const lengths = tokens.map((t) => t.length);
    expect([...lengths].sort((a, b) => b - a)).toEqual(lengths);
    // The two longest tokens sit last in the query — proof the cap keeps the
    // most specific operands rather than the first eight seen.
    expect(tokens[0]).toBe('iiiiiiiiiiii');
    expect(tokens[1]).toBe('hhhhhhhhhhh');
  });

  test('a query with no qualifying token yields an empty operand list', () => {
    expect(extractTrigramTokens('a of to the')).toEqual([]);
    expect(extractTrigramTokens('   ')).toEqual([]);
    expect(extractTrigramTokens('')).toEqual([]);
  });
});

describe('searchTrigram — PGLite engine arm', () => {
  test('reaches a particle-suffixed mention that searchKeyword cannot (gap reproduced)', async () => {
    await seedVendorLog();

    // The gap, asserted rather than assumed: the body writes "인터엠디는",
    // never the bare base form, and the query carries a second token — so the
    // lexeme-grain arm (and PGLite's whole-query CJK ILIKE fallback) miss it.
    const keyword = await engine.searchKeyword('인터엠디 미팅', { orFallback: true });
    expect(keyword.map((r) => r.slug)).not.toContain('notes/vendor-log');

    const trigram = await engine.searchTrigram('인터엠디 미팅');
    expect(trigram.map((r) => r.slug)).toContain('notes/vendor-log');
  });

  test('reaches a compound-only mention (카카오 → 카카오헬스케어)', async () => {
    await seedVendorLog();

    const keyword = await engine.searchKeyword('카카오 제휴', { orFallback: true });
    expect(keyword.map((r) => r.slug)).not.toContain('notes/vendor-log');

    const trigram = await engine.searchTrigram('카카오 제휴');
    expect(trigram.map((r) => r.slug)).toContain('notes/vendor-log');
  });

  test('an unrelated query still matches nothing (the arm is not a wildcard)', async () => {
    await seedVendorLog();
    const trigram = await engine.searchTrigram('블록체인 발행량');
    expect(trigram.map((r) => r.slug)).not.toContain('notes/vendor-log');
  });

  test('a query with no qualifying token returns [] without querying', async () => {
    await seedVendorLog();
    expect(await engine.searchTrigram('a of to')).toEqual([]);
    expect(await engine.searchTrigram('')).toEqual([]);
  });

  test('honors the page-grain filters — type filter narrows the arm', async () => {
    await seedVendorLog();
    expect((await engine.searchTrigram('인터엠디 미팅', { type: 'note' })).length).toBeGreaterThan(0);
    expect(await engine.searchTrigram('인터엠디 미팅', { type: 'person' })).toEqual([]);
  });

  test('returns one row per page (best-per-page dedup) across many matching chunks', async () => {
    await engine.putPage('notes/vendor-log', {
      type: 'note',
      title: 'Quarterly Vendor Log',
      compiled_truth: VENDOR_BODY,
    });
    await engine.upsertChunks('notes/vendor-log', [
      { chunk_index: 0, chunk_text: '인터엠디는 좋은 회사다.', chunk_source: 'compiled_truth' },
      { chunk_index: 1, chunk_text: '인터엠디와 계약을 갱신했다.', chunk_source: 'timeline' },
      { chunk_index: 2, chunk_text: '인터엠디에서 연락이 왔다.', chunk_source: 'timeline' },
    ]);
    const trigram = await engine.searchTrigram('인터엠디 미팅');
    expect(trigram.filter((r) => r.slug === 'notes/vendor-log').length).toBe(1);
  });
});

describe('hybridSearch — search.trigram_arm knob gate', () => {
  test('bundle default (off): the trigram arm contributes nothing', async () => {
    await seedVendorLog();
    const results = await hybridSearch(engine, '인터엠디 미팅', { limit: 10 });
    expect(results.map((r) => r.slug)).not.toContain('notes/vendor-log');
  });

  test('search.trigram_arm=true fuses the arm end-to-end', async () => {
    await seedVendorLog();
    await engine.setConfig('search.trigram_arm', 'true');
    const results = await hybridSearch(engine, '인터엠디 미팅', { limit: 10 });
    expect(results.map((r) => r.slug)).toContain('notes/vendor-log');
  });

  test('per-call trigram_arm:true wins over the (off) bundle default', async () => {
    await seedVendorLog();
    const results = await hybridSearch(engine, '인터엠디 미팅', { limit: 10, trigram_arm: true });
    expect(results.map((r) => r.slug)).toContain('notes/vendor-log');
  });

  test('per-call trigram_arm:false wins over config=true', async () => {
    await seedVendorLog();
    await engine.setConfig('search.trigram_arm', 'true');
    const results = await hybridSearch(engine, '인터엠디 미팅', { limit: 10, trigram_arm: false });
    expect(results.map((r) => r.slug)).not.toContain('notes/vendor-log');
  });
});
