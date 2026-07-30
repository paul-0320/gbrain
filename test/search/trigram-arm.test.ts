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
  dropFloodTokens,
  buildTrigramDfLikePattern,
  parseTrigramDfRows,
  TRIGRAM_MAX_TOKENS,
  TRIGRAM_DF_MIN_COUNT,
  TRIGRAM_DF_RATIO,
  TRIGRAM_DF_PROBE_SQL,
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

// ───────────────────────────────────────────────────────────────────────────
// Corpus-frequency gate (the no-IDF flood guard).
//
// The arm's score is a SUM of per-token word_similarity with no rarity
// weighting, so a corpus-common noun scores ~1.0 against every chunk that
// contains it. On the production corpus that cost a 62-question regression
// gate 12 answers (57 → 45, 0 rescued): "자동" occurs in 23.9% of text chunks,
// so dozens of unrelated pages tied at the top of the arm's list and filled
// the reranker's input window. dropFloodTokens removes those operands.
// ───────────────────────────────────────────────────────────────────────────

describe('dropFloodTokens — pure', () => {
  test('ratio gate bites once the corpus is large enough to exceed the floor', () => {
    // total 10,000 → threshold = max(50, 300) = 300.
    const counts = { flood: 301, rare: 300 };
    expect(dropFloodTokens(['flood', 'rare'], counts, 10_000)).toEqual(['rare']);
  });

  test('threshold is strictly-greater — a token exactly on the line survives', () => {
    expect(dropFloodTokens(['edge'], { edge: 300 }, 10_000)).toEqual(['edge']);
    expect(dropFloodTokens(['edge'], { edge: 301 }, 10_000)).toEqual([]);
  });

  test('the absolute floor dominates on a small corpus (no degenerate gating)', () => {
    // total 100 → ratio ceiling is 3, but TRIGRAM_DF_MIN_COUNT holds the line
    // at 50. A token in 40 of 100 chunks — 40% of the corpus — still survives,
    // because 100 chunks cannot flood a 25-candidate rerank window.
    expect(dropFloodTokens(['common'], { common: 40 }, 100)).toEqual(['common']);
    expect(dropFloodTokens(['common'], { common: 51 }, 100)).toEqual([]);
  });

  test('the ratio only overtakes the floor above ~1,667 text chunks', () => {
    const justUnder = Math.floor(TRIGRAM_DF_MIN_COUNT / TRIGRAM_DF_RATIO) - 100; // ratio ceiling < 50
    expect(dropFloodTokens(['t'], { t: TRIGRAM_DF_MIN_COUNT }, justUnder)).toEqual(['t']);
    const wellOver = 100_000; // ratio ceiling = 3000
    expect(dropFloodTokens(['t'], { t: 2_999 }, wellOver)).toEqual(['t']);
    expect(dropFloodTokens(['t'], { t: 3_001 }, wellOver)).toEqual([]);
  });

  test('every operand flooded → [] (a normal outcome, not an error)', () => {
    expect(dropFloodTokens(['자동', '실행'], { 자동: 5_340, 실행: 3_218 }, 22_351)).toEqual([]);
  });

  test('production calibration: the flood terms go, the arm’s targets stay', () => {
    // Measured shares on the 22,351-chunk corpus that produced the 57→45 loss.
    // At 3% the ceiling is 670 chunks, so the cut also takes 번호 (4.7%) and
    // 정부 (3.7%) — mid-frequency nouns that sit close to the line. Pinned
    // here deliberately: this is the collateral cost of a binary gate, and if
    // the constant is ever retuned this expectation should change with it.
    const total = 22_351;
    const pct = (p: number) => Math.round((p / 100) * total);
    const counts: Record<string, number> = {
      자동: pct(23.9), 실행: pct(14.4), 문제: pct(14.3), 코드: pct(12.9), 모델: pct(7.4),
      번호: pct(4.7), 정부: pct(3.7), 청크: pct(2.6), 주말: pct(0.8), 리랭커: pct(0.5), 인터엠디: pct(0.4),
    };
    const tokens = Object.keys(counts);
    expect(dropFloodTokens(tokens, counts, total)).toEqual([
      '청크', '주말', '리랭커', '인터엠디',
    ]);
    // The load-bearing part: every token that caused a lost answer is gone,
    // and the entity name the arm exists to reach is untouched.
    expect(dropFloodTokens(['자동', '실행', '인터엠디'], counts, total)).toEqual(['인터엠디']);
  });

  test('accepts a Map as well as a plain record', () => {
    const counts = new Map<string, number>([['flood', 400], ['rare', 1]]);
    expect(dropFloodTokens(['flood', 'rare'], counts, 10_000)).toEqual(['rare']);
  });

  test('an unmeasured token is kept — a partial probe must not delete operands', () => {
    expect(dropFloodTokens(['measured', 'missing'], { measured: 400 }, 10_000)).toEqual(['missing']);
  });

  test('preserves the longest-first order it was handed', () => {
    const counts = { aaaa: 1, bbb: 1, cc: 1 };
    expect(dropFloodTokens(['aaaa', 'bbb', 'cc'], counts, 10_000)).toEqual(['aaaa', 'bbb', 'cc']);
    expect(dropFloodTokens([], counts, 10_000)).toEqual([]);
  });
});

describe('buildTrigramDfLikePattern — pure', () => {
  test('wraps the token in containment wildcards', () => {
    expect(buildTrigramDfLikePattern('인터엠디')).toBe('%인터엠디%');
  });

  test('escapes LIKE metacharacters so a token cannot gate itself away', () => {
    // Unescaped, '%' inside a token would match every chunk and the token
    // would always look like a flood term.
    expect(buildTrigramDfLikePattern('50%')).toBe('%50\\%%');
    expect(buildTrigramDfLikePattern('a_b')).toBe('%a\\_b%');
    expect(buildTrigramDfLikePattern('c:\\tmp')).toBe('%c:\\\\tmp%');
  });
});

describe('TRIGRAM_DF_PROBE_SQL — SQL-side behavior', () => {
  test('counts literal containment, honoring ESCAPE (a_b does not match axb)', async () => {
    await engine.putPage('notes/escape-probe', {
      type: 'note',
      title: 'Escape Probe',
      compiled_truth: 'literal a_b marker',
    });
    await engine.upsertChunks('notes/escape-probe', [
      { chunk_index: 0, chunk_text: 'literal a_b marker', chunk_source: 'compiled_truth' },
      { chunk_index: 1, chunk_text: 'literal axb marker', chunk_source: 'timeline' },
    ]);

    const token = 'a_b';
    const rows = await engine.executeRaw<Record<string, unknown>>(
      TRIGRAM_DF_PROBE_SQL,
      [[token], [buildTrigramDfLikePattern(token)]],
    );
    const { totalTextChunks, counts } = parseTrigramDfRows(rows);
    expect(totalTextChunks).toBe(2);
    // 1, not 2 — the underscore is a literal here, not a LIKE single-char wildcard.
    expect(counts.get(token)).toBe(1);
  });
});

/** 60 chunks, every one containing 자동 and 실행 — enough to clear the 50-chunk floor. */
async function seedFloodCorpus(): Promise<void> {
  await engine.putPage('notes/bulk-log', {
    type: 'note',
    title: 'Bulk Batch Log',
    compiled_truth: '배치 작업 로그 모음',
  });
  await engine.upsertChunks(
    'notes/bulk-log',
    Array.from({ length: 60 }, (_, i) => ({
      chunk_index: i,
      chunk_text: `${i}번째 배치에서 자동 실행 작업이 정상 종료되었다.`,
      chunk_source: (i === 0 ? 'compiled_truth' : 'timeline') as 'compiled_truth' | 'timeline',
    })),
  );
}

describe('searchTrigram — corpus-frequency gate', () => {
  test('an all-flood query yields an empty arm (both operands gated away)', async () => {
    await seedFloodCorpus();
    await seedVendorLog(); // 61 text chunks total → threshold = max(50, 1.83) = 50

    // Sanity: without the gate these operands WOULD match — every one of the
    // 60 chunks contains both, at word_similarity ~1.0.
    const df = await engine.executeRaw<Record<string, unknown>>(
      TRIGRAM_DF_PROBE_SQL,
      [['자동', '실행'], ['자동', '실행'].map(buildTrigramDfLikePattern)],
    );
    const parsed = parseTrigramDfRows(df);
    expect(parsed.counts.get('자동')).toBe(60);
    expect(parsed.totalTextChunks).toBe(61);

    expect(await engine.searchTrigram('자동 실행')).toEqual([]);
  });

  test('a mixed query keeps the rare operand and still reaches its page', async () => {
    await seedFloodCorpus();
    await seedVendorLog();

    const hits = await engine.searchTrigram('자동 인터엠디');
    const slugs = hits.map((r) => r.slug);
    // The rare operand survives and does its job …
    expect(slugs).toContain('notes/vendor-log');
    // … while the flood operand no longer drags in the 60-chunk bulk page.
    expect(slugs).not.toContain('notes/bulk-log');
  });

  test('the absolute floor keeps a small corpus ungated even at 100% frequency', async () => {
    // 20 chunks, all containing 주말 — 100% of the corpus, but 20 <= 50, so the
    // ratio gate never engages and the arm behaves exactly as before.
    await engine.putPage('notes/weekend-log', {
      type: 'note',
      title: 'Weekend Log',
      compiled_truth: '주말 점검 기록',
    });
    await engine.upsertChunks(
      'notes/weekend-log',
      Array.from({ length: 20 }, (_, i) => ({
        chunk_index: i,
        chunk_text: `${i}주차 주말에는 배치를 돌리지 않기로 했다.`,
        chunk_source: (i === 0 ? 'compiled_truth' : 'timeline') as 'compiled_truth' | 'timeline',
      })),
    );

    const hits = await engine.searchTrigram('주말 점검');
    expect(hits.map((r) => r.slug)).toContain('notes/weekend-log');
  });
});

describe('hybridSearch — the gate makes an all-flood query a no-op', () => {
  test('arm on and arm off return the same results when every operand floods', async () => {
    await seedFloodCorpus();
    await seedVendorLog();

    const off = await hybridSearch(engine, '자동 실행', { limit: 10 });
    const on = await hybridSearch(engine, '자동 실행', { limit: 10, trigram_arm: true });
    expect(on.map((r) => r.slug)).toEqual(off.map((r) => r.slug));
  });

  test('a mixed query still gains the rare operand’s reach with the arm on', async () => {
    await seedFloodCorpus();
    await seedVendorLog();

    const off = await hybridSearch(engine, '자동 인터엠디', { limit: 10 });
    const on = await hybridSearch(engine, '자동 인터엠디', { limit: 10, trigram_arm: true });
    expect(off.map((r) => r.slug)).not.toContain('notes/vendor-log');
    expect(on.map((r) => r.slug)).toContain('notes/vendor-log');
  });
});
