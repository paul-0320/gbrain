/**
 * Opt-in trigram recall arm — Postgres engine half.
 *
 * The PGLite half lives in test/search/trigram-arm.test.ts and runs
 * everywhere. This file exists because the two engines build the arm's SQL
 * independently, and the interesting parts are Postgres-only:
 *   - `$n <% cc.chunk_text` against a REAL pg_trgm (the PGLite build is a
 *     WASM port; a divergence here would ship silently),
 *   - the arm running inside `withScopedReadTransaction` +
 *     `SET LOCAL statement_timeout`,
 *   - the v126 GIN gin_trgm_ops index actually existing after migrations.
 *
 * The gap is reproduced in the same test rather than assumed: on Postgres the
 * keyword arm goes through `websearch_to_tsquery('english', …)`, which cannot
 * segment Korean — a body that only ever writes "인터엠디는" holds a lexeme
 * the base-form query token never equals, and the AND→OR relaxation cannot
 * help because it relaxes at that same lexeme grain.
 *
 * Gated by DATABASE_URL — skips entirely without a real Postgres.
 * Fixture text is invented (a fictional vendor log); no real corpus content.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test';
import { hasDatabase, setupDB, teardownDB, getEngine, getConn } from './helpers.ts';

const SKIP_PG = !hasDatabase();
const describePG = SKIP_PG ? describe.skip : describe;

const VENDOR_BODY =
  '인터엠디는 좋은 회사다. 카카오헬스케어와 협업 방안을 논의했다.';

describePG('searchTrigram — Postgres engine arm', () => {
  beforeAll(async () => {
    await setupDB();
    const engine = getEngine();
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
  });

  afterAll(async () => {
    await teardownDB();
  });

  test('v126 created the partial GIN trigram index', async () => {
    const rows = await getConn().unsafe(
      `SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_chunks_text_trgm'`,
    );
    expect(rows.length).toBe(1);
    expect(String(rows[0].indexdef)).toContain('gin_trgm_ops');
    expect(String(rows[0].indexdef)).toContain("modality = 'text'");
  });

  test('reaches a particle- AND compound-hidden pair in one chunk (gap reproduced)', async () => {
    const engine = getEngine();

    // Both base forms are hidden — "인터엠디는" carries a particle,
    // "카카오헬스케어와" is a compound — so neither is a standalone lexeme.
    // websearch_to_tsquery misses on strict AND and on the OR relaxation
    // alike: there is no lexeme for the relaxed query to land on either.
    const keyword = await engine.searchKeyword('인터엠디 카카오', { orFallback: true });
    expect(keyword.map((r) => r.slug)).not.toContain('notes/vendor-log');

    // The arm reaches it because both operands match the SAME chunk,
    // satisfying the co-occurrence floor.
    const trigram = await engine.searchTrigram('인터엠디 카카오');
    expect(trigram.map((r) => r.slug)).toContain('notes/vendor-log');
  });

  test('a single surviving operand silences the arm', async () => {
    const engine = getEngine();
    expect(await engine.searchTrigram('인터엠디')).toEqual([]);
    expect(await engine.searchTrigram('인터엠디 블록체인')).toEqual([]);
  });

  test('an unrelated query still matches nothing (the arm is not a wildcard)', async () => {
    const trigram = await getEngine().searchTrigram('블록체인 발행량');
    expect(trigram.map((r) => r.slug)).not.toContain('notes/vendor-log');
  });

  test('a query with no qualifying token returns [] without querying', async () => {
    const engine = getEngine();
    expect(await engine.searchTrigram('a of to')).toEqual([]);
    expect(await engine.searchTrigram('')).toEqual([]);
  });

  test('honors the page-grain type filter', async () => {
    const engine = getEngine();
    expect((await engine.searchTrigram('인터엠디 카카오', { type: 'note' })).length).toBeGreaterThan(0);
    expect(await engine.searchTrigram('인터엠디 카카오', { type: 'person' })).toEqual([]);
  });

  // Corpus-frequency gate — the Postgres half. The probe rides inside the same
  // scoped read transaction as the main query, so this also covers the
  // real-pg_trgm/postgres.js path for `unnest($1::text[], $2::text[])` and the
  // ESCAPE clause, neither of which the PGLite mirror can prove.
  test('flood operands are gated away on Postgres too', async () => {
    const engine = getEngine();
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

    // 61 text chunks → threshold = max(50, 1.83) = 50; 자동/실행 occur in 60.
    expect(await engine.searchTrigram('자동 실행')).toEqual([]);

    // One survivor is not enough to co-occur with anything …
    expect(await engine.searchTrigram('자동 인터엠디')).toEqual([]);

    // … but two rare survivors still reach their page without dragging in the
    // 60-chunk bulk page the flood operand would have pulled.
    const mixed = await engine.searchTrigram('자동 인터엠디 카카오');
    expect(mixed.map((r) => r.slug)).toContain('notes/vendor-log');
    expect(mixed.map((r) => r.slug)).not.toContain('notes/bulk-log');
  });
});
