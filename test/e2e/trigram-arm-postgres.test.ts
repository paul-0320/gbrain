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

  test('reaches a particle-suffixed mention that searchKeyword cannot (gap reproduced)', async () => {
    const engine = getEngine();

    const keyword = await engine.searchKeyword('인터엠디 미팅', { orFallback: true });
    expect(keyword.map((r) => r.slug)).not.toContain('notes/vendor-log');

    const trigram = await engine.searchTrigram('인터엠디 미팅');
    expect(trigram.map((r) => r.slug)).toContain('notes/vendor-log');
  });

  test('reaches a compound-only mention (카카오 → 카카오헬스케어)', async () => {
    const engine = getEngine();

    const keyword = await engine.searchKeyword('카카오 제휴', { orFallback: true });
    expect(keyword.map((r) => r.slug)).not.toContain('notes/vendor-log');

    const trigram = await engine.searchTrigram('카카오 제휴');
    expect(trigram.map((r) => r.slug)).toContain('notes/vendor-log');
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
    expect((await engine.searchTrigram('인터엠디 미팅', { type: 'note' })).length).toBeGreaterThan(0);
    expect(await engine.searchTrigram('인터엠디 미팅', { type: 'person' })).toEqual([]);
  });
});
