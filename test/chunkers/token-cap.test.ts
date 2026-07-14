/**
 * Markdown chunker v4 / code chunker v5 — estimated-token hard cap
 * regression tests.
 *
 * Reproduces a field failure: a local llama-server embedding backend
 * (`-ub 2048`) crashes deterministically (trace/BPT trap → EOF at the
 * client) when a single chunk exceeds ~2,050 real tokens. Two content
 * shapes triggered it:
 *
 *   1. Korean rollup docs with a per-line "출처: https://…" Notion link.
 *      The URLs' ASCII mass pushes CJK density below 0.30, flipping
 *      countCJKAwareWords to whitespace counting, where a 150-char URL
 *      counts as ONE word → chunks ballooned to 3-4K chars ≈ 2,000+
 *      real tokens (URL soup tokenizes at ~1.6 chars/token).
 *
 *   2. Large JSON code blocks (~7K chars) that the word pipeline
 *      undercounts the same way (few whitespace tokens).
 *
 * The fix: every emitted chunk must satisfy
 *   estimateEmbeddingTokens(chunk) <= maxTokens (default 1500)
 * where the estimate deliberately OVERSTATES real tokenizer counts.
 */

import { describe, test, expect } from 'bun:test';
import { chunkText, capByEstimatedTokens, DEFAULT_MAX_EST_TOKENS } from '../../src/core/chunkers/recursive.ts';
import { chunkCodeText } from '../../src/core/chunkers/code.ts';
import { estimateEmbeddingTokens } from '../../src/core/cjk.ts';

/** Synthesize the failing shape: Korean rollup lines each ending in a long Notion URL. */
function urlDenseKoreanRollup(lines: number): string {
  const out: string[] = ['# 운영회의 롤업 — 링크 밀집 재현', ''];
  for (let i = 0; i < lines; i++) {
    const hex32 = (i * 2654435761 >>> 0).toString(16).padStart(8, '0').repeat(4);
    out.push(
      `- **안건 ${i}**: 담당자 확인 후 다음 회의에서 결정하기로 함 · 연락처 062-000-${String(1000 + i)} · ` +
      `이메일 contact${i}@example.com · 출처: https://www.notion.so/acme/${hex32}?v=abcdef0123456789&pvs=4`,
    );
  }
  return out.join('\n');
}

/** Synthesize a large pretty-printed JSON block (the W3-1 shape). */
function bigJsonBlock(targetChars: number): string {
  const entries: string[] = [];
  let i = 0;
  let len = 0;
  while (len < targetChars) {
    const row =
      `  "item_${i}": { "name": "품목-${i}", "url": "https://example.com/api/v2/items/${i}?token=abc${i}def", "qty": ${i % 100}, "memo": "재고 확인 요청됨" }`;
    entries.push(row);
    len += row.length;
    i++;
  }
  return `{\n${entries.join(',\n')}\n}`;
}

describe('v4 estimated-token cap — URL-dense Korean rollup (populate v2 crash shape)', () => {
  test('every chunk stays under the estimated-token cap', () => {
    const md = urlDenseKoreanRollup(60);
    const chunks = chunkText(md);
    expect(chunks.length).toBeGreaterThan(0);
    for (const c of chunks) {
      expect(estimateEmbeddingTokens(c.text)).toBeLessThanOrEqual(DEFAULT_MAX_EST_TOKENS);
    }
  });

  test('no chunk reaches the measured 3K-char danger zone for URL soup', () => {
    const md = urlDenseKoreanRollup(60);
    const chunks = chunkText(md);
    // 1500 est tokens at the OTHER weight (0.75/char) bounds chunks to
    // ~2,000 chars for pure ASCII — well under the ~3,300 chars where
    // URL-dense content crosses ~2,050 real tokens (1.6 chars/token).
    for (const c of chunks) {
      expect(c.text.length).toBeLessThanOrEqual(2600);
    }
  });

  test('content is preserved (no lines dropped by the cap)', () => {
    const md = urlDenseKoreanRollup(60);
    const chunks = chunkText(md);
    const joined = chunks.map((c) => c.text).join('\n');
    // Spot-check first / middle / last rollup lines survive chunking.
    for (const marker of ['안건 0', '안건 30', '안건 59']) {
      expect(joined).toContain(marker);
    }
  });
});

describe('v4 estimated-token cap — large JSON blocks (W3-1 crash shape)', () => {
  test('7K-char pretty JSON through the prose path stays under the cap', () => {
    const md = `설정 파일 원문 보존:\n\n\`\`\`\n${bigJsonBlock(7000)}\n\`\`\`\n`;
    const chunks = chunkText(md);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(estimateEmbeddingTokens(c.text)).toBeLessThanOrEqual(DEFAULT_MAX_EST_TOKENS);
    }
  });

  test('7K-char minified JSON (single whitespace-less token) stays under the cap', () => {
    const minified = bigJsonBlock(7000).replace(/\n\s*/g, '');
    const chunks = chunkText(minified);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(estimateEmbeddingTokens(c.text)).toBeLessThanOrEqual(DEFAULT_MAX_EST_TOKENS);
    }
  });

  test('json fence via the code chunker stays under the cap (+header slack)', async () => {
    const chunks = await chunkCodeText(bigJsonBlock(7000), 'fence.json');
    expect(chunks.length).toBeGreaterThan(0);
    for (const c of chunks) {
      // buildChunk prepends a short "[JSON] fence.json:…" header AFTER the
      // body-level cap; allow ~60 est tokens of header slack. Real-token
      // safety margin (2,050 − overestimated 1,500) absorbs this easily.
      expect(estimateEmbeddingTokens(c.text)).toBeLessThanOrEqual(DEFAULT_MAX_EST_TOKENS + 60);
    }
  });
});

describe('v4 word-count floor — behavior preserved for normal content', () => {
  test('Latin prose chunking is unchanged by the floor (avg word < 6 chars)', () => {
    const prose = Array.from({ length: 120 }, (_, i) =>
      `This is sentence number ${i} and it talks about ordinary things in plain words.`,
    ).join(' ');
    const chunks = chunkText(prose);
    // Historical behavior: ~1,560 whitespace words → multiple ~300-word chunks.
    expect(chunks.length).toBeGreaterThan(3);
    for (const c of chunks) {
      const words = c.text.split(/\s+/).length;
      expect(words).toBeLessThanOrEqual(300 * 1.5 + 50); // merge cap + overlap
    }
  });

  test('Korean prose (CJK-dense, no URLs) never triggers the token cap', () => {
    const prose = Array.from({ length: 80 }, (_, i) =>
      `오늘 회의에서는 ${i}번째 안건을 검토했고 참석자 전원이 다음 분기 일정에 동의했다.`,
    ).join(' ');
    const chunks = chunkText(prose);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      // CJK-dense chunks are char-counted (≈450 max) — nowhere near 1500.
      expect(estimateEmbeddingTokens(c.text)).toBeLessThanOrEqual(700);
    }
  });
});

describe('capByEstimatedTokens unit behavior', () => {
  test('returns input unchanged when under the cap', () => {
    expect(capByEstimatedTokens('short text', 1500)).toEqual(['short text']);
    expect(capByEstimatedTokens('', 1500)).toEqual([]);
  });

  test('prefers newline cut points within the lookback window', () => {
    const line = 'x'.repeat(100);
    const text = Array.from({ length: 40 }, () => line).join('\n');
    const pieces = capByEstimatedTokens(text, 1000);
    expect(pieces.length).toBeGreaterThan(1);
    for (const p of pieces) {
      // Every piece should be whole lines (multiples of the 100-char line).
      for (const l of p.split('\n')) {
        expect(l).toBe(line);
      }
    }
  });

  test('makes forward progress on whitespace-less input (hard cut)', () => {
    const blob = 'a'.repeat(10_000);
    const pieces = capByEstimatedTokens(blob, 1000);
    expect(pieces.length).toBeGreaterThan(1);
    expect(pieces.join('')).toBe(blob);
    for (const p of pieces) {
      expect(estimateEmbeddingTokens(p)).toBeLessThanOrEqual(1000);
    }
  });
});

describe('estimateEmbeddingTokens — weight sanity', () => {
  test('overestimates URL-dense ASCII (0.75/char ≥ measured ~0.63/char)', () => {
    const url = 'https://www.notion.so/acme/a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4?v=abc&pvs=4';
    const est = estimateEmbeddingTokens(url);
    expect(est).toBeGreaterThanOrEqual(Math.floor(url.length * 0.7));
  });

  test('counts CJK at 1 token/char', () => {
    expect(estimateEmbeddingTokens('가나다라마')).toBe(5);
  });

  test('whitespace is nearly free', () => {
    expect(estimateEmbeddingTokens('   \n\t  ')).toBeLessThanOrEqual(1);
  });

  test('empty string is 0', () => {
    expect(estimateEmbeddingTokens('')).toBe(0);
  });
});
