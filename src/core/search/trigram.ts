/**
 * Trigram arm — query tokenization (`search.trigram_arm`).
 *
 * The hybrid keyword/title arms both go through `websearch_to_tsquery` on an
 * FTS config that cannot segment or stem CJK text. Under the 'english' config
 * a Korean chunk is tokenized on whitespace, so a surface form carrying a
 * particle ("인터엠디는") or a compound ("카카오헬스케어") becomes ONE lexeme
 * that the base-form query token ("인터엠디" / "카카오") never equals. The
 * chunk is then unreachable through the lexical arms no matter how the
 * fallback relaxes — OR-of-terms still compares whole lexemes.
 *
 * pg_trgm's `word_similarity(query_fragment, document)` is the missing
 * primitive: it slides the query's trigram set over the document and scores
 * the best-matching word extent, so a base form scores high against its own
 * inflected/compounded surface form. This module owns the QUERY side of that
 * arm — turning a natural-language query into the small set of fragments the
 * engines bind as `$n <% cc.chunk_text` operands.
 *
 * Pure + engine-agnostic on purpose: both `postgres-engine.searchTrigram` and
 * `pglite-engine.searchTrigram` must derive the SAME operand list, and the
 * unit tests pin the tokenizer without touching a database.
 *
 * ── Corpus-frequency gate (the no-IDF flood guard) ──────────────────────────
 * The arm's score is a SUM of per-token `word_similarity`, and
 * `word_similarity` carries NO rarity weighting: a common operational noun
 * scores ~1.0 against every chunk that happens to contain it, exactly like a
 * rare entity name does. Measured on the 22,351-chunk production corpus, a
 * 62-question regression gate went 57 → 45 answers found (12 lost, 0 rescued)
 * with the arm on. The autopsy: "자동" occurs in 23.9% of text chunks and
 * "실행" in 14.4%, so a query containing both handed dozens of unrelated pages
 * a perfect 2.00 tie at the top of the arm's list; RRF fuses the arm at the
 * keyword arm's weight, so that noise filled the reranker's 25-candidate
 * input window and pushed the true answer — rank 1 with the arm OFF — out of
 * it entirely.
 *
 * IDF is the textbook fix, but there is no term-statistics table to read it
 * from. `dropFloodTokens` is the cheap approximation: measure each operand's
 * document frequency once per search and drop the ones common enough to
 * flood, keeping the rare operands the arm actually exists to serve.
 */

import { escapeLikePattern } from './sql-ranking.ts';

/**
 * Characters that make a token "CJK-ish" for the length floor below.
 *
 * Wider than `src/core/cjk.ts`'s `CJK_SLUG_CHARS` on purpose — that constant
 * is load-bearing for slug grammar and chunker density heuristics and must
 * not be widened for a search-arm concern (its own doc says so). Here we also
 * accept Hangul Jamo (U+1100–11FF), Hangul Compatibility Jamo (U+3130–318F)
 * and CJK Extension A (U+3400–4DBF), because a decomposed or archaic form
 * still needs the short-token floor rather than the Latin one.
 */
const CJK_TOKEN_CHAR = /[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿一-鿿가-힯]/;

/** Wrapping openers: quotes, backticks, and bracket forms (ASCII + CJK). */
const LEADING_TRIM = /^[\s"'`«»“”‘’([{<「『【《〈]+/;

/**
 * Wrapping closers plus trailing sentence punctuation (ASCII + CJK).
 * Deliberately does NOT strip interior characters — `foo-bar` and `v1.2`
 * keep their shape; only the token's tail is trimmed.
 */
const TRAILING_TRIM = /[\s"'`«»“”‘’)\]}>」』】》〉.,!?;:…·、。！？，；：]+$/;

/**
 * Minimum length for a token containing CJK characters. Korean content words
 * are frequently 2 syllables ("회의", "예산"), and pg_trgm pads short strings
 * so a 2-char token still produces usable trigrams.
 */
export const TRIGRAM_MIN_CJK_TOKEN_LEN = 2;

/**
 * Minimum length for a Latin/digit token. Higher than the CJK floor because
 * short ASCII tokens are overwhelmingly stopwords and function words ("the",
 * "and", "for", "was"), and `word_similarity` has no IDF to demote them —
 * a 3-char operand would match a large fraction of any English corpus at the
 * default 0.6 threshold.
 */
export const TRIGRAM_MIN_LATIN_TOKEN_LEN = 4;

/**
 * Hard cap on operands. Each token becomes one more `<%` disjunct and one
 * more `word_similarity()` call per candidate row; the arm is a recall aid,
 * not an exhaustive scan, so the longest (most specific) tokens win.
 */
export const TRIGRAM_MAX_TOKENS = 8;

/** True when the token contains at least one CJK / Hangul character. */
function isCJKToken(token: string): boolean {
  return CJK_TOKEN_CHAR.test(token);
}

/** Strip wrapping quotes/brackets and trailing sentence punctuation. */
function trimToken(raw: string): string {
  return raw.replace(LEADING_TRIM, '').replace(TRAILING_TRIM, '');
}

/**
 * Extract the `word_similarity` operands for the trigram arm.
 *
 * Pipeline: whitespace split → trim wrappers/tail punctuation → dedup →
 * length filter (CJK ≥ 2, otherwise ≥ 4) → longest-first → cap at 8.
 *
 * Dedup is case-insensitive because pg_trgm lowercases both sides before
 * generating trigrams (IGNORECASE is compiled in): `Notion` and `notion` are
 * literally the same operand, so keeping both would burn a slot of the cap
 * for zero recall. The FIRST-seen spelling is the one returned.
 *
 * Returns `[]` for a query with no qualifying token — callers MUST treat that
 * as "arm contributes nothing" and skip the query entirely rather than build
 * a `WHERE` clause with no disjuncts.
 */
export function extractTrigramTokens(query: string): string[] {
  if (!query) return [];
  const seen = new Set<string>();
  const kept: string[] = [];
  for (const raw of query.split(/\s+/)) {
    const token = trimToken(raw);
    if (token.length === 0) continue;
    const key = token.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const floor = isCJKToken(token) ? TRIGRAM_MIN_CJK_TOKEN_LEN : TRIGRAM_MIN_LATIN_TOKEN_LEN;
    if (token.length < floor) continue;
    kept.push(token);
  }
  // Stable sort (ES2019+): equal-length tokens keep query order, so the
  // operand list — and therefore the generated SQL — is deterministic.
  kept.sort((a, b) => b.length - a.length);
  return kept.slice(0, TRIGRAM_MAX_TOKENS);
}

/**
 * Document-frequency ceiling, as a fraction of the corpus's text chunks. An
 * operand appearing in more than this share is treated as a flood term and
 * dropped from the arm.
 *
 * Calibrated against the production corpus (22,351 text chunks) that produced
 * the regression, where 3% works out to a 670-chunk ceiling.
 *
 * Dropped — including every token that caused the 12 lost answers:
 *   자동 23.9%, 실행 14.4%, 문제 14.3%, 코드 12.9%, 모델 7.4%, 번호 4.7%, 정부 3.7%
 * Kept — including the entity names the arm exists to serve:
 *   청크 2.6%, 주말 0.8%, 리랭커 0.5%, 인터엠디 0.4%
 *
 * Two honest caveats. First, the cut is NOT in an empty band: 정부 (3.7%) is
 * dropped and 청크 (2.6%) is kept, so tokens in the 2–5% range sit close
 * enough to the line that a small change in either the constant or the corpus
 * flips them. What the threshold does buy with real margin is the part that
 * matters — the flood terms start at 7.4% and the arm's actual targets are
 * an order of magnitude below the line at 0.4–0.8%.
 *
 * Second, it is tuned to ONE corpus; a brain with different topic
 * concentration will want a different number. Both caveats point the same
 * way: the durable fix is real IDF weighting inside the score, so a
 * mid-frequency token is merely down-weighted instead of being deleted by a
 * binary gate. This constant is the cheap stand-in until then.
 */
export const TRIGRAM_DF_RATIO = 0.03;

/**
 * Absolute floor for the drop threshold, in chunks.
 *
 * Without it the ratio gate degenerates on a small corpus: at 200 text chunks
 * the ceiling would be 6, so a token appearing in 7 chunks — genuinely rare in
 * any meaningful sense — gets dropped, and on a fresh brain the arm would gate
 * away nearly every operand it was handed. Flooding the reranker's input
 * window is not even possible at that scale, so there is nothing to guard
 * against. The gate only starts biting once `TRIGRAM_DF_RATIO * total` exceeds
 * this floor, i.e. from ~1,667 text chunks upward.
 */
export const TRIGRAM_DF_MIN_COUNT = 50;

/**
 * Drop operands common enough to flood the arm's candidate list.
 *
 * Threshold: `count > max(TRIGRAM_DF_MIN_COUNT, TRIGRAM_DF_RATIO * total)`.
 * Strictly greater, so a token sitting exactly on the line survives.
 *
 * Order is preserved (callers rely on the longest-first ordering from
 * `extractTrigramTokens`). A token missing from `counts` is treated as
 * frequency 0 and kept — a failed or partial measurement must not silently
 * delete operands, and the arm is opt-in anyway.
 *
 * Returning `[]` (every operand flooded) is a NORMAL outcome, not an error:
 * it means the query consists entirely of corpus-common words, which is
 * exactly the case where the arm contributes noise instead of recall. Callers
 * MUST skip the arm's query entirely rather than emit a disjunct-less WHERE.
 */
export function dropFloodTokens(
  tokens: string[],
  counts: Map<string, number> | Record<string, number>,
  totalTextChunks: number,
): string[] {
  if (tokens.length === 0) return [];
  const read = (token: string): number => {
    const raw = counts instanceof Map ? counts.get(token) : counts[token];
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : 0;
  };
  const threshold = Math.max(TRIGRAM_DF_MIN_COUNT, TRIGRAM_DF_RATIO * totalTextChunks);
  return tokens.filter((token) => read(token) <= threshold);
}

/**
 * Build the `LIKE` pattern that measures one operand's document frequency.
 *
 * The DF probe is a containment count, NOT the arm's own `<%` predicate:
 * `LIKE '%tok%'` is what the v126 GIN gin_trgm_ops index answers cheapest, and
 * for a flood check "how many chunks contain this substring" is the right
 * question — a token that literally appears in a quarter of the corpus is the
 * thing being guarded against, regardless of how word_similarity would score
 * it. It also slightly OVER-counts relative to `<%` (containment is looser
 * than word-extent similarity), which errs toward dropping, the safe side.
 *
 * `%`, `_` and `\` inside the token are escaped, so the SQL MUST pair this
 * with an explicit `ESCAPE '\'` clause. Without the escape, a query token
 * containing `%` would match every chunk and gate itself away.
 */
export function buildTrigramDfLikePattern(token: string): string {
  return `%${escapeLikePattern(token)}%`;
}

/**
 * The document-frequency probe, as ONE round trip: corpus size plus a
 * per-operand containment count.
 *
 * Bind `$1` = the operand array (echoed back so the caller can key the counts
 * by token without relying on row order) and `$2` = the matching
 * `buildTrigramDfLikePattern` array. Runs inside the arm's existing scoped
 * read transaction, so it inherits the same statement timeout and — on an
 * RLS deployment — the same visible rows.
 *
 * Shared verbatim by both engines rather than duplicated per engine like
 * searchKeyword's SQL is: a gate that drifted between Postgres and PGLite
 * would silently mean one of them still floods, which is precisely the
 * regression this exists to prevent.
 *
 * Scope note: numerator and denominator are BOTH unfiltered by the caller's
 * type/source/hard-exclude filters. That is deliberate — this measures how
 * common a word is in the corpus, not in the caller's slice, and filtering one
 * side without the other would skew the ratio. The counts never leave this
 * function's caller, so there is nothing to leak.
 */
export const TRIGRAM_DF_PROBE_SQL = `
  WITH totals AS (
    SELECT count(*)::bigint AS total
      FROM content_chunks
     WHERE modality = 'text'
  )
  SELECT
    totals.total AS total_text_chunks,
    probe.token  AS token,
    (SELECT count(*)::bigint
       FROM content_chunks cc
      WHERE cc.modality = 'text'
        AND cc.chunk_text LIKE probe.pattern ESCAPE '\\') AS df
  FROM totals, unnest($1::text[], $2::text[]) AS probe(token, pattern)
`;

/**
 * Fold `TRIGRAM_DF_PROBE_SQL` rows into the shape `dropFloodTokens` wants.
 * Tolerates an empty result set (returns total 0, which makes the absolute
 * floor the only active bound) and bigint-as-string counts from either driver.
 */
export function parseTrigramDfRows(
  rows: ReadonlyArray<Record<string, unknown>>,
): { totalTextChunks: number; counts: Map<string, number> } {
  const counts = new Map<string, number>();
  let totalTextChunks = 0;
  for (const row of rows) {
    const total = Number(row.total_text_chunks);
    if (Number.isFinite(total)) totalTextChunks = total;
    const token = row.token;
    const df = Number(row.df);
    if (typeof token === 'string' && Number.isFinite(df)) counts.set(token, df);
  }
  return { totalTextChunks, counts };
}
