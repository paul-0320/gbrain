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
 */

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
