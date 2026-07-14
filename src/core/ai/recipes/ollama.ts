import type { Recipe } from '../types.ts';

export const ollama: Recipe = {
  id: 'ollama',
  name: 'Ollama (local)',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  base_url_default: 'http://localhost:11434/v1',
  auth_env: {
    required: [], // Ollama runs unauthenticated locally; users pass `ollama` as the key.
    optional: ['OLLAMA_BASE_URL', 'OLLAMA_API_KEY'],
    setup_url: 'https://ollama.ai',
  },
  touchpoints: {
    embedding: {
      // #2271: modern local embed models added so assertTouchpoint accepts them.
      // Each carries its own native dim (qwen3-embed-8b=4096, arctic-l-v2=1024);
      // the recipe-wide default_dims below is only the nomic fallback.
      models: [
        'nomic-embed-text',
        'mxbai-embed-large',
        'all-minilm',
        'qwen3-embedding:8b',
        'qwen3-embedding:4b',
        'qwen3-embedding:0.6b',
        'qwen3-embed-8b',
        'snowflake-arctic-embed-l-v2',
      ],
      default_dims: 768, // nomic-embed-text native dim
      // Ollama honors the OpenAI `dimensions` param (MRL truncation) for
      // Matryoshka models like qwen3-embedding; validated dims listed here.
      // NOTE: dims_options is Tier 1 in isCustomDimValidForProvider and
      // short-circuits BOTH accept and reject, so trust_custom_dims below is
      // inert while this list exists — kept to match upstream (#2271) and to
      // take over if the allow-list is ever dropped.
      dims_options: [256, 512, 768, 1024, 1536, 2048, 2560, 4096],
      trust_custom_dims: true, // #2271: local models carry varied native dims
      cost_per_1m_tokens_usd: 0,
      price_last_verified: '2026-04-20',
      // Ollama's batch capacity depends on the locally loaded model + the
      // OLLAMA_NUM_PARALLEL config; no static cap to declare. v0.32 (#779).
      no_batch_cap: true,
    },
  },
  setup_hint: 'Install Ollama from https://ollama.ai, then `ollama pull nomic-embed-text` and `ollama serve`.',
};
