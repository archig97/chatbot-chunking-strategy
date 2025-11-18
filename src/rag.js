// textbook_rag.js
import 'dotenv/config';
import { loadJSON, cosine } from './utils.js';

// Ollama + RAG config
const OLLAMA = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const GEN_MODEL = process.env.GEN_MODEL || 'llama3.2:3b';
const TOP_K = parseInt(process.env.TOP_K || '3', 10);
const SIM_THRESHOLD = Number(process.env.SIM_THRESHOLD || '0.20');

/**
 * Get an embedding vector from Ollama for a given text.
 */
async function embed(text, embModel) {
  const res = await fetch(`${OLLAMA}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: embModel, prompt: text })
  });

  if (!res.ok) {
    throw new Error(`Embedding failed: ${res.status} ${res.statusText}`);
  }

  const json = await res.json();
  return json.embedding;
}

/**
 * Rank chunks by cosine similarity and return top K above threshold.
 */
function topKByCosine(index, queryVec, k, threshold) {
  const scored = index.chunks
    .map(c => ({ ...c, score: cosine(queryVec, c.embedding) }))
    .sort((a, b) => b.score - a.score);

  const filtered = scored.filter(s => s.score >= threshold);
  return (filtered.length ? filtered : scored).slice(0, k);
}

/**
 * Build a simple prompt for textbook Q&A.
 * Contexts are the retrieved textbook chunks.
 */
function buildPrompt(question, contexts) {
  const header = [
    'You are a helpful teaching assistant answering questions about a textbook.',
    'You must answer **only** using the provided textbook excerpts.',
    'If the answer is not clearly supported by the excerpts, reply exactly:',
    '"this is beyond my scope."',
    '',
    'Be concise, factual, and do not invent information.'
  ].join('\n');

  const contextBlock = contexts
    .map((c, i) => `<<excerpt ${i + 1} (score=${c.score.toFixed(2)})>>\n${c.text}`)
    .join('\n\n');

  return [
    header,
    '',
    '--- TEXTBOOK EXCERPTS ---',
    contextBlock,
    '',
    '--- QUESTION ---',
    question,
    '',
    '--- ANSWER ---'
  ].join('\n');
}

/**
 * Main RAG entry point: given a question string,
 * retrieve textbook chunks and generate an answer.
 */
export async function answerQuestion(question) {
  // Load your pre-built textbook index
  const index = await loadJSON('data/index.json');
  if (!index || !index.chunks?.length) {
    return { text: 'this is beyond my scope.' };
  }

  // Embed the question with the same model used for the index
  const qVec = await embed(question, index.embModel);

  // Retrieve top-k relevant chunks
  const hits = topKByCosine(index, qVec, TOP_K, SIM_THRESHOLD) || [];

  // Extra guardrail: if similarity is very low, bail out
  if (!hits.length || hits[0].score < SIM_THRESHOLD) {
    return { text: 'this is beyond my scope.' };
  }

  // Build prompt using the retrieved textbook excerpts
  const prompt = buildPrompt(question ?? '', hits);

  // Call Ollama to generate an answer
  const res = await fetch(`${OLLAMA}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: GEN_MODEL,
      prompt,
      stream: false,
      options: {
        num_predict: 220,
        temperature: 0.1,
      }
    })
  });

  if (!res.ok) {
    return { text: 'this is beyond my scope.' };
  }

  const json = await res.json();
  const out = (json.response || '').trim();

  // Simple safety/quality guard
  const safe = out && !/^\s*As an AI|^\s*I\s+don\'t\s+have|^\s*I\'m\s+not\s+sure/i.test(out)
    ? out
    : 'this is beyond my scope.';

  console.log(safe);
  return { text: safe };
}
