import 'dotenv/config';
import { loadJSON, cosine } from './utils.js';
import { PromptTemplate } from "@langchain/core/prompts";
import { formatContexts, formatExamples } from "./formatters.js";

const OLLAMA = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const GEN_MODEL = process.env.GEN_MODEL || 'llama3.2:3b';
const TOP_K = parseInt(process.env.TOP_K || '3', 10);
const SIM_THRESHOLD = Number(process.env.SIM_THRESHOLD || '0.20');

async function embed(text, embModel) {
const res = await fetch(`${OLLAMA}/api/embeddings`, {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({ model: embModel, prompt: text })
});
if (!res.ok) throw new Error(`Embedding failed: ${res.status} ${res.statusText}`);
const json = await res.json();
return json.embedding;
}








function topKByCosine(index, queryVec, k, threshold) {
    const scored = index.chunks.map(c => ({ ...c, score: cosine(queryVec, c.embedding) }))
    .sort((a, b) => b.score - a.score);
    const filtered = scored.filter(s => s.score >= threshold);
    return (filtered.length ? filtered : scored).slice(0, k);
    }

    
      export async function buildPrompt(a, b, c) {
        // Support BOTH styles:
        //   buildPrompt(question, contexts, examples)
        //   buildPrompt({ question, contexts, examples })
        let question = '', contexts = [], examples = [];
        if (a && typeof a === 'object' && (a.question !== undefined || a.contexts !== undefined || a.examples !== undefined)) {
          ({ question = '', contexts = [], examples = [] } = a);
        } else {
          question = a ?? '';
          contexts = b ?? [];
          examples = c ?? [];
        }
      
        const template = `
      You are an expert teaching assistant for Snap (a block-based programming language similar to Scratch).
      Your task is to answer only using the provided context, which may include manuals, examples, or excerpts from Snap documentation.
      
      --- CONTEXT (Manual/Notes) ---
      {context}
      --- EXAMPLES (Answer Style) ---
      {formattedExamples}
      --- USER QUESTION ---
      {question}

      {lowSolution}

      {outputStyle}

      {requirement}
      `;
      
        const prompt = new PromptTemplate({
          template,
          inputVariables: ["context", "formattedExamples", "question", "lowSolution", "outputStyle","requirement"],
        });
      
        return prompt.format({
          context: formatContexts(contexts),            
          formattedExamples: formatExamples(examples), 
          lowSolution : LOW_SOLUTION_SNAP,
          outputStyle : OUTPUT_STYLE_STEPS_THEN_HINT,
          requirement : REQUIREMENT_BEGINNER ,
          question: String(question ?? ''),
        });
      }

      
   
    /*
    function buildPrompt(question, contexts) {
    const header = [
    'You are a helpful assistant that answers **only** using the context provided.',
    'If the answer is not explicitly supported by the context, reply exactly: "this is beyond my scope."',
    'Be concise and factual. Do not invent information.'
    ].join('\n');
    
    const contextBlock = contexts.map((c, i) => `<<chunk ${i + 1} (score=${c.score.toFixed(2)})>>\n${c.text}`).join('\n\n');
    
    return `${header}\n\nContext:\n${contextBlock}\n\nQuestion: ${question}\nAnswer:`;
    }*/
    
    export async function answerQuestion(question) {
    const index = await loadJSON('data/index.json');
    if (!index || !index.chunks?.length) {
    return { text: 'this is beyond my scope.' };
    }

    const qVec = await embed(question, index.embModel);
const hits = topKByCosine(index, qVec, TOP_K, SIM_THRESHOLD) || [];

// extra guardrail: if best similarity is very low, short-circuit
if (!hits.length || hits[0].score < SIM_THRESHOLD) {
return { text: 'this is beyond my scope.' };
}


const examples = userExamples || [];     // [] is fine

const prompt = await buildPrompt({
  question: question ?? '',
  contexts: hits,
  examples,
  lowSolution: LOW_SOLUTION_SNAP,
  outputStyle: OUTPUT_STYLE_STEPS_THEN_HINT,
  requirement: REQUIREMENT_BEGINNER 
});



const res = await fetch(`${OLLAMA}/api/generate`, {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
    model: GEN_MODEL,
    prompt: prompt,       // make sure this is the string from buildPrompt.format()
    stream: false,
    options: {
      num_predict: 220,
      temperature: 0.1,
      stop: ['\n---', '\n# Example', '\n--- REQUIREMENTS ---']
    }
  })
});
if (!res.ok) {
    // fail closed
    return { text: 'this is beyond my scope.' };
    }
    
    const json = await res.json();
    // normalize final text (ollama returns { response, done, ... })
    const out = (json.response || '').trim();
    
    // final guard: if LLM ignored instructions, enforce policy
    const safe = out && !/^\s*I\s+don\'t\s+have|^\s*As\s+an\s+AI|^\s*I\'m\s+not\s+sure/i.test(out)
    ? out
    : 'this is beyond my scope.';
    console.log(safe);
    // lightweight heuristic: if it mentions "based on my knowledge" or lacks context cues and top score barely passes threshold, still allow but it's okay.
    return { text: safe };
    }