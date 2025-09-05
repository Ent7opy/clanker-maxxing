import "dotenv/config";
import { loadGodotDocs } from "./load_docs";
import { buildRetrieverFromDocs, makeQaChain } from "./rag";

export async function buildDocsChain() {
  const docs = await loadGodotDocs(600);
  const retriever = await buildRetrieverFromDocs(docs, { chunkSize: 1200, chunkOverlap: 200, k: 8 });
  return makeQaChain(retriever);
}

// For quick manual smoke test:
if (import.meta.url === `file://${process.argv[1]}`) {
  (async () => {
    const chain = await buildDocsChain();
    const q = process.argv.slice(2).join(" ") || "How do I export to Android in Godot?";
    const ans = await chain.invoke({ question: q });
    console.log(ans);
  })();
}


