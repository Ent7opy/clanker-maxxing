import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { DocumentInterface } from "@langchain/core/documents";

export async function buildRetrieverFromDocs(
  docs: DocumentInterface[],
  opts = { chunkSize: 1200, chunkOverlap: 200, k: 8 }
) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: opts.chunkSize,
    chunkOverlap: opts.chunkOverlap,
  });

  const splits = await splitter.splitDocuments(docs);

  // Cheap dedup by (source + first 200 chars)
  const seen = new Set<string>();
  const unique = splits.filter((d) => {
    const key = (d.metadata?.source ?? "unknown") + "|" + d.pageContent.slice(0, 200);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
  const store = await MemoryVectorStore.fromDocuments(unique, embeddings);
  return store.asRetriever(opts.k);
}

export function makeQaChain(retriever: any) {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a concise Godot documentation assistant.
Answer ONLY from the provided context. If unsure, say "I don't know".
After the answer, list "Sources:" with URLs or file paths.`,
    ],
    ["human", "Question: {question}\n\nContext:\n{context}"],
  ]);

  const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const parser = new StringOutputParser();

  return RunnableSequence.from([
    {
      question: new RunnablePassthrough(),
      docs: retriever,
    },
    async ({ question, docs }: { question: string; docs: DocumentInterface[] }) => {
      const body = docs.map((d) => d.pageContent).join("\n---\n");
      const cites = Array.from(new Set(docs.map((d) => d.metadata?.source))).slice(0, 8);
      const context = body + "\n\nSources:\n" + cites.join("\n");
      return { question, context };
    },
    prompt,
    model,
    parser,
  ]);
}


