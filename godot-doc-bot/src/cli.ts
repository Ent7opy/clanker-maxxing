import "dotenv/config";
import { buildDocsChain } from "./ingest";

(async () => {
  const chain = await buildDocsChain();
  const q = process.argv.slice(2).join(" ") || "What is a Node in Godot?";
  const ans = await chain.invoke({ question: q });
  console.log("\nQ:", q, "\n");
  console.log(ans);
})();


