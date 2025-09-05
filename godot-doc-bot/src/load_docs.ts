import { SitemapLoader } from "@langchain/community/document_loaders/web/sitemap";
import { JSDOM } from "jsdom";
import { Document, type DocumentInterface } from "@langchain/core/documents";

const SITEMAP_URL = "https://docs.godotengine.org/en/stable/sitemap.xml";

export async function loadGodotDocs(maxUrls = 600): Promise<DocumentInterface[]> {
  const loader = new SitemapLoader(SITEMAP_URL);

  // Get all sitemap URLs, then filter down to stable docs pages
  const elements = await loader.parseSitemap();
  const urls = elements
    .map((e) => e.loc)
    .filter((u) => /https:\/\/docs\.godotengine\.org\/en\/stable\//.test(u));

  const limited = urls.slice(0, maxUrls);

  const documents = await Promise.all(
    limited.map(async (url: string) => {
      const response = await fetch(url);
      const html = await response.text();

      const dom = new JSDOM(html);
      const doc = dom.window.document;
      const container = doc.querySelector("div[role='main']") ?? doc.body;

      // Remove obvious chrome
      container.querySelectorAll("nav, aside, header, footer").forEach((n: Element) => n.remove());

      // Preserve code blocks by wrapping in triple backticks
      container.querySelectorAll("pre").forEach((pre: Element) => {
        const content = pre.textContent ?? "";
        pre.replaceWith(doc.createTextNode("```\n" + content + "\n```"));
      });

      const text = (container.textContent ?? "")
        .replace(/\u00A0/g, " ")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

      return new Document({
        pageContent: text,
        metadata: {
          source: url,
          site: "godot-docs",
          version: "stable",
          kind: "docs",
        },
      });
    })
  );

  return documents;
}


