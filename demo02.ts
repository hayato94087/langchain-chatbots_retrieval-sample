import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const loader = new CheerioWebBaseLoader(
  "https://ja.wikipedia.org/wiki/LangChain"
);
const rawDocs = await loader.load();
console.log(rawDocs);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});
const allSplits = await textSplitter.splitDocuments(rawDocs);
console.log(allSplits);

const vectorstore = await MemoryVectorStore.fromDocuments(
  allSplits,
  new OpenAIEmbeddings()
);

const retriever = vectorstore.asRetriever(2);

const docs = await retriever.invoke("LangChainのライセンス形式は？");
console.log(docs);

const docs2 = await retriever.invoke("LangChainがサポートするプログラミング言語は？");
console.log(docs2);