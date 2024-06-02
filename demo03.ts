import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const llm = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0
});

const loader = new CheerioWebBaseLoader(
  "https://ja.wikipedia.org/wiki/LangChain"
);
const rawDocs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});
const allSplits = await textSplitter.splitDocuments(rawDocs);

const vectorstore = await MemoryVectorStore.fromDocuments(
  allSplits,
  new OpenAIEmbeddings()
);

const retriever = vectorstore.asRetriever(3);
const docs = await retriever.invoke("ライセンス形式は？");

const SYSTEM_TEMPLATE = `# 指示
以下の質問に回答してください。質問に対する情報がコンテキストによって提供されない場合、または明確な情報源が存在しない場合は、『わかりません』とだけ回答してください。推測や創作はしないでください。

質問に対する情報が見つからない場合、必ず『わかりません』と回答してください。例えば、以下の質問に対してコンテキストに情報が含まれない場合です。

質問：「少年ジャンプで掲載されていた『ナルト』について教えて

# コンテキスト
{context}

<context>
{context}
</context>
`;

const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM_TEMPLATE],
  new MessagesPlaceholder("messages"),
]);

const documentChain = await createStuffDocumentsChain({
  llm,
  prompt: questionAnsweringPrompt,
});

const result = await documentChain.invoke({
  messages: [new HumanMessage("LangChainのライセンス形式は？")],
  context: docs,
});
console.log(result);


const result2 = await documentChain.invoke({
  messages: [new HumanMessage("LangChainのライセンス形式は？")],
  context: [],
});
console.log(result2);