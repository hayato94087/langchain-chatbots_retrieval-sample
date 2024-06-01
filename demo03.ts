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
import { HumanMessage, AIMessage } from "@langchain/core/messages";

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

const SYSTEM_TEMPLATE = `ユーザーの質問に、以下の「コンテキスト」に基づいて回答してください。「コンテキスト」に質問に関連する情報が含まれていない場合は、何かを作り出さずに「わかりません」とだけ答えてください。「コンテキスト」は<context>と</context>の間に含まれている文章です。":

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