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
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { RunnableBranch } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

// LLM の対話モデル
const llm = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
});

// Webドキュメントをダウンロード
const loader = new CheerioWebBaseLoader(
  "https://ja.wikipedia.org/wiki/LangChain"
);
const rawDocs = await loader.load();

// テキストを分割してチャンクを作成
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});
const allSplits = await textSplitter.splitDocuments(rawDocs);

// ベクトルストアを作成
const vectorstore = await MemoryVectorStore.fromDocuments(
  allSplits,
  new OpenAIEmbeddings()
);

// ベクターストアから情報を取得するRetrieverを作成
const retriever = vectorstore.asRetriever(3);

// システムテンプレート
const SYSTEM_TEMPLATE = `# 指示
以下の質問に回答してください。質問に対する情報がコンテキストによって提供されない場合、または明確な情報源が存在しない場合は、『わかりません』とだけ回答してください。推測や創作はしないでください。

質問に対する情報が見つからない場合、必ず『わかりません』と回答してください。例えば、以下の質問に対してコンテキストに情報が含まれない場合です。

質問：「少年ジャンプで掲載されていた『ナルト』について教えて

# コンテキスト
{context}
`;

// 質問応答のプロンプト
const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM_TEMPLATE],
  new MessagesPlaceholder("messages"),
]);

// ドキュメントチェーンを作成
const documentChain = await createStuffDocumentsChain({
  llm,
  prompt: questionAnsweringPrompt,
});

// ユーザーからの問い合わせを取得
const parseRetrieverInput = (params: { messages: BaseMessage[] }) => {
  const lastMessage = params.messages[params.messages.length - 1];
  if (lastMessage) {
    return lastMessage.content;
  }
  return "";
};

// Retrieverとドキュメントチェーンを組み合わせたチェーンを作成
const retrievalChain = RunnablePassthrough.assign({
  context: RunnableSequence.from([parseRetrieverInput, retriever]),
}).assign({
  answer: documentChain,
});

const result = await retrievalChain.invoke({
  messages: [new HumanMessage("LangChainのライセンス形式は？")],
});
console.log(result)

const result2 = await retrievalChain.invoke({
  messages: [new HumanMessage("もっと教えて")],
});
console.log(result2);

const queryTransformPrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("messages"),
  [
    "user",
    "上記の会話を踏まえ、会話に関連する情報を得るための検索クエリを生成してください。クエリのみを回答し、それ以外のことは書かないでください。",
  ],
]);
const queryTransformationChain = queryTransformPrompt.pipe(llm);
const result3 = await queryTransformationChain.invoke({
  messages: [
    new HumanMessage("LangChainのライセンス形式は？"),
    new AIMessage(
      "LangChainのライセンス形式はMITライセンスです。"
    ),
    new HumanMessage("もっと教えて"),
  ],
});
console.log(result3)

const queryTransformingRetrieverChain = RunnableBranch.from([
  [
    (params: { messages: BaseMessage[] }) => params.messages.length === 1,
    RunnableSequence.from([parseRetrieverInput, retriever]),
  ],
  queryTransformPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriever),
]).withConfig({ runName: "chat_retriever_chain" });

const conversationalRetrievalChain = RunnablePassthrough.assign({
  context: queryTransformingRetrieverChain,
}).assign({
  answer: documentChain,
});

const result4 = await conversationalRetrievalChain.invoke({
  messages: [new HumanMessage("LangChainのライセンス形式は？")],
});
console.log(result4)

const result5 = await conversationalRetrievalChain.invoke({
  messages: [
    new HumanMessage("LangChainのライセンス形式は？"),
    new AIMessage(
      "LangChainのライセンス形式はMITライセンスです。"
    ),
    new HumanMessage("もっと教えて"),
  ],
});
console.log(result5)

const result6 = await conversationalRetrievalChain.invoke({
  messages: [
    new HumanMessage("LangChainのライセンス形式は？"),
    new AIMessage(
      "LangChainのライセンス形式はMITライセンスです。"
    ),
    new HumanMessage("もっと教えて"),
    new HumanMessage("転生したらスライムだった件の作者は誰？")
  ],
});
console.log(result6)