import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config'

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0
});

const result = await model.invoke("猫についてジョークを言ってください");
console.log(result)