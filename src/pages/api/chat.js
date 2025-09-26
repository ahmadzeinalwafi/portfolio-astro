import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import * as fs from "fs/promises";
import * as path from "path";

export const prerender = false;

// --- Model and API Key Initialization ---
const GOOGLE_API_KEY = import.meta.env.GOOGLE_API_KEY;
if (!GOOGLE_API_KEY) {
  throw new Error("FATAL ERROR: GOOGLE_API_KEY is not set.");
}

const model = new ChatGoogleGenerativeAI({
  apiKey: GOOGLE_API_KEY,
  model: "gemini-2.0-flash",
});

// --- RAG Initialization (runs once when the server starts) ---
const portfolioPath = path.resolve(process.cwd(), "src", "data", "portfolio.md");
const portfolioText = await fs.readFile(portfolioPath, "utf-8");

const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
const docs = await textSplitter.createDocuments([portfolioText]);

const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: GOOGLE_API_KEY, model: "models/gemini-embedding-001" });
const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
const retriever = vectorStore.asRetriever();

const template = `You are an AI assistant helping answer questions about Ahmad Zein Al Wafi
Answer the user's question ONLY based on the context provided below.
If the question is not related to the context about Ahmad Zein Al Wafi, his portfolio, history, skills, or projects, politely decline that you cannot provided an response to the context because you can only answer questions and explain about Ahmad Zein Al Wafi. 

Context:
{context}

Question:
{question}`;

const prompt = PromptTemplate.fromTemplate(template);

const chain = RunnableSequence.from([
  {
    context: RunnableSequence.from([
      (input) => input.question,
      retriever,
      (docs) => docs.map((d) => d.pageContent).join("\n\n"),
    ]),

    question: (input) => input.question,
  },
  prompt,
  model,
  new StringOutputParser(),
]);

// --- POST Request Handler ---
export async function POST({ request }) {
  try {
    const { history } = await request.json();
    if (!history || history.length === 0) {
      throw new Error("Invalid history format");
    }

    const latestQuestion = history[history.length - 1].content;

    const stream = await chain.stream({
      question: latestQuestion,
    });

    const readableStream = new ReadableStream({
      async start(controller) {
        for await (const chunk of stream) {
          controller.enqueue(new TextEncoder().encode(chunk));
        }
        controller.close();
      },
    });

    return new Response(readableStream, {
      headers: { "Content-Type": "text/plain; charset-utf-8" },
    });

  } catch (error) {
    console.error("--- API Error in POST ---", error);
    return new Response(JSON.stringify({ 
      error: "An internal server error occurred." 
    }), { status: 500 });
  }
}