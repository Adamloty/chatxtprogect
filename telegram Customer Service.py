from telegram.ext import Application, CommandHandler, MessageHandler, filters
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)

loader = TextLoader("data.txt")
pages = loader.load_and_split()

text_chunks = text_splitter.split_documents(pages)

vecstore = Chroma.from_documents(text_chunks, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.7),
    chain_type="stuff",
    retriever=vecstore.as_retriever(),
)

TELEGRAM_TOKEN = '6703759362:AAF7hAMgsD3gS51Lop7fmyyVfxZkvHu46nM'

async def start(update, context):
    await update.message.reply_text('Hello! This is your Customer Service . How can I help you today?')

async def handle_message(update, context):
    user_message = update.message.text
    message_chunks = text_splitter.split_text(user_message)
    answers = []
    for chunk in message_chunks:
        answer = qa.run(chunk)
        if not answer or 'I don`t understand a question' in answer:
            answer = "Can you clarify the question further?"
        answers.append(answer)
    full_answer = ' '.join(answers)
    await update.message.reply_text(full_answer)

application = Application.builder().token(TELEGRAM_TOKEN).build()

application.add_handler(CommandHandler("start", start))

application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

application.run_polling()
