# from typing import cast
# from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableConfig
# from operator import itemgetter
# from utils import create_json_from_text, create_knowledge_graph, extract_text_from_pdf, split_text_into_chunks, summarize_chunks 
from dotenv import load_dotenv
# import chainlit as cl
import os 

load_dotenv()
# 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")
model.invoke("สวัสดี")



# @cl.on_chat_start
# async def start():
#     model = ChatOpenAI(model="gpt-4o-mini")
#     embedding = OpenAIEmbeddings(model="text-embedding-3-small")
#     cl.Message("สวัสดีครับ ผมคือ Alfred AI ผมสามารถช่วยคุณสรุปเอกสารได้ครับ")
#     files = None

#     # Wait for the user to upload a file
#     while files is None:
#         files = await cl.AskFileMessage(
#             content="อัพโหลดเอกสารที่ต้องการสรุป",
#             accept=["application/pdf"],
#             max_size_mb=20,
#             timeout=180,
#         ).send()

#     file = files[0]

#     # Load the PDF file
#     documents = extract_text_from_pdf(file.path)
#     chunks = split_text_into_chunks(documents)

#     await cl.Message(content="กำลังบันทึกข้อมูลไปที่ระบบ...").send()
#     db = Chroma.from_documents(chunks, embedding,persist_directory="pdf_db")

#     await cl.Message("กำลังสรุปเอกสาร... กรุณารอสักครู่ครับ").send()

#     # Summarize the PDF file
#     summary_list = summarize_chunks(chunks,model)
#     summary_string = "\n".join(summary_list)

#     await cl.Message("กำลังสร้าง JSON จากข้อมูล... กรุณารอสักครู่ครับ").send()
#     json_data = create_json_from_text(summary_string)

#     await cl.Message("กำลังสร้าง Knowledge Graph จากข้อมูล... กรุณารอสักครู่ครับ").send()
#     create_knowledge_graph(json_data)

#     await cl.Message("Knowledge Graph ถูกสร้างเรียบร้อยแล้ว ").send()

#     #display the knowledge graph
#     with open("knowledge_graph.png", "rb") as image_file:
#         await cl.Message(image_file.read()).send()
    
#     #display the summary
#     await cl.Message(summary_string).send()

#     #setting runnable 
#     prompt_rag = """
#     You are a helpful assistant named Alfred AI that can answer questions about the uploaded PDF file.
#     You need to answer the question based on the context.
#     if you don't know the answer, just say "ไม่ทราบครับ"

#     Question: {message}
#     Context: {context}
#     """

#     naive_retrieval = db.as_retriever(search_kwargs={"k": 3})
#     prompt_template = ChatPromptTemplate.from_template(prompt_rag)

#     setup_and_retrieval = RunnableParallel({"question": itemgetter("question") |  RunnablePassthrough(), "context": itemgetter("question") | naive_retrieval}) | RunnablePassthrough.assign(context=itemgetter("context"))
#     runnable = setup_and_retrieval | {"response": prompt_template| model, "context": itemgetter("context")}

#     cl.user_session.set("runnable", runnable)
#     cl.user_session.set("db", db)
#     cl.user_session.set("summary_string", summary_string)

#     await cl.user_message("จัดการข้อมูลเรียบร้อยแล้ว สามารถถามคำถามได้ครับ").send()

# @cl.on_message
# async def message(message: cl.Message):
#   # summary_string = cl.user_session.get("summary_string")
#   runnable = cast(RunnableParallel, cl.user_session.get("runnable"))

#   ai_message = cl.Message(content="")

#   async for chunk in runnable.astream(
#       {"question": message.content},
#       config=RunnableConfig(callbacks=[cl.LangChainStreamCallbackHandler()])
#   ):
#     ai_message.stream_text(chunk)

