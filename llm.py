from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""
You are a helpful and safe AI assistant.
Conversation so far:
{chat_history}
User: {user_input}
AI:
"""
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    output_key="text"
)

while True:
    user_input = input("Enter: ")
    res = chain.invoke({"user_input": user_input})
    print(res["text"])
