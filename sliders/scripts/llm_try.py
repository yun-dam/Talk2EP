from sliders.llm import get_llm_client, init_llm, load_fewshot_prompt_template
import os
import asyncio
from pydantic import BaseModel


class Joke(BaseModel):
    joke: str


current_dir = os.path.dirname(os.path.abspath(__file__))
init_llm("prompts", os.path.join(current_dir, ".env"))

llm = get_llm_client(model="gpt-4.1", temperature=0.0)

template = load_fewshot_prompt_template(
    template_file="test.prompt",
    template_blocks=[
        ("instruction", "You are a helpful assistants."),
        ("input", "Question: {{question}}\nDocument: {{document}}"),
    ],
)

llm_chain = template | llm


async def main():
    res = await llm_chain.ainvoke(
        {"question": "What is the capital of France?", "document": "The capital of France is Paris."},
    )
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
