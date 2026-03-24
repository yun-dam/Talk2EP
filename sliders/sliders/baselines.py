from sliders.document import Document
from sliders.llm_models import SequentialAnswer
from sliders.llm_tools.code import run_python_code
from sliders.log_utils import logger
from sliders.rlm.rlm_repl import RLM_REPL
from sliders.system import System
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler

from langgraph.prebuilt import create_react_agent


class LLMWithToolUseSystem(System):
    def _setup_chains(self):
        tool_use_llm_client = get_llm_client(**self.config["models"]["tool_use"])
        self.tool_use_agent = create_react_agent(
            model=tool_use_llm_client,
            tools=[run_python_code],
        )

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> str:
        metadata = {}
        logger.info(f"Running tool use system for question: {question}")

        handler = LoggingHandler(
            prompt_file="default",
            metadata={
                "question": question,
                "stage": "tool_use",
                "question_id": kwargs.get("question_id", None),
            },
        )

        documents = "\n".join([document.content for document in documents])
        res = await self.tool_use_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Question: {question}
# Documents 
{documents}
""",
                    }
                ]
            },
            config={"callbacks": [handler]},
        )

        return res["messages"][-1].content, metadata


class LLMWithoutToolUseSystem(System):
    def _setup_chains(self):
        answer_llm_client = get_llm_client(**self.config["models"]["answer"])
        answer_template = load_fewshot_prompt_template(
            template_file="baselines/direct_without_tool_use.prompt",
            template_blocks=[],
        )
        self.answer_chain = answer_template | answer_llm_client

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> str:
        logger.info(f"Running without tool use system for question: {question}")
        metadata = {}
        handler = LoggingHandler(
            prompt_file="baselines/direct_without_tool_use.prompt",
            metadata={
                "question": question,
                "stage": "answer",
                "question_id": kwargs.get("question_id", None),
                **(metadata or {}),
            },
        )
        res = await self.answer_chain.ainvoke(
            {"question": question, "document": "\n".join([document.content for document in documents])},
            config={"callbacks": [handler]},
        )
        res = res.content
        metadata["answer_chain"] = str(res)
        return res, metadata


class LLMSequentialSystem(System):
    def _setup_chains(self):
        answer_llm_client = get_llm_client(**self.config["models"]["answer"])
        answer_template = load_fewshot_prompt_template(
            template_file=self.config["models"]["answer"]["template_file"],
            template_blocks=[],
        )
        self.answer_chain = answer_template | answer_llm_client.with_structured_output(SequentialAnswer)

    async def run(self, question: str, document: Document, *args, **kwargs) -> str:
        for i, chunk in enumerate(document.chunks):
            handler = LoggingHandler(
                prompt_file=self.config["models"]["answer"]["template_file"],
                metadata={
                    "question": question,
                    "stage": "answer",
                    "question_id": kwargs.get("question_id", None),
                },
            )
            res = await self.answer_chain.ainvoke(
                {
                    "question": question,
                    "document": f"# Chunk ({i}/{len(document.chunks)})\n\n" + chunk["content"],
                    "last_scratchpad": "",
                },
                config={"callbacks": [handler]},
            )
            if res.found_answer:
                logger.info(f"Found answer in chunk {i}, returning answer")
                break
        return res


class RLMSystem(System):
    def _setup_chains(self):
        pass

    async def run(self, question: str, documents: list[Document], *args, **kwargs) -> str:
        metadata = {"question": question}
        rlm = RLM_REPL(model="gpt-5", recursive_model="gpt-5", enable_logging=True, max_iterations=10)
        context = "\n".join([document.content for document in documents])
        res = rlm.completion(context=context, query=question)
        return res, metadata
