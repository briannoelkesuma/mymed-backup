import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Optional, Tuple, List
import re

import logging

load_dotenv()

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

logger = logging.getLogger(__name__)
class LLMJudge:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o"
        self.temperature = 0

    async def evaluate_context_relevance(self, question: str, context: str) -> Tuple[float, str]:
        """ Evaluates how relevant the retrieved context is to the query. """
        try:
            prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
            response = await self._get_completion(prompt)
            return self._parse_feedback(response)
        except Exception as e:
            logger.error(f"Context relevance evaluation failed: {e}")
            return 0.0, str(e)

    async def evaluate_groundedness(self, question: str, context: str, answer: str) -> Tuple[float, str]:
        """ Evaluates how well the answer is supported by the context. """
        try:
            prompt = GROUNDEDNESS_PROMPT.format(question=question, context=context, answer=answer)
            response = await self._get_completion(prompt)
            return self._parse_feedback(response)
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
            return 0.0, str(e)

    async def evaluate_answer_relevance(self, question: str, answer: str) -> Tuple[float, str]:
        """ Evaluates how relevant the answer is to the question. """
        try:
            prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
            response = await self._get_completion(prompt)
            return self._parse_feedback(response)
        except Exception as e:
            logger.error(f"Answer relevance evaluation failed: {e}")
            return 0.0, str(e)

    async def _get_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def _parse_feedback(self, feedback: str) -> Tuple[float, str]:
        try:
            rating_match = re.search(r'Total rating:\s*(\d+(?:\.\d+)?)', feedback)
            eval_match = re.search(r'Evaluation:(.*?)(?:Total rating:|$)', feedback, re.DOTALL)
            
            rating = float(rating_match.group(1)) if rating_match else 0.0
            evaluation = eval_match.group(1).strip() if eval_match else ""
            
            return rating, evaluation
        except Exception as e:
            logger.error(f"Failed to parse feedback: {e}")
            return 0.0, str(e)

CONTEXT_RELEVANCE_PROMPT = """
You will evaluate how relevant the retrieved context is to the user_question.
Rate on a scale of 1 to 4:

1: Not relevant: The retrieved context is unrelated to the query.
2: Partially relevant: The retrieved context includes some related content but is mostly off-topic.
3: Mostly relevant: The retrieved context is helpful but could be more directly related.
4: Highly relevant: The retrieved context is fully relevant and directly answers the query.

Provide your feedback as follows:

Feedback:::
Evaluation: (explain how well the context is relevant to the query)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Question: {question}
Retrieved Context: {context}
"""

GROUNDEDNESS_PROMPT = """
You will evaluate whether the system_answer is supported by the retrieved context.
Rate on a scale of 1 to 4:

1: Not grounded: The system_answer is unrelated to the retrieved context.
2: Weakly grounded: The system_answer has some alignment but contains major unsupported claims.
3: Mostly grounded: The system_answer is mostly supported but contains minor inaccuracies.
4: Fully grounded: The system_answer is fully supported by the retrieved context without any hallucinations.

Provide your feedback as follows:

Feedback:::
Evaluation: (explain how well the answer is supported by the retrieved context)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Question: {question}
Retrieved Context: {context}
Answer: {answer}
"""

ANSWER_RELEVANCE_PROMPT = """
You will evaluate whether the system_answer is relevant to the user_question.
Rate on a scale of 1 to 4:

1: Not relevant: The system_answer does not address the user_question.
2: Partially relevant: The system_answer is somewhat related but misses key aspects.
3: Mostly relevant: The system_answer answers the question but could be improved.
4: Highly relevant: The system_answer fully and accurately addresses the user_question.

Provide your feedback as follows:

Feedback:::
Evaluation: (explain how relevant the answer is to the query)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Question: {question}
Answer: {answer}
"""