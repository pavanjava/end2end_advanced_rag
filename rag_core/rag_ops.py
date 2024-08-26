import litellm
from litellm import completion
from litellm.caching import Cache
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper
from rag_core.hybrid_qdrant_operations import HybridQdrantOperations
from utils.decorators import compute_execution_time
from typing import Union
from dotenv import load_dotenv, find_dotenv
import os
import requests


class RAGOperations:
    response_type = Union[ModelResponse, CustomStreamWrapper]
    _ = load_dotenv(find_dotenv())

    def __init__(self):
        litellm.cache = Cache(type="redis", host="localhost", port="6379")
        self.guard_rails_api_base = os.environ['GUARDRAILS_API_BASE']
        self.qdrant_ops = HybridQdrantOperations()

    @compute_execution_time
    def start_conversation(self, user_query):
        # send request for moderation to check the request health.
        if self._content_moderator(content=user_query):
            prompt = self._create_prompt(user_query=user_query)
            messages = [{
                "role": "system",
                "content": "You are a AI assistant and your role is to answer the USER_QUERY based on the CONTEXT "
                           "provided but not on the prior knowledge you have."
            }, {"role": "user", "content": prompt}]
            _response = self._chat_completions(messages=messages)
            # send response for moderation to check the response health.
            if self._content_moderator(content=_response):
                return _response
            else:
                return {'response': 'Sorry, I Can not process the request !'}
        else:
            return {'response': 'Sorry, I Can not process the request !'}

    def _create_prompt(self, user_query):
        response = self.qdrant_ops.hybrid_search(text=user_query, top_k=5)
        return f'''
        <CONTEXT_BEGIN>
        {response}
        <CONTEXT_END>
        <USER_QUERY_START>
        {user_query}
        <USER_QUERY_END>
        '''

    # acts as pre and post guardrails
    @compute_execution_time
    def _content_moderator(self, content) -> bool:
        payload = {"user_query": [
            {
                "role": "user",
                "content": content
            }
        ]}
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", self.guard_rails_api_base, json=payload, headers=headers)
        if "unsafe" in response.text:
            return False
        else:
            return True

    @compute_execution_time
    def _chat_completions(self, messages) -> response_type:
        # Make completion calls
        # message : [{"role": "user", "content": "Tell me a joke."}] format
        response1 = completion(
            model="ollama/llama3.1",
            messages=messages,
            cache={"no-cache": False, "no-store": False}
        )

        print(response1.choices[0].message.content)
        return response1.choices[0].message.content
