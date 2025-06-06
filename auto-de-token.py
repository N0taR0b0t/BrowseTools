import os
import re
import json
import tiktoken
import requests
from datetime import datetime


class ChunkAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.model = "gpt-4o-mini"
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.max_tokens = 900

        self.system_prompt = (
            "You are an assistant that helps filter useful content from a document.\n"
            "You will be given a set of text chunks, numbered from 1 to N.\n"
            "Your task is to read all of them and decide where the useful content ends.\n"
            "Return a single integer between 0 and N:\n"
            "- 0 means the entire document is useless.\n"
            "- Any other number N means only the first N chunks contain useful information."
        )

    def load_text_chunks(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        tokens = self.tokenizer.encode(text)

        chunks = []
        i = 0
        while i < len(tokens):
            end = min(i + self.max_tokens, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            matches = list(re.finditer(r"(?<!\b[A-Z])(?<!\d)\.(?=\s[A-Z])", chunk_text))
            if matches:
                best_match = None
                max_len = 0
                for m in matches:
                    candidate_text = chunk_text[:m.end()]
                    candidate_tokens = self.tokenizer.encode(candidate_text)
                    if len(candidate_tokens) <= self.max_tokens and len(candidate_tokens) > max_len:
                        best_match = candidate_text
                        max_len = len(candidate_tokens)
                if best_match:
                    chunk_text = best_match
                    chunk_tokens = self.tokenizer.encode(chunk_text)

            if chunks:
                previous_chunk = chunks[-1]
                combined = previous_chunk + " " + chunk_text
                combined_tokens = self.tokenizer.encode(combined)
                if len(combined_tokens) <= self.max_tokens:
                    chunks[-1] = combined
                    i += len(chunk_tokens)
                    continue

            chunks.append(chunk_text)
            i += len(chunk_tokens)

        return chunks

    def call_openai(self, messages, temperature=0, max_tokens=5):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": payload,
            "status": response.status_code
        }

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            log_entry["assistant"] = content
            with open("llm_chunk_log.json", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
            return content.strip()
        else:
            with open("llm_chunk_log.json", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
            return f"Error: {response.status_code}"

    def decide_cutoff(self, chunks):
        chunk_texts = "\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
        user_prompt = (
            f"""Here are the chunks:\n{chunk_texts}\n\n"""
            "Your job is to determine where the chunks stop providing useful information. "
            "Return a JSON object like this:\n"
            '{ "cutoff": <number>, "justification": "<reasoning>" }\n\n'
            "- `cutoff` must be an integer between 0 and the number of chunks.\n"
            "- `justification` must explain briefly why you selected that cutoff."
            "You must choose the cutoff point such that no unique and useful information appears after it."
            "If a chunk near the end contains unique and useful information but a chunk before it is repetitive, do not hestitate to choose the later chunk."
        )

        reasoning_prompt = (
            "You are a careful assistant analyzing the usefulness of content chunks. "
            "Think step by step before deciding. Evaluate how each chunk contributes. "
            "Do not rush to answer."
        )

        messages = [
            {"role": "system", "content": reasoning_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_response = self.call_openai(messages, max_tokens=500)

        try:
            result = json.loads(raw_response)
            if not isinstance(result, dict) or "cutoff" not in result or "justification" not in result:
                raise ValueError("Missing expected keys in response.")
            return result
        except Exception as e:
            return {
                "cutoff": 0,
                "justification": f"Failed to parse LLM response. Error: {str(e)}. Raw response: {raw_response}"
            }

if __name__ == "__main__":
    analyzer = ChunkAnalyzer()
    chunks = analyzer.load_text_chunks("extracted_content.txt")
    result = analyzer.decide_cutoff(chunks)

    print("\nLLM Cutoff Result:", result)