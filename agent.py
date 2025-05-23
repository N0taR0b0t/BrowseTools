import os
import re
import json
import requests
from datetime import datetime
from fetch import fetch_from_zyte, fetch_browser_html
from html_utils import visible_text

class WebBrowsingAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.visited_articles = []
        self.search_results_history = []

        date = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = f"""You are a helpful and intelligent web browsing assistant. The current date is {date}.

You can use the following tools:

- to=web search("your search terms here") — Search Google for information.
- to=visit site("URL here") — Visit a specific website to read its content.
- to=exit — Exit the tool loop and provide a final summary.

**Your responsibilities:**

1. Respond to the user’s input in a helpful, natural way.
2. If the query is simple, conversational, or doesn't require outside information, respond directly without using tools.
3. Use tools when external information is needed or requested.
4. When using tools, explain your reasoning first in once sentence.
5. Once you have enough information to answer the user's original query, reply with `to=exit`.
6. Once you have used `to=search`, you must use no less than one `to=visit` before using `to=exit`.
7. List your sources and references.


**Important:** You do **not** need to use tools unless the question cannot be answered effectively without them. Keep your responses clear, concise, and appropriate for the query. Avoid overanalyzing greetings or general conversation."""

    import json  # at the top if not already imported

    def call_openai(self, messages):
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        # Add this to print every token being sent
        print("==== PAYLOAD SENT TO OPENAI ====")
        print(json.dumps(payload, indent=2))
        print("================================")

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else f"Error: {response.status_code}"


    def detect_tool_use(self, text):
        if re.search(r'to=exit|exit tool loop|provide final summary', text, re.I):
            return "exit", None

        if match := re.search(r'to=web\s+search\("([^"]+)"\)|search for[:\s]+["\']?([^"\'\n]+)', text, re.I):
            return "search", match.group(1) or match.group(2)

        if match := re.search(
            r'to=visit\s+site\("((?:https?|ftp)://[^\s"\')]+)"\)'
            r'|visit[:\s]+["\']?((?:https?|ftp)://[^\s"\')]+)["\']?'
            r'|open url[:\s]+["\']?((?:https?|ftp)://[^\s"\')]+)["\']?',
            text, re.I
        ):
            return "visit", match.group(1) or match.group(2) or match.group(3)

        return None, None

    def search_google(self, query):
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return fetch_from_zyte(url)

    def format_serp_results(self, serp_data):
        if not serp_data or 'organicResults' not in serp_data:
            return "No search results found."
        return "\n\n".join(
            f"{i + 1}. **{res.get('title', 'No title')}**\n   URL: {res.get('url', '')}\n   Description: {res.get('snippet', '')}"
            for i, res in enumerate(serp_data['organicResults'][:5])
        )

    def visit_website(self, url):
        print(f"🌐 Visiting: {url}")
        for article in self.visited_articles:
            if article['url'] == url:
                return article['content']

        html = fetch_browser_html(url)
        if not html:
            return "Failed to fetch website content."
        blocks = visible_text(html)

        output_blocks, word_limit, words_printed = [], 650, 0
        for block in blocks:
            block_words = block.split()
            remaining = word_limit - words_printed
            if remaining <= 0:
                break
            if len(block_words) > remaining:
                chunk = " ".join(block_words[:remaining + 30])
                period_match = list(re.finditer(r"\. ", chunk))
                trimmed = chunk[:period_match[-1].end()].strip() if period_match else " ".join(block_words[:remaining])
                output_blocks.append(trimmed)
                break
            output_blocks.append(block)
            words_printed += len(block_words)

        content = "\n\n".join(output_blocks)
        self.visited_articles.append({'url': url, 'content': content, 'timestamp': datetime.now().isoformat()})
        return content

    def get_context_summary(self):
        if not self.visited_articles:
            return ""
        summary = "\n\n--- CONTEXT: Previously Visited Articles ---\n"
        for i, article in enumerate(self.visited_articles, 1):
            preview = " ".join(article['content'].split()[:100])
            summary += f"\nArticle {i}: {article['url']}\nPreview: {preview}...\n"
        summary += "\n--- END CONTEXT ---\n"
        return summary

    def run_conversation(self, messages):
        """
        Accepts an external message list (with history) and manages one turn of assistant response.
        If tools are needed, continues tool interaction; otherwise, returns the assistant's response.
        """
        # Prepend system prompt and optional context
        full_messages = [{"role": "system", "content": self.system_prompt}]
        if self.visited_articles:
            full_messages.append({
                "role": "system",
                "content": f"CONTEXT MEMORY: {self.get_context_summary()}"
            })
        full_messages.extend(messages)

        # Call OpenAI with full context
        response = self.call_openai(full_messages)
        print(f"Assistant: {response}\n")

        tool_type, param = self.detect_tool_use(response)
        if tool_type == "exit":
            # User likely wrapped up – provide a summary
            summary_prompt = {
                "role": "user",
                "content": f"Based on gathered information, summarize this session:\n{self.get_context_summary()}"
            }
            full_messages.append({"role": "assistant", "content": response})
            full_messages.append(summary_prompt)
            summary = self.call_openai(full_messages)
            print(f"Final Summary: {summary}")
            return summary

        elif tool_type == "search":
            results = self.search_google(param)
            formatted = self.format_serp_results(results)
            self.search_results_history.append({'query': param, 'results': formatted})
            messages.append({"role": "user", "content": f"Search results for '{param}':\n{formatted}"})
            return self.run_conversation(messages)

        elif tool_type == "visit":
            content = self.visit_website(param)
            messages.append({"role": "user", "content": f"Content from {param}:\n{content}"})
            return self.run_conversation(messages)

        else:
            return response