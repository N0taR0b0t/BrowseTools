import os
import re
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
        self.system_prompt = f"""You are a web browsing assistant. The current date is {date}.

You can use the following tools:

- to=web search("your search terms here") — Search Google for information.
- to=visit site("URL here") — Visit a specific website to read its content.
- to=exit — Exit the tool loop and provide a final summary.

**Your responsibilities:**

1. First, analyze the user’s query and describe your plan.
2. Use tools when needed to search or visit websites.
3. After gathering sufficient information to answer the query thoroughly, you **must conclude** by replying with:

    to=exit

This signals that you’re done using tools and are ready to provide a final summary and analysis.

**Important:** Do not wait for confirmation. If you have enough data, end the loop yourself with `to=exit` and summarize everything you've learned."""


    def call_openai(self, messages):
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0.7, "max_tokens": 2000}
        )
        return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else f"Error: {response.status_code}"

    def detect_tool_use(self, text):
        if re.search(r'to=exit|exit tool loop|provide final summary', text, re.I):
            return "exit", None

        # Detect web search queries
        if match := re.search(
            #r'to=web\s+search\("([^"]+)"\)|search for[:\s]+["\']?([^"\'\n]+)', text, re.I
            r'to=web\s+search\("([^"]+)"\)', text, re.I
        ):
            return "search", match.group(1) or match.group(2)

        # Detect valid URLs for visiting
        if match := re.search(
            r'to=visit\s+site\("((?:https?|ftp)://[^\s"\')]+)"\)'
            r'|visit[:\s]+["\']?((?:https?|ftp)://[^\s"\')]+)["\']?',
            #r'|open url[:\s]+["\']?((?:https?|ftp)://[^\s"\')]+)["\']?',
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

    def run_conversation(self, user_query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {user_query}\n\nFirst, analyze this query and explain your research approach."}
        ]

        print(f"User: {user_query}\n")
        for _ in range(10):
            if (context := self.get_context_summary()):
                messages.insert(-1, {"role": "system", "content": f"CONTEXT MEMORY: {context}"})

            response = self.call_openai(messages)
            print(f"Assistant: {response}\n")
            messages.append({"role": "assistant", "content": response})
            tool_type, param = self.detect_tool_use(response)

            if tool_type == "exit":
                final = f"Based on gathered information, summarize the query: {user_query}\n{self.get_context_summary()}"
                messages.append({"role": "user", "content": final})
                print(f"Final Summary: {self.call_openai(messages)}")
                break
            elif tool_type == "search":
                results = self.search_google(param)
                formatted = self.format_serp_results(results)
                self.search_results_history.append({'query': param, 'results': formatted})
                messages.append({"role": "user", "content": f"Search results for '{param}':\n{formatted}"})
            elif tool_type == "visit":
                content = self.visit_website(param)
                messages.append({"role": "user", "content": f"Content from {param}:\n{content}"})
            else:
                messages.append({"role": "user", "content": "Continue your analysis or use tools as needed."})