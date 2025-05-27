import os
import re
import json
import requests
from datetime import datetime
from fetch import fetch_from_zyte, fetch_browser_html
from html_utils import visible_text


class WebBrowsingAgent:
    # ──────────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        date = datetime.now().strftime("%Y-%m-%d")
        # --- short, explicit prompt ------------------------------------------------
        self.system_prompt = f"""You are a web browsing assistant. Today is {date}.

        You can use these tools, one at a time:
        - to=web search("query")  — get search engine results
        - to=visit site("URL")  — view a website's contents
        - to=exit  — end tool use and return to conversation

        Your role is to assist naturally and intelligently.

        • When you're done gathering information, use to=exit. Then summarize what you found before continuing the conversation.
        • During tool use, focus only on searching and visiting—resume chatting only after to=exit.
        • IMPORTANT: Once you issue `to=web search(...)`, you will be stuck in the browse loop until you use `to=exit`. Do NOT chat again until you do this.

        Use tools only when necessary, but don’t hesitate when the task calls for them."""

        # ──────────────────────────────────────────────────────────────────────────
        self.visited_articles = []
        self.search_results_history = []

    # ──────────────────────────────────────────────────────────────────────────────
    # OpenAI wrapper
    # ──────────────────────────────────────────────────────────────────────────────
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
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        # simple logging
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": payload,
            "status": response.status_code
        }
        if response.status_code == 200:
            ans = response.json()["choices"][0]["message"]["content"]
            log_entry["assistant"] = ans
            with open("llm_master_log.json", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
            return ans
        with open("llm_master_log.json", "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
        return f"Error: {response.status_code}"

    # ──────────────────────────────────────────────────────────────────────────────
    # Tool-use detection
    # ──────────────────────────────────────────────────────────────────────────────
    def detect_tool_use(self, text):
        if re.search(r'\bto=exit\b', text, re.I):
            return "exit", None
        if m := re.search(r'to=web\s+search\("([^"]+)"\)', text, re.I):
            return "search", m.group(1)
        if m := re.search(
            r'to=visit\s+site\("((?:https?|ftp)://[^\s"\')]+)"\)', text, re.I
        ):
            return "visit", m.group(1)
        return None, None

    # ──────────────────────────────────────────────────────────────────────────────
    # Browsing helpers
    # ──────────────────────────────────────────────────────────────────────────────
    def search_google(self, query):
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return fetch_from_zyte(url)

    def format_serp_results(self, serp_data):
        if not serp_data or 'organicResults' not in serp_data:
            return "No search results found."
        return "\n\n".join(
            f"{i+1}. **{r.get('title','No title')}**\n   URL: {r.get('url','')}\n   {r.get('snippet','')}"
            for i, r in enumerate(serp_data['organicResults'][:5])
        )

    def visit_website(self, url):
        html = fetch_browser_html(url)
        if not html:
            return "Failed to fetch website content."
        blocks = visible_text(html)

        output, limit, used = [], 650, 0
        for block in blocks:
            words = block.split()
            if used >= limit:
                break
            take = min(len(words), limit - used)
            segment = " ".join(words[:take])
            output.append(segment)
            used += take
        text = "\n\n".join(output)
        self.visited_articles.append(
            {"url": url, "content": text, "timestamp": datetime.now().isoformat()}
        )
        return text

    def get_context_summary(self):
        if not self.visited_articles:
            return ""
        header = "\n\n--- CONTEXT: Visited Articles ---\n"
        body = ""
        for i, art in enumerate(self.visited_articles, 1):
            body += f"\n{i}. {art['url']}\n{art['content']}\n"
        return header + body + "\n--- END CONTEXT ---\n"

    # ──────────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────────
    def run_conversation(self, first_user_query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": first_user_query}
        ]

        tool_loop_active = False

        while True:
            if tool_loop_active and (ctx := self.get_context_summary()):
                # pass context only while in tool loop
                messages.insert(-1, {"role": "system", "content": f"CONTEXT:{ctx}"})

            assistant_reply = self.call_openai(messages)
            print(f"Assistant: {assistant_reply}\n")

            tool_type, param = self.detect_tool_use(assistant_reply)

            # Enforce tool loop rules
            if tool_loop_active and tool_type not in ("visit", "exit"):
                meta_warning = (
                    "**Warning:** You initiated a tool-use sequence with `to=web search(...)`. "
                    "You must now follow up with `to=visit site(...)` to read from a result, "
                    "or `to=exit` to end tool use and summarize before continuing the conversation."
                )
                messages.append({"role": "user", "content": meta_warning})
                continue  # loop back without accepting this assistant reply

            # Accept reply
            messages.append({"role": "assistant", "content": assistant_reply})


            tool_type, param = self.detect_tool_use(assistant_reply)

            # ── handle tool instructions ─────────────────────────────────────────
            if tool_type == "search":
                tool_loop_active = True
                serp = self.search_google(param)
                formatted = self.format_serp_results(serp)
                self.search_results_history.append({"query": param, "results": formatted})
                messages.append({"role": "user",
                                 "content": f"Search results for '{param}':\n{formatted}"})
                continue

            if tool_type == "visit":
                tool_loop_active = True
                page_text = self.visit_website(param)
                messages.append({"role": "user",
                                 "content": f"Content from {param}:\n{page_text}"})
                continue

            if tool_type == "exit":
                tool_loop_active = False
                # Ask the model to summarise before returning to chat mode
                messages.append({"role": "user",
                                 "content": "Please provide a concise summary of what you learned."})
                summary = self.call_openai(messages)
                print(f"\n📄 Summary: {summary}\n")
                messages.append({"role": "assistant", "content": summary})
                # after summary, fall through to normal chat
                continue

            # ── regular chat turn (no tool call) ────────────────────────────────
            user_input = input("👤 You: ")
            messages.append({"role": "user", "content": user_input})