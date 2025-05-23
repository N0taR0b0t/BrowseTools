import os
import json
import re
import requests
import shelve
from pathlib import Path
from datetime import datetime
import logging
from bs4 import BeautifulSoup, Tag

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants for text extraction
DROP_TAGS = {"script", "style", "noscript", "template", "svg", "meta", "link"}
SEMANTIC_TAGS = ["main", "article", "section"]
JUNK_CONTAINER_TAGS = {"nav", "header", "footer", "aside", "form"}
BOILERPLATE_PATTERNS = [
    re.compile(r"\bcookie(s| consent)\b", re.I),
    re.compile(r"\bprivacy\b", re.I),
    re.compile(r"\bterms\b", re.I),
    re.compile(r"\bsign\s*in\b", re.I),
]
MIN_WORDS = 15

def fetch_from_zyte(url):
    """Fetch SERP data from Zyte API with caching"""
    cache_file = "SERP_cache"
    with shelve.open(cache_file) as cache:
        if url in cache:
            print("Loaded SERP from cache.")
            return cache[url]
        else:
            print("Fetching SERP from Zyte API...")
            api_response = requests.post(
                "https://api.zyte.com/v1/extract",
                auth=(os.getenv("ZYTE_API_KEY"), ""),
                json={
                    "url": url,
                    "serp": True,
                    "serpOptions": {"extractFrom": "browserHtml"},
                },
            )
            data = api_response.json().get("serp")
            if data is not None:
                cache[url] = data
                return data

def fetch_browser_html(url):
    """Fetch browser-rendered HTML via Zyte API with caching"""
    CACHE_FILE = "zyte_browser_html_cache"
    with shelve.open(CACHE_FILE) as cache:
        if url in cache:
            logger.info("Loaded browser HTML from cache.")
            return cache[url]
        else:
            logger.info("Fetching browser HTML from Zyte API...")
            response = requests.post(
                "https://api.zyte.com/v1/extract",
                auth=(os.getenv("ZYTE_API_KEY"), ""),
                json={"url": url, "browserHtml": True},
            )
            if response.status_code != 200:
                logger.error(f"Failed to fetch HTML: {response.status_code} - {response.text}")
                return None

            browser_html = response.json().get("browserHtml")
            if browser_html is not None:
                cache[url] = browser_html
                return browser_html

def _clean_text(el: Tag) -> str:
    """Clean text from HTML element"""
    txt = el.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\s+([,.!?;:])", r"\1", txt)
    return txt

def _is_trivial(text: str) -> bool:
    """Check if text is trivial/boilerplate"""
    if len(text.split()) < MIN_WORDS:
        return True
    if any(p.search(text) for p in BOILERPLATE_PATTERNS):
        total = len(text.split())
        matches = sum(len(p.findall(text)) for p in BOILERPLATE_PATTERNS)
        if matches / max(total, 1) > 0.6:
            return True
    return False

def _jaccard(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings"""
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def _deduplicate(blocks: list[str], thresh: float = 0.85) -> list[str]:
    """Remove duplicate text blocks"""
    unique = []
    for blk in blocks:
        if all(_jaccard(blk, keep) < thresh for keep in unique):
            unique.append(blk)
    return unique

def _extract_segments(parent: Tag) -> list[str]:
    """Extract text segments from HTML parent element"""
    segments = []
    for child in parent.find_all(
        ["p", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"], recursive=False
    ):
        if isinstance(child, Tag):
            txt = _clean_text(child)
            if not _is_trivial(txt):
                segments.append(txt)
    return segments

def visible_text(html: str) -> list[str]:
    """Extract visible text blocks from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(lambda t: t.name in DROP_TAGS):
        el.decompose()

    blocks: list[str] = []

    # First try semantic tags
    for tag_name in SEMANTIC_TAGS:
        for parent in soup.find_all(tag_name):
            if parent.find_parent(lambda t: t.name in JUNK_CONTAINER_TAGS):
                continue
            blocks.extend(_extract_segments(parent))

    # Fallback to divs if no semantic content found
    if not blocks:
        for div in soup.find_all("div"):
            if div.find_parent(lambda t: t.name in JUNK_CONTAINER_TAGS):
                continue
            txt = _clean_text(div)
            if not _is_trivial(txt):
                blocks.append(txt)

    return _deduplicate(blocks)

class WebBrowsingAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Context memory for visited articles
        self.visited_articles = []
        self.search_results_history = []

        # Include current date in system prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = f"""You are a web browsing assistant. The current date is {current_date}.

Available tools:
- to=web search("your search terms here") - Search Google for information
- to=visit site("URL here") - Visit a specific website to read its content
- to=exit - Exit the tool loop and provide a final summary

IMPORTANT: Before using any tools, first analyze the user's query and explain your research plan. You can provide information directly if you already know it.

You have complete freedom to:
1. Respond directly without using tools if appropriate
2. Plan your research strategy before searching
3. Decide when to search vs. visit sites
4. Choose when to exit the tool loop

When you do use tools, clearly indicate your intention with the tool command on its own line.

After gathering sufficient information, use "to=exit" to provide a final summary and analysis."""

    def call_openai(self, messages):
        """Call OpenAI API with the conversation"""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    def detect_tool_use(self, text):
        """Detect if the response contains a tool use command with more flexibility"""
        # Check for exit command - allow variations
        exit_pattern = r'to=exit|exit tool loop|provide final summary'
        if re.search(exit_pattern, text, re.IGNORECASE):
            return "exit", None

        # Check for web search with more flexible patterns
        search_pattern = r'to=web\s+search\("([^"]+)"\)|search for[:\s]+["\'"]?([^"\'"\n]+)["\'"]?'
        search_match = re.search(search_pattern, text, re.IGNORECASE)
        if search_match:
            query = search_match.group(1) if search_match.group(1) else search_match.group(2)
            return "search", query.strip()

        # Check for site visit with more flexible patterns
        visit_pattern = r'to=visit\s+site\("([^"]+)"\)|visit[:\s]+["\'"]?([^"\'"\n]+)["\'"]?|open url[:\s]+["\'"]?([^"\'"\n]+)["\'"]?'
        visit_match = re.search(visit_pattern, text, re.IGNORECASE)
        if visit_match:
            url = visit_match.group(1) if visit_match.group(1) else (visit_match.group(2) if visit_match.group(2) else visit_match.group(3))
            return "visit", url.strip()

        return None, None

    def search_google(self, query):
        """Search Google and return SERP results"""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return fetch_from_zyte(url)

    def format_serp_results(self, serp_data):
        """Format SERP results for the AI to understand"""
        if not serp_data or 'organicResults' not in serp_data:
            return "No search results found."

        formatted = "Search Results:\n\n"
        for i, result in enumerate(serp_data['organicResults'][:5], 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', 'No description')

            formatted += f"{i}. **{title}**\n"
            formatted += f"   URL: {url}\n"
            formatted += f"   Description: {snippet}\n\n"

        return formatted

    def visit_website(self, url):
        """Visit a website and extract readable content"""
        print(f"🌐 Visiting: {url}")

        # Check if already visited
        for article in self.visited_articles:
            if article['url'] == url:
                print("Already visited this URL, using cached content.")
                return article['content']

        # Fetch the HTML
        html_content = fetch_browser_html(url)
        if not html_content:
            return "Failed to fetch website content."

        # Extract visible text
        text_blocks = visible_text(html_content)

        # Limit to ~650 words
        word_limit = 650
        words_printed = 0
        output_blocks = []

        for block in text_blocks:
            block_words = block.split()
            remaining = word_limit - words_printed

            if remaining <= 0:
                break

            if len(block_words) > remaining:
                truncated = " ".join(block_words[:remaining + 30])
                period_match = list(re.finditer(r"\. ", truncated))
                if period_match:
                    last_period_idx = period_match[-1].end()
                    trimmed = truncated[:last_period_idx].strip()
                    output_blocks.append(trimmed)
                    words_printed += len(trimmed.split())
                else:
                    chunk = " ".join(block_words[:remaining])
                    output_blocks.append(chunk)
                    words_printed += remaining
                break
            else:
                output_blocks.append(block)
                words_printed += len(block_words)

        content = "\n\n".join(output_blocks)

        # Store in memory
        self.visited_articles.append({
            'url': url,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

        return content

    def get_context_summary(self):
        """Generate a summary of visited articles for context"""
        if not self.visited_articles:
            return ""

        summary = "\n\n--- CONTEXT: Previously Visited Articles ---\n"
        for i, article in enumerate(self.visited_articles, 1):
            summary += f"\nArticle {i}: {article['url']}\n"
            # Include first 100 words as preview
            words = article['content'].split()[:100]
            preview = " ".join(words)
            if len(article['content'].split()) > 100:
                preview += "..."
            summary += f"Preview: {preview}\n"
        summary += "\n--- END CONTEXT ---\n\n"
        return summary

    def run_conversation(self, user_query):
        """Run a conversation with tool use capability and planning phase"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {user_query}\n\nFirst, analyze this query and explain your research approach before using any tools."}
        ]

        print(f"User: {user_query}\n")

        # Tool use loop
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Add context summary if we have visited articles
            context_summary = self.get_context_summary()
            if context_summary and iteration > 1:
                # Insert context before the last user message
                context_message = {"role": "system", "content": f"CONTEXT MEMORY: {context_summary}"}
                messages.insert(-1, context_message)

            # Get response from GPT-4o-mini
            ai_response = self.call_openai(messages)
            print(f"Assistant: {ai_response}\n")

            # Add AI response to conversation
            messages.append({"role": "assistant", "content": ai_response})

            # Check if AI wants to use tools
            tool_type, tool_param = self.detect_tool_use(ai_response)

            if tool_type == "exit":
                print("🏁 Exiting tool loop. Providing final summary...\n")
                # Generate final summary with all context
                final_context = self.get_context_summary()
                final_prompt = f"""Based on all the information gathered from searches and website visits, provide a comprehensive summary and analysis of the user's original query: "{user_query}"

{final_context}

Please synthesize all the information and provide key insights, conclusions, and any relevant details."""

                messages.append({"role": "user", "content": final_prompt})
                final_response = self.call_openai(messages)
                print(f"Final Summary: {final_response}")
                break

            elif tool_type == "search":
                print(f"🔍 Executing search: '{tool_param}'\n")
                serp_results = self.search_google(tool_param)

                if serp_results:
                    formatted_results = self.format_serp_results(serp_results)
                    self.search_results_history.append({
                        'query': tool_param,
                        'results': formatted_results,
                        'timestamp': datetime.now().isoformat()
                    })
                    messages.append({"role": "user", "content": f"Here are the search results for '{tool_param}':\n\n{formatted_results}"})
                else:
                    messages.append({"role": "user", "content": "No search results were found. Please try a different search query or use 'to=exit' to summarize what we know so far."})

            elif tool_type == "visit":
                print(f"🌐 Visiting website: {tool_param}\n")
                website_content = self.visit_website(tool_param)
                messages.append({"role": "user", "content": f"Here is the content from {tool_param}:\n\n{website_content}"})

            else:
                # No tool use detected, but that's okay - the LLM might be providing information
                # Only prompt for tool use if the response seems incomplete
                if "would you like more information" in ai_response.lower() or "i can search" in ai_response.lower():
                    messages.append({"role": "user", "content": "You can continue with your analysis, use search or visit tools if needed, or use 'to=exit' when you've gathered enough information."})
                else:
                    # Let the conversation continue naturally
                    messages.append({"role": "user", "content": "Continue with your analysis. You can use tools if needed or provide more information directly."})

                    if iteration >= max_iterations:
                        print("⚠️ Maximum iterations reached. Providing summary with available information...")

                    return ai_response

def main():
    try:
        agent = WebBrowsingAgent()
        search_term = input("Enter your search term: ")

        print(f"\n🤖 Web Browsing Agent (Current date: {datetime.now().strftime('%Y-%m-%d')})\n")
        print("=" * 50)

        agent.run_conversation(search_term)

        # Save session data
        session_data = {
            'search_term': search_term,
            'visited_articles': agent.visited_articles,
            'search_history': agent.search_results_history,
            'timestamp': datetime.now().isoformat()
        }

        session_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"\n📁 Session data saved to: {session_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()