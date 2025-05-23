import os
import json
import re
import requests
import shelve
from pathlib import Path

# SERP functionality
def fetch_from_zyte(url):
    cache_file = "SERP_cache"
    with shelve.open(cache_file) as cache:
        if url in cache:
            print("Loaded from cache.")
            return cache[url]
        else:
            print("Fetching from Zyte API...")
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

def search_google(query):
    """Search Google and return SERP results"""
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    return fetch_from_zyte(url)

# GPT-4o-mini integration
class WebBrowsingAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.system_prompt = """You are a web browsing assistant. You can search the web to find information.

When you need to search for information, use the format:
to=web search("your search terms here")

For example:
- to=web search("Tesla stock price today")
- to=web search("latest AI news 2025")

After receiving search results, you can analyze them and suggest which links to visit next.
Be concise and helpful in your responses."""

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
            "max_tokens": 1000
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
        """Detect if the response contains a tool use command"""
        pattern = r'to=web\s+search\("([^"]+)"\)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)  # Return the search query
        return None

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

    def run_conversation(self, user_query):
        """Run a conversation with tool use capability"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        print(f"User: {user_query}\n")

        # Get initial response from GPT-4o-mini
        ai_response = self.call_openai(messages)
        print(f"Assistant: {ai_response}\n")

        # Check if AI wants to use web search
        search_query = self.detect_tool_use(ai_response)

        if search_query:
            print(f"🔍 Executing search: '{search_query}'\n")

            # Perform the search
            serp_results = search_google(search_query)

            if serp_results:
                # Format results for AI
                formatted_results = self.format_serp_results(serp_results)

                # Add search results to conversation
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({"role": "user", "content": f"Here are the search results:\n\n{formatted_results}\n\nPlease analyze these results and suggest which link would be most relevant to visit first."})

                # Get AI's analysis of search results
                final_response = self.call_openai(messages)
                print(f"Assistant (after search): {final_response}")

                # Save raw SERP data for inspection
                output_path = Path("latest_serp_results.json")
                output_path.write_text(json.dumps(serp_results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"\n📁 Raw SERP data saved to: {output_path}")

            else:
                print("❌ No search results returned from Zyte API")

        return ai_response

# Main function
def main():
    try:
        agent = WebBrowsingAgent()
        
        # Get user input for search term
        search_term = input("Enter your search term: ")
        
        print("\n🤖 Web Browsing Agent\n")
        print("=" * 50)
        
        agent.run_conversation(search_term)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Show the code structure
print("\nScript has been modified to accept user input.")