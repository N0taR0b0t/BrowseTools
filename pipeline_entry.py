import json
from datetime import datetime
from agent import WebBrowsingAgent

def main():
    try:
        agent = WebBrowsingAgent()
        search_term = input("Enter your search term: ")

        print(f"\n🤖 Web Browsing Agent (Current date: {datetime.now().strftime('%Y-%m-%d')})\n")
        print("=" * 50)

        agent.run_conversation(search_term)

        session_data = {
            'search_term': search_term,
            'visited_articles': agent.visited_articles,
            'search_history': agent.search_results_history,
            'timestamp': datetime.now().isoformat()
        }
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"\n📁 Session data saved to: {filename}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()