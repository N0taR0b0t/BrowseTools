import logging
import re

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