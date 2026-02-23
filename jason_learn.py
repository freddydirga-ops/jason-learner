#!/usr/bin/env python3
"""
Jason Autonomous Learning System v3.0 â Cloud Edition
- Uses SearXNG (Northflank) for search
- Uses Qdrant FastEmbed for embeddings (no Ollama needed)
- Stores vectors in Qdrant (Northflank)
- Runs as a Northflank service container

Fred & Jason â February 2026

Usage:
  python3 jason_learn.py run      â start continuous learning loop
  python3 jason_learn.py once     â learn one batch now
  python3 jason_learn.py status   â show learning progress
  python3 jason_learn.py add "topic" â add a topic to the queue
"""

import sys, json, time, os, hashlib, urllib.request, urllib.parse, re
from datetime import datetime
from html.parser import HTMLParser

# ââ Config ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Use env vars for service URLs (Northflank internal networking)
SEARXNG    = os.environ.get("SEARXNG_URL", "http://jason-searxng:8080")
QDRANT     = os.environ.get("QDRANT_URL", "http://jason-qdrant:6333")
COLLECTION = "jason_knowledge"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FastEmbed default
VECTOR_SIZE = 384  # MiniLM-L6-v2 output dimension

# Cloud-friendly paths (inside container)
DATA_DIR   = os.environ.get("DATA_DIR", "/data")
STATE_FILE = os.path.join(DATA_DIR, "jason_learn_state.json")
LOG_FILE   = os.path.join(DATA_DIR, "jason_learn.log")

HEADERS    = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
BATCH_SIZE = 3        # topics per learning session
SLEEP_BETWEEN = 3600  # 1 hour between batches

# ââ FastEmbed singleton ââââââââââââââââââââââââââââââââââââââââââââââââââââââ
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
        log(f"â FastEmbed loaded: {EMBED_MODEL}")
    return _embedder


# ââ Master Knowledge Curriculum âââââââââââââââââââââââââââââââââââââââââââââââ
CURRICULUM = {

    "programming_languages": [
        "Python advanced features async await generators",
        "Swift concurrency actors async await",
        "JavaScript TypeScript advanced patterns",
        "Rust memory safety ownership borrowing",
        "Go goroutines channels concurrency",
        "C systems programming memory management",
        "SQL advanced queries optimization indexing",
        "Bash shell scripting advanced techniques",
        "WebAssembly WASM browser native performance",
        "Solidity smart contract programming",
    ],

    "systems_and_os": [
        "Linux kernel architecture internals",
        "macOS system programming APIs",
        "Docker container internals namespaces cgroups",
        "Kubernetes orchestration architecture",
        "Virtual machines hypervisors KVM QEMU",
        "File systems ext4 APFS ZFS internals",
        "Process scheduling memory management OS",
        "Unix signals IPC pipes sockets",
        "System calls kernel interface programming",
        "ARM64 Apple Silicon architecture optimization",
    ],

    "networking": [
        "TCP IP networking stack internals",
        "HTTP HTTP2 HTTP3 QUIC protocols",
        "WebSocket real-time bidirectional communication",
        "DNS resolution CDN architecture",
        "VPN tunneling WireGuard OpenVPN",
        "Network security firewall intrusion detection",
        "Load balancing reverse proxy nginx",
        "gRPC protocol buffers microservices",
        "GraphQL API design subscriptions",
        "Network packet analysis Wireshark",
        "IPv6 transition mechanisms",
        "BGP routing internet infrastructure",
    ],

    "apis_and_web": [
        "REST API design best practices authentication",
        "OAuth2 OpenID Connect JWT authentication",
        "API rate limiting throttling caching strategies",
        "Webhook event-driven architecture patterns",
        "OpenAPI Swagger specification documentation",
        "API gateway patterns microservices",
        "Server sent events streaming APIs",
        "Web scraping automation BeautifulSoup Playwright",
        "Browser automation Selenium Puppeteer",
        "FastAPI async Python web framework",
    ],

    "databases": [
        "PostgreSQL advanced features JSONB full text search",
        "Redis caching pub sub data structures",
        "Vector databases Qdrant Pinecone embeddings",
        "SQLite embedded database optimization",
        "MongoDB document database aggregation",
        "Database sharding replication consistency",
        "Time series databases InfluxDB",
        "Graph databases Neo4j relationship queries",
        "Database indexing B-tree hash indexes",
        "ACID transactions isolation levels",
    ],

    "blockchain_trading": [
        "Bitcoin blockchain architecture consensus",
        "Ethereum EVM smart contracts Solidity",
        "Binance API spot trading Python integration",
        "Binance WebSocket real-time price streams",
        "Crypto trading strategies RSI MACD bollinger bands",
        "Algorithmic trading backtesting Python",
        "DeFi decentralized finance protocols",
        "Crypto portfolio risk management",
        "Binance testnet paper trading setup",
        "Technical analysis candlestick patterns crypto",
        "Zero knowledge proofs ZK-SNARKs privacy",
        "Merkle trees blockchain data structures",
        "Wallet key management HD wallets BIP39",
    ],

    "cryptography_security": [
        "AES symmetric encryption modes GCM CBC",
        "TLS SSL certificate PKI infrastructure",
        "Password hashing bcrypt argon2 scrypt",
        "Secure random number generation entropy",
        "End to end encryption Signal protocol",
        "OWASP top 10 web vulnerabilities",
        "Secure coding practices memory safety",
    ],

    "ai_ml": [
        "Large language models transformer architecture",
        "Fine tuning LLMs LoRA QLoRA techniques",
        "RAG retrieval augmented generation systems",
        "Vector embeddings semantic search similarity",
        "Ollama local LLM deployment optimization",
        "LangChain agent frameworks tool use",
        "Prompt engineering techniques chain of thought",
        "Model quantization GGUF GGML efficiency",
        "Multi-agent AI systems coordination",
        "Hugging Face transformers model hub",
        "qwen3 model architecture capabilities",
    ],

    "devops_infrastructure": [
        "CI CD pipelines GitHub Actions automation",
        "Infrastructure as code Terraform Ansible",
        "Monitoring observability Prometheus Grafana",
        "Container security scanning hardening",
        "Cloud providers AWS GCP Azure comparison",
        "Serverless functions Lambda architecture",
        "Secret management Vault HashiCorp",
        "n8n workflow automation advanced nodes",
        "Message queues RabbitMQ Kafka streaming",
        "Cloudflare Pages deployment CDN",
    ],

    "automation_tools": [
        "MCP Model Context Protocol tool development",
        "Claude API anthropic tool use function calling",
        "Browser automation headless Chrome Playwright",
        "cron scheduling launchd macOS automation",
        "Python subprocess system automation",
        "AppleScript JXA macOS automation",
        "Telegram bot API automation messaging",
        "n8n custom nodes development",
    ],

    "hotel_chatbot_twace": [
        "Hotel chatbot best practices guest experience",
        "Hotel property management system PMS integration",
        "Chatbot NLP intent recognition hospitality",
        "WhatsApp Business API chatbot integration",
        "Hotel booking pre-stay communication automation",
        "Bali tourism hospitality market overview",
        "Customer service chatbot training data",
        "LLM chatbot knowledge base hotel FAQ",
    ],

    "hardware_iot": [
        "Apple Silicon M2 M4 architecture optimization",
        "GPU CUDA parallel computing inference",
        "Bluetooth BLE protocol stack",
        "Camera interfaces AVFoundation macOS",
        "Audio processing CoreAudio pipelines",
    ],
}


# ââ Logging âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except:
        pass


# ââ State Management ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    queue = []
    for category, topics in CURRICULUM.items():
        for topic in topics:
            queue.append({"topic": topic, "category": category, "learned": False})
    return {
        "queue": queue,
        "total_learned": 0,
        "total_chunks": 0,
        "last_run": None,
        "started": datetime.now().isoformat(),
        "version": "3.0-cloud"
    }


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ââ HTTP Helpers ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def http_post(url, data, timeout=60):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_put(url, data, timeout=30):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"}, method="PUT")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ââ HTML Text Extractor âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header", "aside", "noscript"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header", "aside", "noscript"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            clean = data.strip()
            if len(clean) > 30:
                self.text.append(clean)


def extract_text(html):
    p = TextExtractor()
    p.feed(html)
    text = " ".join(p.text)
    text = re.sub(r'\s+', ' ', text)
    return text[:6000]


# ââ Search via SearXNG (Northflank service) ââââââââââââââââââââââââââââââââââ
def search(query, max_results=3):
    """Use SearXNG on Northflank â no rate limits, no blocking."""
    try:
        q    = urllib.parse.quote_plus(query)
        data = json.loads(http_get(f"{SEARXNG}/search?q={q}&format=json", timeout=12))
        results = data.get("results", [])[:max_results]
        return [{"url": r.get("url", ""), "snippet": r.get("content", "")} for r in results]
    except Exception as e:
        log(f"   â SearXNG search failed: {e}")
        return []


# ââ Qdrant Storage with FastEmbed ââââââââââââââââââââââââââââââââââââââââââââ
def ensure_collection():
    try:
        http_get(f"{QDRANT}/collections/{COLLECTION}")
        log(f"â Qdrant collection '{COLLECTION}' exists")
    except:
        http_put(f"{QDRANT}/collections/{COLLECTION}",
                 {"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}})
        log(f"â Qdrant collection '{COLLECTION}' created (dim={VECTOR_SIZE})")


def embed(text):
    """Generate embedding using FastEmbed (local, CPU-only, no API needed)."""
    embedder = get_embedder()
    embeddings = list(embedder.embed([text]))
    return embeddings[0].tolist()


def store_chunks(text, topic, category, source_url):
    chunks = [text[i:i+500] for i in range(0, len(text), 380)]
    stored = 0
    for chunk in chunks[:8]:
        if len(chunk) < 80:
            continue
        try:
            v   = embed(chunk)
            mid = int(hashlib.md5(f"{chunk}{time.time()}".encode()).hexdigest()[:8], 16)
            http_put(f"{QDRANT}/collections/{COLLECTION}/points", {
                "points": [{"id": mid, "vector": v, "payload": {
                    "text":     chunk,
                    "topic":    topic,
                    "category": category,
                    "source":   source_url,
                    "date":     datetime.now().strftime("%Y-%m-%d")
                }}]
            })
            stored += 1
            time.sleep(0.15)
        except Exception as e:
            log(f"   â  Embed/store error: {e}")
    return stored


# ââ Learn One Topic âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def learn_topic(topic, category):
    log(f"ð Learning: [{category}] {topic}")
    total_chunks = 0
    results = search(topic, max_results=3)

    if not results:
        log(f"   â No search results")
        return 0

    for r in results[:2]:
        url = r["url"]
        if not url.startswith("http"):
            url = "https://" + url
        try:
            html   = http_get(url, timeout=12)
            text   = extract_text(html)
            if len(text) > 200:
                chunks = store_chunks(text, topic, category, url)
                total_chunks += chunks
                log(f"   â¦ {url[:65]} â {chunks} chunks")
            time.sleep(1)
        except Exception as e:
            log(f"   â  Skipped {url[:55]}: {e}")

    return total_chunks


# ââ Run One Batch âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def run_batch(state):
    pending = [t for t in state["queue"] if not t["learned"]]
    if not pending:
        log("ð All topics learned! Resetting queue for next cycle...")
        for t in state["queue"]:
            t["learned"] = False
        pending = state["queue"]

    batch = pending[:BATCH_SIZE]
    log(f"\nð Batch: {len(batch)} topics | {len(pending)} remaining\n")

    for item in batch:
        chunks = learn_topic(item["topic"], item["category"])
        item["learned"]       = True
        state["total_learned"] += 1
        state["total_chunks"]  += chunks
        state["last_run"]      = datetime.now().isoformat()
        save_state(state)
        time.sleep(2)

    log(f"\nâ Batch done. Learned: {state['total_learned']} topics | {state['total_chunks']} chunks total\n")


# ââ Status ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def show_status():
    state   = load_state()
    pending = [t for t in state["queue"] if not t["learned"]]
    done    = [t for t in state["queue"] if t["learned"]]
    total   = len(state["queue"])

    print(f"\nð§  Jason Knowledge System v3.0 (Cloud Edition)")
    print(f"{'â'*50}")
    print(f"  SearXNG:         {SEARXNG}")
    print(f"  Qdrant:          {QDRANT}")
    print(f"  Embed model:     {EMBED_MODEL}")
    print(f"  Vector dim:      {VECTOR_SIZE}")
    print(f"  Total topics:    {total}")
    print(f"  Learned:         {len(done)}")
    print(f"  Remaining:       {len(pending)}")
    print(f"  Total chunks:    {state['total_chunks']}")
    print(f"  Last run:        {state.get('last_run', 'never')}")
    print(f"\n  Categories:")
    cats = {}
    for t in state["queue"]:
        c = t["category"]
        cats[c] = cats.get(c, {"done": 0, "total": 0})
        cats[c]["total"] += 1
        if t["learned"]:
            cats[c]["done"] += 1
    for cat, counts in sorted(cats.items()):
        bar = "â" * counts["done"] + "â" * (counts["total"] - counts["done"])
        print(f"    {cat:<35} {bar} {counts['done']}/{counts['total']}")
    print()


# ââ Main ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
if __name__ == "__main__":
    log(f"ð¤ Jason Learning System v3.0 â Cloud Edition")
    log(f"   SearXNG: {SEARXNG}")
    log(f"   Qdrant:  {QDRANT}")
    log(f"   Embed:   {EMBED_MODEL} (dim={VECTOR_SIZE})")

    ensure_collection()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"  # default to "run" in container

    if cmd == "status":
        show_status()

    elif cmd == "add" and len(sys.argv) > 2:
        topic = " ".join(sys.argv[2:])
        state = load_state()
        state["queue"].append({"topic": topic, "category": "custom", "learned": False})
        save_state(state)
        log(f"â Added topic: {topic}")

    elif cmd == "once":
        state = load_state()
        run_batch(state)

    elif cmd == "run":
        log(f"   Batch: {BATCH_SIZE} topics every {SLEEP_BETWEEN//60} minutes")
        state = load_state()
        show_status()
        while True:
            try:
                run_batch(state)
                log(f"ð¤ Sleeping {SLEEP_BETWEEN//60} minutes...")
                time.sleep(SLEEP_BETWEEN)
            except KeyboardInterrupt:
                log("â¹ Learning system stopped")
                break
            except Exception as e:
                log(f"â  Error in learning loop: {e}")
                time.sleep(60)

