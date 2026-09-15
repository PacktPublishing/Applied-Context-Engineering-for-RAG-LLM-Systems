"""
RESEARCH ASSISTANT - CAPSTONE PROJECT
Complete context engineering system

Run: python research_assistant.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Protocol
from enum import Enum
from abc import abstractmethod
from collections import defaultdict
import uuid
import re
import json
import statistics

# =============================================================================
# CONTEXT SOURCES
# =============================================================================


class ContextSource(Protocol):
    """Interface for context sources."""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]: ...

    @property
    @abstractmethod
    def source_type(self) -> str: ...


@dataclass
class Document:
    """A document in the store."""

    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentStore:
    """In-memory document store."""

    def __init__(self):
        self.documents: List[Document] = []

    @property
    def source_type(self) -> str:
        return "document_store"

    def add(self, doc: Document) -> None:
        self.documents.append(doc)

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_terms = set(query.lower().split())
        results = []

        for doc in self.documents:
            content_terms = set(doc.content.lower().split())
            overlap = len(query_terms & content_terms)

            if overlap > 0:
                results.append(
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "source": self.source_type,
                        "metadata": doc.metadata,
                        "score": overlap / len(query_terms),
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class KnowledgeBase:
    """Structured knowledge base."""

    def __init__(self):
        self.facts: Dict[str, List[str]] = {}

    @property
    def source_type(self) -> str:
        return "knowledge_base"

    def add_fact(self, topic: str, fact: str) -> None:
        if topic not in self.facts:
            self.facts[topic] = []
        self.facts[topic].append(fact)

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = []
        query_lower = query.lower()

        for topic, facts in self.facts.items():
            if topic.lower() in query_lower or query_lower in topic.lower():
                for i, fact in enumerate(facts):
                    results.append(
                        {
                            "id": f"{topic}_{i}",
                            "content": fact,
                            "source": self.source_type,
                            "metadata": {"topic": topic},
                            "score": 0.9,
                        }
                    )

        return results[:limit]


class ConversationHistory:
    """Conversation history store."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    @property
    def source_type(self) -> str:
        return "conversation"

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        recent = self.messages[-limit * 2 :] if self.messages else []

        return [
            {
                "id": f"msg_{i}",
                "content": f"{msg['role']}: {msg['content']}",
                "source": self.source_type,
                "metadata": {"role": msg["role"]},
                "score": 0.8,
            }
            for i, msg in enumerate(recent)
        ]


# =============================================================================
# CONTEXT PIPELINE
# =============================================================================


@dataclass
class ContextRankerConfig:
    min_score: float = 0.3
    diversity_weight: float = 0.2


class ContextRanker:
    """Rank and filter context."""

    def __init__(self, config: ContextRankerConfig = None):
        self.config = config or ContextRankerConfig()

    def rank(self, documents: List[Dict], query: str) -> List[Dict]:
        filtered = [d for d in documents if d.get("score", 0) >= self.config.min_score]

        seen_sources = set()
        for doc in sorted(filtered, key=lambda x: x["score"], reverse=True):
            source = doc.get("source", "unknown")
            boost = 0.1 if source not in seen_sources else 0
            doc["final_score"] = doc["score"] + boost
            seen_sources.add(source)

        filtered.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return filtered

    def select_top_k(
        self, documents: List[Dict], k: int = 5, max_tokens: int = 10000
    ) -> List[Dict]:
        selected = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = len(doc.get("content", "").split()) * 1.3

            if total_tokens + doc_tokens <= max_tokens:
                selected.append(doc)
                total_tokens += doc_tokens

            if len(selected) >= k:
                break

        return selected


@dataclass
class PipelineMetrics:
    sources_queried: int = 0
    documents_retrieved: int = 0
    documents_selected: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class ContextResult:
    context: str
    sources: List[str]
    metrics: PipelineMetrics
    quality_score: float = 0.0


class ContextPipeline:
    """Complete context retrieval pipeline."""

    def __init__(self):
        self.sources: List[ContextSource] = []
        self.ranker = ContextRanker()

    def add_source(self, source: ContextSource) -> None:
        self.sources.append(source)

    def retrieve(
        self, query: str, max_documents: int = 10, max_tokens: int = 15000
    ) -> ContextResult:
        start_time = datetime.now()

        all_documents = []
        for source in self.sources:
            docs = source.retrieve(query, limit=max_documents)
            all_documents.extend(docs)

        ranked = self.ranker.rank(all_documents, query)
        selected = self.ranker.select_top_k(
            ranked, k=max_documents, max_tokens=max_tokens
        )

        context_parts = []
        sources = set()

        for doc in selected:
            context_parts.append(doc["content"])
            sources.add(doc["source"])

        formatted_context = "\n\n---\n\n".join(context_parts)

        latency = (datetime.now() - start_time).total_seconds() * 1000
        tokens = int(len(formatted_context.split()) * 1.3)

        avg_score = sum(d.get("final_score", 0) for d in selected) / max(
            len(selected), 1
        )

        return ContextResult(
            context=formatted_context,
            sources=list(sources),
            metrics=PipelineMetrics(
                sources_queried=len(self.sources),
                documents_retrieved=len(all_documents),
                documents_selected=len(selected),
                total_tokens=tokens,
                latency_ms=latency,
            ),
            quality_score=avg_score,
        )


# =============================================================================
# PROMPT SYSTEM
# =============================================================================


@dataclass
class PromptTemplate:
    name: str
    version: str
    system_prompt: str
    user_template: str
    variables: List[str]


RESEARCH_TEMPLATE = PromptTemplate(
    name="research",
    version="1.0.0",
    system_prompt="""You are a research assistant that provides accurate, well-sourced information.

Guidelines:
- Base answers on provided context
- Cite sources when available
- Be concise but thorough""",
    user_template="""Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context above.""",
    variables=["context", "query"],
)


@dataclass
class AssembledPrompt:
    system: str
    user: str
    template_name: str
    template_version: str
    token_estimate: int


class PromptManager:
    """Manage prompt assembly."""

    def __init__(self):
        self.templates = {"research": RESEARCH_TEMPLATE}

    def assemble(
        self, template_name: str, variables: Dict[str, str]
    ) -> AssembledPrompt:
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        system = template.system_prompt
        user = template.user_template.format(**variables)
        token_estimate = int(len((system + user).split()) * 1.3)

        return AssembledPrompt(
            system=system,
            user=user,
            template_name=template_name,
            template_version=template.version,
            token_estimate=token_estimate,
        )


# =============================================================================
# OBSERVABILITY
# =============================================================================


class MetricsCollector:
    """Collect metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)

    def record(self, name: str, value: float) -> None:
        self.metrics[name].append(value)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def get_stats(self, name: str) -> Dict[str, float]:
        values = self.metrics.get(name, [])
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": (
                sorted(values)[int(len(values) * 0.95)]
                if len(values) > 1
                else values[0]
            ),
        }

    def get_counter(self, name: str) -> int:
        return self.counters.get(name, 0)


class StructuredLogger:
    """Structured logging."""

    def __init__(self, service_name: str = "research_assistant"):
        self.service_name = service_name
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        self.context.update(kwargs)

    def _log(self, level: str, message: str, **extra) -> None:
        log = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "service": self.service_name,
            "message": message,
            **self.context,
            **extra,
        }
        print(json.dumps(log))

    def info(self, message: str, **extra) -> None:
        self._log("INFO", message, **extra)

    def error(self, message: str, **extra) -> None:
        self._log("ERROR", message, **extra)


# =============================================================================
# ENTERPRISE FEATURES
# =============================================================================


class TenantTier(Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"


@dataclass
class TenantLimits:
    requests_per_minute: int
    tokens_per_day: int


TIER_LIMITS = {
    TenantTier.FREE: TenantLimits(10, 100_000),
    TenantTier.STARTER: TenantLimits(60, 1_000_000),
    TenantTier.PROFESSIONAL: TenantLimits(300, 10_000_000),
}


@dataclass
class Tenant:
    id: str
    name: str
    tier: TenantTier

    @classmethod
    def create(cls, name: str, tier: TenantTier = TenantTier.STARTER) -> "Tenant":
        return cls(id=str(uuid.uuid4())[:8], name=name, tier=tier)


class TenantManager:
    """Manage tenants."""

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.usage: Dict[str, Dict] = {}

    def register(self, tenant: Tenant) -> None:
        self.tenants[tenant.id] = tenant
        self.usage[tenant.id] = {"requests": 0, "tokens": 0}

    def get(self, tenant_id: str) -> Optional[Tenant]:
        return self.tenants.get(tenant_id)

    def check_limits(self, tenant_id: str, estimated_tokens: int) -> Dict:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {"allowed": False, "reason": "Unknown tenant"}

        limits = TIER_LIMITS[tenant.tier]
        usage = self.usage.get(tenant_id, {})

        if usage.get("tokens", 0) + estimated_tokens > limits.tokens_per_day:
            return {"allowed": False, "reason": "Token limit exceeded"}

        return {"allowed": True}

    def record_usage(self, tenant_id: str, tokens: int) -> None:
        if tenant_id in self.usage:
            self.usage[tenant_id]["requests"] += 1
            self.usage[tenant_id]["tokens"] += tokens


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class AccessController:
    """Access control."""

    def __init__(self):
        self.user_roles: Dict[str, List[str]] = {}
        self.role_perms = {
            "viewer": [Permission.READ],
            "editor": [Permission.READ, Permission.WRITE],
            "admin": [Permission.READ, Permission.WRITE, Permission.ADMIN],
        }

    def assign_role(self, user_id: str, role: str) -> None:
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        roles = self.user_roles.get(user_id, [])
        for role in roles:
            perms = self.role_perms.get(role, [])
            if permission in perms:
                return True
        return False


class PIIHandler:
    """PII detection and redaction."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    }

    def redact(self, text: str) -> str:
        result = text
        result = re.sub(self.PATTERNS["email"], "[EMAIL]", result, flags=re.IGNORECASE)
        result = re.sub(self.PATTERNS["phone"], "[PHONE]", result)
        return result

    def contains_pii(self, text: str) -> bool:
        for pattern in self.PATTERNS.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


class AuditLogger:
    """Audit logging."""

    def __init__(self):
        self.events: List[Dict] = []

    def log(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        outcome: str,
        details: Dict = None,
    ) -> None:
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tenant_id": tenant_id,
                "user_id": user_id,
                "action": action,
                "outcome": outcome,
                "details": details or {},
            }
        )

    def query(self, tenant_id: str, limit: int = 10) -> List[Dict]:
        return [e for e in self.events if e["tenant_id"] == tenant_id][-limit:]


# =============================================================================
# RESEARCH ASSISTANT
# =============================================================================


@dataclass
class ResearchRequest:
    tenant_id: str
    user_id: str
    query: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class ResearchResponse:
    request_id: str
    success: bool
    response: str = ""
    sources: List[str] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str = ""


class ResearchAssistant:
    """Complete Research Assistant."""

    def __init__(self):
        # Core
        self.context_pipeline = ContextPipeline()
        self.prompt_manager = PromptManager()
        self._setup_sources()

        # Observability
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger()

        # Enterprise
        self.tenants = TenantManager()
        self.access = AccessController()
        self.pii = PIIHandler()
        self.audit = AuditLogger()

    def _setup_sources(self) -> None:
        docs = DocumentStore()
        docs.add(
            Document(
                "ce1",
                "Context engineering is the systematic design of context for LLMs.",
                "docs",
            )
        )
        docs.add(
            Document(
                "ce2",
                "The four pillars are retrieval, ranking, assembly, and delivery.",
                "docs",
            )
        )

        kb = KnowledgeBase()
        kb.add_fact("context engineering", "Critical for production AI systems")
        kb.add_fact(
            "llm", "Large Language Models process context to generate responses"
        )

        history = ConversationHistory()

        self.context_pipeline.add_source(docs)
        self.context_pipeline.add_source(kb)
        self.context_pipeline.add_source(history)

    def register_tenant(
        self, name: str, tier: TenantTier = TenantTier.STARTER
    ) -> Tenant:
        tenant = Tenant.create(name, tier)
        self.tenants.register(tenant)
        return tenant

    def setup_user(self, user_id: str, role: str) -> None:
        self.access.assign_role(user_id, role)

    def research(self, request: ResearchRequest) -> ResearchResponse:
        start = datetime.now()
        self.metrics.increment("requests.total")

        # Check limits
        if not self.tenants.check_limits(request.tenant_id, 10000)["allowed"]:
            return ResearchResponse(request.request_id, False, error="Limit exceeded")

        # Check permission
        if not self.access.check_permission(request.user_id, Permission.READ):
            self.audit.log(
                request.tenant_id, request.user_id, "research", "denied:permission"
            )
            return ResearchResponse(
                request.request_id, False, error="Permission denied"
            )

        # Redact PII
        clean_query = self.pii.redact(request.query)

        # Retrieve context
        ctx_result = self.context_pipeline.retrieve(clean_query)
        self.metrics.record("context.quality", ctx_result.quality_score)

        # Assemble prompt
        prompt = self.prompt_manager.assemble(
            "research", {"context": ctx_result.context, "query": clean_query}
        )

        # Generate response (simulated)
        response_text = f"Based on the context about '{clean_query[:30]}...': "
        response_text += "Context engineering is a systematic approach to designing optimal context for LLMs."
        tokens_used = prompt.token_estimate + len(response_text.split()) * 2

        # Record
        latency = (datetime.now() - start).total_seconds() * 1000
        self.tenants.record_usage(request.tenant_id, tokens_used)
        self.metrics.record("request.latency", latency)
        self.metrics.record("tokens.used", tokens_used)

        self.audit.log(
            request.tenant_id,
            request.user_id,
            "research",
            "success",
            {"tokens": tokens_used},
        )

        return ResearchResponse(
            request_id=request.request_id,
            success=True,
            response=response_text,
            sources=ctx_result.sources,
            tokens_used=tokens_used,
            latency_ms=latency,
        )

    def get_dashboard(self) -> Dict:
        return {
            "requests": self.metrics.get_counter("requests.total"),
            "latency": self.metrics.get_stats("request.latency"),
            "quality": self.metrics.get_stats("context.quality"),
            "tokens": self.metrics.get_stats("tokens.used"),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESEARCH ASSISTANT - CAPSTONE PROJECT")
    print("=" * 60)

    # Initialize
    assistant = ResearchAssistant()

    # Setup
    print("\n" + "-" * 50)
    print("SETUP")
    print("-" * 50)

    tenant = assistant.register_tenant("Demo Company", TenantTier.PROFESSIONAL)
    assistant.setup_user("alice", "admin")
    print(f"Tenant: {tenant.name} ({tenant.id})")
    print(f"User: alice (admin)")

    # Research requests
    print("\n" + "-" * 50)
    print("RESEARCH REQUESTS")
    print("-" * 50)

    # Request 1
    response1 = assistant.research(
        ResearchRequest(
            tenant_id=tenant.id, user_id="alice", query="What is context engineering?"
        )
    )
    print(f"\nRequest 1: {'SUCCESS' if response1.success else 'FAILED'}")
    print(f"  Response: {response1.response[:80]}...")
    print(f"  Tokens: {response1.tokens_used}, Latency: {response1.latency_ms:.0f}ms")
    print(f"  Sources: {response1.sources}")

    # Request 2 (with PII) — show the redaction that happens before retrieval
    pii_query = "Find info for john@example.com about LLMs"
    print(f"\nRequest 2 (PII): query   = {pii_query!r}")
    print(f"  Redacted before use: {assistant.pii.redact(pii_query)!r}")
    response2 = assistant.research(
        ResearchRequest(tenant_id=tenant.id, user_id="alice", query=pii_query)
    )
    print(f"  Result: {'SUCCESS' if response2.success else 'FAILED'}")

    # Request 3 (unauthorized)
    response3 = assistant.research(
        ResearchRequest(tenant_id=tenant.id, user_id="unknown", query="Show me data")
    )
    print(f"\nRequest 3 (unauthorized): {response3.error}")

    # Structured logging — one JSON line per event, ready for log aggregation
    print("\n" + "-" * 50)
    print("STRUCTURED LOGGING")
    print("-" * 50)

    assistant.logger.set_context(tenant_id=tenant.id)
    assistant.logger.info(
        "research completed",
        request_id=response1.request_id,
        tokens=response1.tokens_used,
        sources=response1.sources,
    )

    # Dashboard
    print("\n" + "-" * 50)
    print("DASHBOARD")
    print("-" * 50)

    dashboard = assistant.get_dashboard()
    print(f"Total requests: {dashboard['requests']}")
    if dashboard["latency"]:
        print(f"Avg latency: {dashboard['latency'].get('mean', 0):.0f}ms")
    if dashboard["quality"]:
        print(f"Avg quality: {dashboard['quality'].get('mean', 0):.2f}")

    # Audit
    print("\n" + "-" * 50)
    print("AUDIT TRAIL")
    print("-" * 50)

    for event in assistant.audit.query(tenant.id):
        print(f"  {event['action']} by {event['user_id']}: {event['outcome']}")

    print("\n" + "=" * 60)
    print("CAPSTONE COMPLETE")
    print("=" * 60)
