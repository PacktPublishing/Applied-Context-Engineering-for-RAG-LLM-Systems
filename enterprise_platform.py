"""
ENTERPRISE CONTEXT PLATFORM
Complete integration of all enterprise patterns

Run: python enterprise_platform.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import uuid

# Import from other modules (for demo, we'll inline simplified versions)


# =============================================================================
# TENANT MANAGEMENT
# =============================================================================


@dataclass
class TenantConfig:
    max_requests_per_minute: int = 100
    max_tokens_per_day: int = 1_000_000
    pii_redaction_enabled: bool = True
    audit_logging_enabled: bool = True


@dataclass
class Tenant:
    id: str
    name: str
    tier: str
    config: TenantConfig

    @classmethod
    def create(cls, name: str, tier: str = "starter") -> "Tenant":
        return cls(
            id=str(uuid.uuid4())[:8], name=name, tier=tier, config=TenantConfig()
        )


# =============================================================================
# SECURITY & COMPLIANCE
# =============================================================================


class PIIHandler:
    """Handle PII detection and redaction."""

    def __init__(self):
        self.redaction_tokens = {"email": "[EMAIL]", "phone": "[PHONE]", "ssn": "[SSN]"}

    def redact(self, text: str) -> str:
        # Simplified - in production use regex patterns
        import re

        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
        text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
        return text


class AuditLogger:
    """Immutable audit logging."""

    def __init__(self):
        self.events: List[Dict] = []

    def log(
        self, tenant_id: str, user_id: str, action: str, resource: str, outcome: str
    ) -> str:
        event_id = str(uuid.uuid4())[:8]
        self.events.append(
            {
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tenant_id": tenant_id,
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "outcome": outcome,
            }
        )
        return event_id

    def query(self, tenant_id: str, limit: int = 100) -> List[Dict]:
        return [e for e in self.events if e["tenant_id"] == tenant_id][-limit:]


class AccessController:
    """Role-based access control."""

    def __init__(self):
        self.user_roles: Dict[str, List[str]] = {}
        self.role_permissions = {
            "viewer": ["read"],
            "editor": ["read", "write"],
            "admin": ["read", "write", "delete", "admin"],
        }

    def assign_role(self, user_id: str, role: str) -> None:
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)

    def check_permission(self, user_id: str, permission: str) -> bool:
        roles = self.user_roles.get(user_id, [])
        for role in roles:
            perms = self.role_permissions.get(role, [])
            if permission in perms:
                return True
        return False


# =============================================================================
# VERSIONING
# =============================================================================


@dataclass
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class VersionManager:
    """Manage versioned artifacts."""

    def __init__(self):
        self.artifacts: Dict[str, Dict[str, Any]] = {}

    def store(self, artifact_id: str, version: Version, content: Any) -> None:
        if artifact_id not in self.artifacts:
            self.artifacts[artifact_id] = {}
        self.artifacts[artifact_id][str(version)] = {
            "content": content,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

    def get(self, artifact_id: str, version: str = None) -> Optional[Any]:
        if artifact_id not in self.artifacts:
            return None

        versions = self.artifacts[artifact_id]
        if version:
            entry = versions.get(version)
            return entry["content"] if entry else None

        # Return latest
        if versions:
            latest = max(versions.keys())
            return versions[latest]["content"]
        return None


# =============================================================================
# QUALITY & METRICS
# =============================================================================


@dataclass
class QualityMetrics:
    relevance_score: float = 0.0
    coverage_score: float = 0.0
    token_count: int = 0
    latency_ms: float = 0.0


class QualityTracker:
    """Track context quality over time."""

    def __init__(self):
        self.metrics: List[QualityMetrics] = []
        self.thresholds = {
            "min_relevance": 0.7,
            "min_coverage": 0.8,
            "max_tokens": 50000,
        }

    def record(self, metrics: QualityMetrics) -> None:
        self.metrics.append(metrics)

    def check_quality(self, metrics: QualityMetrics) -> Dict[str, bool]:
        return {
            "relevance_ok": metrics.relevance_score >= self.thresholds["min_relevance"],
            "coverage_ok": metrics.coverage_score >= self.thresholds["min_coverage"],
            "tokens_ok": metrics.token_count <= self.thresholds["max_tokens"],
        }

    def get_averages(self) -> Dict[str, float]:
        if not self.metrics:
            return {}

        return {
            "avg_relevance": sum(m.relevance_score for m in self.metrics)
            / len(self.metrics),
            "avg_coverage": sum(m.coverage_score for m in self.metrics)
            / len(self.metrics),
            "avg_tokens": sum(m.token_count for m in self.metrics) / len(self.metrics),
        }


# =============================================================================
# ENTERPRISE PLATFORM
# =============================================================================


@dataclass
class PlatformRequest:
    tenant_id: str
    user_id: str
    query: str
    context_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlatformResponse:
    request_id: str
    tenant_id: str
    response: str
    context_used: str
    quality_metrics: QualityMetrics
    audit_event_id: str
    success: bool
    error: Optional[str] = None


class EnterpriseContextPlatform:
    """Complete enterprise context platform."""

    def __init__(self):
        # Core components
        self.tenants: Dict[str, Tenant] = {}
        self.pii_handler = PIIHandler()
        self.audit_logger = AuditLogger()
        self.access_controller = AccessController()
        self.version_manager = VersionManager()
        self.quality_tracker = QualityTracker()

        # Usage tracking
        self.usage: Dict[str, Dict] = {}

    def register_tenant(self, tenant: Tenant) -> None:
        """Register a new tenant."""
        self.tenants[tenant.id] = tenant
        self.usage[tenant.id] = {"requests_today": 0, "tokens_today": 0}

    def setup_user(self, tenant_id: str, user_id: str, role: str) -> None:
        """Setup user with role."""
        self.access_controller.assign_role(user_id, role)
        self.audit_logger.log(
            tenant_id=tenant_id,
            user_id="system",
            action="user_setup",
            resource=user_id,
            outcome=f"assigned_role:{role}",
        )

    def process_request(self, request: PlatformRequest) -> PlatformResponse:
        """Process a context request through the platform."""
        request_id = str(uuid.uuid4())[:8]

        # Validate tenant
        tenant = self.tenants.get(request.tenant_id)
        if not tenant:
            return PlatformResponse(
                request_id=request_id,
                tenant_id=request.tenant_id,
                response="",
                context_used="",
                quality_metrics=QualityMetrics(),
                audit_event_id="",
                success=False,
                error="Unknown tenant",
            )

        # Check permissions
        if not self.access_controller.check_permission(request.user_id, "read"):
            audit_id = self.audit_logger.log(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                action="context_request",
                resource="context",
                outcome="denied:permission",
            )
            return PlatformResponse(
                request_id=request_id,
                tenant_id=request.tenant_id,
                response="",
                context_used="",
                quality_metrics=QualityMetrics(),
                audit_event_id=audit_id,
                success=False,
                error="Permission denied",
            )

        # Simulate context retrieval
        raw_context = f"Context for query: {request.query}"

        # PII redaction if enabled
        if tenant.config.pii_redaction_enabled:
            context = self.pii_handler.redact(raw_context)
        else:
            context = raw_context

        # Calculate quality metrics
        metrics = QualityMetrics(
            relevance_score=0.85,
            coverage_score=0.90,
            token_count=len(context.split()) * 2,
            latency_ms=50.0,
        )

        # Track quality
        self.quality_tracker.record(metrics)
        quality_check = self.quality_tracker.check_quality(metrics)

        # Generate response
        response = f"Response based on: {context[:100]}"

        # Update usage
        self.usage[request.tenant_id]["requests_today"] += 1
        self.usage[request.tenant_id]["tokens_today"] += metrics.token_count

        # Audit log
        audit_id = self.audit_logger.log(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            action="context_request",
            resource="context",
            outcome="success",
        )

        return PlatformResponse(
            request_id=request_id,
            tenant_id=request.tenant_id,
            response=response,
            context_used=context,
            quality_metrics=metrics,
            audit_event_id=audit_id,
            success=True,
        )

    def get_tenant_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get dashboard data for a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}

        usage = self.usage.get(tenant_id, {})
        quality_avgs = self.quality_tracker.get_averages()
        recent_audit = self.audit_logger.query(tenant_id, limit=10)

        return {
            "tenant": {"id": tenant.id, "name": tenant.name, "tier": tenant.tier},
            "usage": {
                "requests_today": usage.get("requests_today", 0),
                "tokens_today": usage.get("tokens_today", 0),
                "token_limit": tenant.config.max_tokens_per_day,
            },
            "quality": quality_avgs,
            "recent_activity": len(recent_audit),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTERPRISE CONTEXT PLATFORM")
    print("=" * 60)

    # Initialize platform
    platform = EnterpriseContextPlatform()

    # Register tenants
    print("\n" + "-" * 50)
    print("REGISTERING TENANTS")
    print("-" * 50)

    enterprise = Tenant.create("Enterprise Corp", "enterprise")
    startup = Tenant.create("StartupXYZ", "starter")

    platform.register_tenant(enterprise)
    platform.register_tenant(startup)

    print(f"Registered: {enterprise.name} ({enterprise.id}, tier={enterprise.tier})")
    print(f"Registered: {startup.name} ({startup.id}, tier={startup.tier})")

    # Setup users (role-based access control)
    print("\n" + "-" * 50)
    print("SETTING UP USERS (ROLE-BASED ACCESS)")
    print("-" * 50)

    platform.setup_user(enterprise.id, "alice", "admin")
    platform.setup_user(enterprise.id, "bob", "viewer")
    platform.setup_user(startup.id, "charlie", "editor")

    print("alice   -> admin  (read, write, delete, admin)")
    print("bob     -> viewer (read)")
    print("charlie -> editor (read, write)")

    # Process a request that carries PII, and show it redacted
    print("\n" + "-" * 50)
    print("PROCESSING REQUESTS (PII REDACTION)")
    print("-" * 50)

    response1 = platform.process_request(
        PlatformRequest(
            tenant_id=enterprise.id,
            user_id="alice",
            query="Q4 metrics? Contact alice@example.com or 555-123-4567, SSN 123-45-6789",
        )
    )

    print(f"Request 1 success: {response1.success}")
    print(f"  Context used: {response1.context_used}")
    quality_check = platform.quality_tracker.check_quality(response1.quality_metrics)
    print(
        f"  Quality: relevance={response1.quality_metrics.relevance_score}, "
        f"coverage={response1.quality_metrics.coverage_score}"
    )
    print(f"  Quality gates: {quality_check}")
    print(f"  Audit ID: {response1.audit_event_id}")

    # Request from an unauthorized user is denied and audited
    response2 = platform.process_request(
        PlatformRequest(
            tenant_id=enterprise.id, user_id="unknown_user", query="Show me the data"
        )
    )

    print(f"\nRequest 2 (unauthorized user): success={response2.success}")
    print(f"  Error: {response2.error}")

    # Tenant isolation: each tenant tracks usage independently
    print("\n" + "-" * 50)
    print("TENANT ISOLATION (INDEPENDENT USAGE)")
    print("-" * 50)

    platform.process_request(
        PlatformRequest(
            tenant_id=startup.id,
            user_id="charlie",
            query="Summarize the onboarding docs",
        )
    )

    for t in (enterprise, startup):
        d = platform.get_tenant_dashboard(t.id)
        print(
            f"  {d['tenant']['name']:16} "
            f"requests={d['usage']['requests_today']} "
            f"tokens={d['usage']['tokens_today']} "
            f"recent_events={d['recent_activity']}"
        )

    # Content versioning
    print("\n" + "-" * 50)
    print("CONTENT VERSIONING")
    print("-" * 50)

    platform.version_manager.store(
        "prompt:support", Version(1, 0, 0), "You are a helpful support agent."
    )
    platform.version_manager.store(
        "prompt:support", Version(1, 1, 0), "You are a helpful, concise support agent."
    )
    print(
        f"  latest prompt:support -> {platform.version_manager.get('prompt:support')}"
    )
    print(
        f"  pinned 1.0.0          -> {platform.version_manager.get('prompt:support', '1.0.0')}"
    )

    # Dashboard
    print("\n" + "-" * 50)
    print("TENANT DASHBOARD (Enterprise Corp)")
    print("-" * 50)

    dashboard = platform.get_tenant_dashboard(enterprise.id)
    print(
        f"  Tenant: {dashboard['tenant']['name']} (tier={dashboard['tenant']['tier']})"
    )
    print(f"  Requests today: {dashboard['usage']['requests_today']}")
    print(
        f"  Tokens used: {dashboard['usage']['tokens_today']} "
        f"/ {dashboard['usage']['token_limit']:,}"
    )
    print(f"  Avg quality (platform-wide): {dashboard['quality']}")

    # Audit trail
    print("\n" + "-" * 50)
    print("AUDIT TRAIL (Enterprise Corp)")
    print("-" * 50)

    audit_events = platform.audit_logger.query(enterprise.id)
    for event in audit_events[:6]:
        print(
            f"  {event['timestamp'][:19]} | {event['action']:14} | "
            f"{event['user_id']:12} | {event['outcome']}"
        )

    print("\n" + "-" * 50)
    print("ENTERPRISE PLATFORM READY")
    print("-" * 50)
