"""
MULTI-TENANCY PATTERNS
Tenant isolation and resource management

Run: python multi_tenancy.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from abc import ABC, abstractmethod
from queue import Queue
from threading import Lock
import uuid

# =============================================================================
# TENANT MODEL
# =============================================================================


class TenantTier(Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class IsolationLevel(Enum):
    SHARED = "shared"
    POOLED = "pooled"
    DEDICATED = "dedicated"


@dataclass
class TenantConfig:
    max_requests_per_minute: int = 100
    max_tokens_per_day: int = 1_000_000
    max_context_size: int = 50000
    max_documents: int = 10000
    custom_models_enabled: bool = False
    priority_processing: bool = False
    dedicated_resources: bool = False
    data_retention_days: int = 90
    encryption_required: bool = True
    audit_logging: bool = True


@dataclass
class Tenant:
    id: str
    name: str
    tier: TenantTier
    isolation_level: IsolationLevel
    config: TenantConfig
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, tier: TenantTier = TenantTier.STARTER) -> "Tenant":
        configs = {
            TenantTier.FREE: TenantConfig(
                max_requests_per_minute=10,
                max_tokens_per_day=100_000,
                max_context_size=10000,
                max_documents=100,
            ),
            TenantTier.STARTER: TenantConfig(
                max_requests_per_minute=100,
                max_tokens_per_day=1_000_000,
                max_context_size=50000,
                max_documents=10000,
            ),
            TenantTier.PROFESSIONAL: TenantConfig(
                max_requests_per_minute=500,
                max_tokens_per_day=10_000_000,
                max_context_size=100000,
                max_documents=100000,
                custom_models_enabled=True,
                priority_processing=True,
            ),
            TenantTier.ENTERPRISE: TenantConfig(
                max_requests_per_minute=2000,
                max_tokens_per_day=100_000_000,
                max_context_size=200000,
                max_documents=1000000,
                custom_models_enabled=True,
                priority_processing=True,
                dedicated_resources=True,
            ),
        }

        isolation = {
            TenantTier.FREE: IsolationLevel.SHARED,
            TenantTier.STARTER: IsolationLevel.SHARED,
            TenantTier.PROFESSIONAL: IsolationLevel.POOLED,
            TenantTier.ENTERPRISE: IsolationLevel.DEDICATED,
        }

        return cls(
            id=str(uuid.uuid4())[:8],
            name=name,
            tier=tier,
            isolation_level=isolation[tier],
            config=configs[tier],
        )


# =============================================================================
# CONTEXT STORES
# =============================================================================


class TenantContextStore(ABC):
    @abstractmethod
    def store(self, tenant_id: str, key: str, context: Any) -> None:
        pass

    @abstractmethod
    def retrieve(self, tenant_id: str, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def delete(self, tenant_id: str, key: str) -> None:
        pass

    @abstractmethod
    def list_keys(self, tenant_id: str) -> List[str]:
        pass


class SharedContextStore(TenantContextStore):
    """Shared storage with logical tenant isolation."""

    def __init__(self):
        self.store_data: Dict[str, Dict[str, Any]] = {}

    def store(self, tenant_id: str, key: str, context: Any) -> None:
        if tenant_id not in self.store_data:
            self.store_data[tenant_id] = {}
        self.store_data[tenant_id][key] = context

    def retrieve(self, tenant_id: str, key: str) -> Optional[Any]:
        return self.store_data.get(tenant_id, {}).get(key)

    def delete(self, tenant_id: str, key: str) -> None:
        if tenant_id in self.store_data:
            self.store_data[tenant_id].pop(key, None)

    def list_keys(self, tenant_id: str) -> List[str]:
        return list(self.store_data.get(tenant_id, {}).keys())


class DedicatedContextStore(TenantContextStore):
    """Dedicated storage per tenant."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.data: Dict[str, Any] = {}

    def store(self, tenant_id: str, key: str, context: Any) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError(f"Store is dedicated to tenant {self.tenant_id}")
        self.data[key] = context

    def retrieve(self, tenant_id: str, key: str) -> Optional[Any]:
        if tenant_id != self.tenant_id:
            raise ValueError(f"Store is dedicated to tenant {self.tenant_id}")
        return self.data.get(key)

    def delete(self, tenant_id: str, key: str) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError(f"Store is dedicated to tenant {self.tenant_id}")
        self.data.pop(key, None)

    def list_keys(self, tenant_id: str) -> List[str]:
        if tenant_id != self.tenant_id:
            raise ValueError(f"Store is dedicated to tenant {self.tenant_id}")
        return list(self.data.keys())


# =============================================================================
# TENANT PIPELINE
# =============================================================================


@dataclass
class TenantRequest:
    tenant_id: str
    request_id: str
    query: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TenantContextPipeline:
    """Context pipeline with tenant isolation."""

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.shared_store = SharedContextStore()
        self.dedicated_stores: Dict[str, DedicatedContextStore] = {}
        self.usage_tracker: Dict[str, Dict] = {}

    def register_tenant(self, tenant: Tenant) -> None:
        self.tenants[tenant.id] = tenant
        self.usage_tracker[tenant.id] = {
            "requests_today": 0,
            "tokens_today": 0,
            "last_reset": datetime.now(),
        }

        if tenant.isolation_level == IsolationLevel.DEDICATED:
            self.dedicated_stores[tenant.id] = DedicatedContextStore(tenant.id)

    def get_store(self, tenant_id: str) -> TenantContextStore:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Unknown tenant: {tenant_id}")

        if tenant.isolation_level == IsolationLevel.DEDICATED:
            return self.dedicated_stores[tenant_id]
        return self.shared_store

    def check_limits(self, tenant_id: str, estimated_tokens: int) -> Dict:
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {"allowed": False, "reason": "Unknown tenant"}

        usage = self.usage_tracker.get(tenant_id, {})
        config = tenant.config

        if usage.get("tokens_today", 0) + estimated_tokens > config.max_tokens_per_day:
            return {"allowed": False, "reason": "Daily token limit exceeded"}

        return {"allowed": True}

    def process(self, request: TenantRequest) -> Dict:
        tenant = self.tenants.get(request.tenant_id)
        if not tenant:
            return {"success": False, "error": "Unknown tenant"}

        limit_check = self.check_limits(request.tenant_id, 5000)
        if not limit_check["allowed"]:
            return {"success": False, "error": limit_check["reason"]}

        store = self.get_store(request.tenant_id)

        # Check cache
        cache_key = f"cache:{request.query[:50]}"
        cached = store.retrieve(request.tenant_id, cache_key)
        if cached:
            return {
                "success": True,
                "response": cached,
                "cached": True,
                "tokens_used": 0,
            }

        # Simulate processing
        response = f"Response for tenant {tenant.name}: {request.query[:50]}"
        tokens_used = len(response.split()) * 2

        store.store(request.tenant_id, cache_key, response)
        self._track_usage(request.tenant_id, tokens_used)

        return {
            "success": True,
            "response": response,
            "cached": False,
            "tokens_used": tokens_used,
        }

    def _track_usage(self, tenant_id: str, tokens: int) -> None:
        if tenant_id not in self.usage_tracker:
            self.usage_tracker[tenant_id] = {
                "requests_today": 0,
                "tokens_today": 0,
                "last_reset": datetime.now(),
            }
        self.usage_tracker[tenant_id]["requests_today"] += 1
        self.usage_tracker[tenant_id]["tokens_today"] += tokens

    def get_tenant_usage(self, tenant_id: str) -> Dict:
        tenant = self.tenants.get(tenant_id)
        usage = self.usage_tracker.get(tenant_id, {})

        if not tenant:
            return {}

        return {
            "tenant_id": tenant_id,
            "tier": tenant.tier.value,
            "requests_today": usage.get("requests_today", 0),
            "tokens_today": usage.get("tokens_today", 0),
            "limits": {"max_tokens": tenant.config.max_tokens_per_day},
        }


# =============================================================================
# RESOURCE POOL
# =============================================================================


class ResourcePool:
    """Manage shared resources across tenants."""

    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.available: Queue = Queue()
        self.allocated: Dict[str, List] = {}
        self.lock = Lock()

        for i in range(pool_size):
            self.available.put(f"resource_{i}")

    def allocate(self, tenant_id: str, count: int = 1) -> List[str]:
        with self.lock:
            if tenant_id not in self.allocated:
                self.allocated[tenant_id] = []

            resources = []
            for _ in range(min(count, self.available.qsize())):
                resource = self.available.get_nowait()
                resources.append(resource)
                self.allocated[tenant_id].append(resource)

            return resources

    def release(self, tenant_id: str, resources: List[str] = None) -> None:
        with self.lock:
            if tenant_id not in self.allocated:
                return

            to_release = resources or self.allocated[tenant_id].copy()

            for resource in to_release:
                if resource in self.allocated[tenant_id]:
                    self.allocated[tenant_id].remove(resource)
                    self.available.put(resource)

    def get_allocation(self, tenant_id: str) -> List[str]:
        return self.allocated.get(tenant_id, [])


# =============================================================================
# DATA ISOLATION
# =============================================================================


class DataIsolationEnforcer:
    """Enforce data isolation between tenants."""

    def __init__(self):
        self.access_log: List[Dict] = []

    def validate_access(
        self, requesting_tenant: str, target_resource: str, operation: str
    ) -> bool:
        resource_tenant = self._extract_tenant(target_resource)

        self.access_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "requesting_tenant": requesting_tenant,
                "target_resource": target_resource,
                "resource_tenant": resource_tenant,
                "operation": operation,
                "allowed": requesting_tenant == resource_tenant,
            }
        )

        if requesting_tenant != resource_tenant:
            raise PermissionError(
                f"Tenant {requesting_tenant} cannot access "
                f"resource belonging to {resource_tenant}"
            )

        return True

    def _extract_tenant(self, resource_path: str) -> str:
        parts = resource_path.split(":")
        if len(parts) >= 2 and parts[0] == "tenant":
            return parts[1]
        raise ValueError(f"Invalid resource path: {resource_path}")

    def get_violations(self) -> List[Dict]:
        return [log for log in self.access_log if not log["allowed"]]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-TENANCY PATTERNS")
    print("=" * 60)

    pipeline = TenantContextPipeline()

    # Register tenants
    print("\n" + "-" * 50)
    print("REGISTERING TENANTS")
    print("-" * 50)

    enterprise = Tenant.create("Enterprise Corp", TenantTier.ENTERPRISE)
    starter = Tenant.create("StartupXYZ", TenantTier.STARTER)

    pipeline.register_tenant(enterprise)
    pipeline.register_tenant(starter)

    print(f"Enterprise: {enterprise.name} (ID: {enterprise.id})")
    print(f"  Tier: {enterprise.tier.value}")
    print(f"  Isolation: {enterprise.isolation_level.value}")
    print(f"  Daily tokens: {enterprise.config.max_tokens_per_day:,}")

    print(f"\nStarter: {starter.name} (ID: {starter.id})")
    print(f"  Tier: {starter.tier.value}")
    print(f"  Isolation: {starter.isolation_level.value}")
    print(f"  Daily tokens: {starter.config.max_tokens_per_day:,}")

    # Process requests
    print("\n" + "-" * 50)
    print("PROCESSING REQUESTS")
    print("-" * 50)

    result1 = pipeline.process(
        TenantRequest(
            tenant_id=enterprise.id, request_id="req-001", query="What is AI?"
        )
    )
    print(f"Enterprise request: {result1['success']}, tokens: {result1['tokens_used']}")

    result2 = pipeline.process(
        TenantRequest(
            tenant_id=starter.id, request_id="req-002", query="How does ML work?"
        )
    )
    print(f"Starter request: {result2['success']}, tokens: {result2['tokens_used']}")

    # Check cache
    result3 = pipeline.process(
        TenantRequest(
            tenant_id=enterprise.id, request_id="req-003", query="What is AI?"
        )
    )
    print(f"Cached request: {result3['cached']}")

    # Usage stats
    print("\n" + "-" * 50)
    print("USAGE STATISTICS")
    print("-" * 50)

    for tenant in [enterprise, starter]:
        usage = pipeline.get_tenant_usage(tenant.id)
        print(f"{tenant.name}:")
        print(f"  Requests: {usage['requests_today']}")
        print(f"  Tokens: {usage['tokens_today']}")

    # Resource pool
    print("\n" + "-" * 50)
    print("RESOURCE POOL")
    print("-" * 50)

    pool = ResourcePool(pool_size=10)
    allocated = pool.allocate(enterprise.id, 3)
    print(f"Allocated to enterprise: {allocated}")
    print(f"Remaining in pool: {pool.available.qsize()}")

    pool.release(enterprise.id)
    print(f"After release: {pool.available.qsize()}")

    # Tenant isolation - the headline: A cannot read B
    print("\n" + "-" * 50)
    print("TENANT ISOLATION")
    print("-" * 50)

    enforcer = DataIsolationEnforcer()
    enterprise_resource = f"tenant:{enterprise.id}:context:secrets"

    # A tenant can access its OWN namespaced resource
    enforcer.validate_access(enterprise.id, enterprise_resource, "read")
    print(f"Enterprise reads its own resource: ALLOWED")

    # A tenant CANNOT access another tenant's resource
    try:
        enforcer.validate_access(starter.id, enterprise_resource, "read")
        print("Starter reads enterprise resource: ALLOWED (isolation broken!)")
    except PermissionError as e:
        print(f"Starter reads enterprise resource: BLOCKED")
        print(f"  -> {e}")

    # Logical isolation in the shared store: each tenant_id is its own namespace
    shared = pipeline.shared_store
    shared.store(starter.id, "note", "starter-only data")
    print(f"\nStarter reads its own key:        {shared.retrieve(starter.id, 'note')}")
    print(f"Enterprise reads starter's key:   {shared.retrieve(enterprise.id, 'note')}")

    # Audit trail of denied attempts
    violations = enforcer.get_violations()
    print(f"\nAccess violations logged: {len(violations)}")
    for v in violations:
        print(
            f"  {v['requesting_tenant']} -> {v['resource_tenant']} "
            f"({v['operation']}) DENIED"
        )
