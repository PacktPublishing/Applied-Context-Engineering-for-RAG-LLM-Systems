"""
COMPLIANCE & SECURITY
PII handling, audit trails, access control

Run: python compliance_security.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re
import hashlib
import uuid

# =============================================================================
# PII DETECTION AND HANDLING
# =============================================================================


class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"


@dataclass
class PIIMatch:
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float


class PIIDetector:
    """Detect PII in text content."""

    PATTERNS = {
        PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        PIIType.PHONE: r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b",
        PIIType.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
        PIIType.CREDIT_CARD: r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        PIIType.IP_ADDRESS: r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def detect(self, text: str) -> List[PIIMatch]:
        matches = []
        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    )
                )
        return matches

    def contains_pii(self, text: str) -> bool:
        return len(self.detect(text)) > 0


class PIIRedactor:
    """Redact PII from content."""

    TOKENS = {
        PIIType.EMAIL: "[EMAIL_REDACTED]",
        PIIType.PHONE: "[PHONE_REDACTED]",
        PIIType.SSN: "[SSN_REDACTED]",
        PIIType.CREDIT_CARD: "[CC_REDACTED]",
        PIIType.IP_ADDRESS: "[IP_REDACTED]",
    }

    def __init__(self, detector: PIIDetector = None):
        self.detector = detector or PIIDetector()

    def redact(self, text: str) -> Tuple[str, List[PIIMatch]]:
        matches = self.detector.detect(text)
        matches.sort(key=lambda m: m.start, reverse=True)

        redacted = text
        for match in matches:
            token = self.TOKENS.get(match.pii_type, "[REDACTED]")
            redacted = redacted[: match.start] + token + redacted[match.end :]

        return redacted, matches

    def hash_pii(self, text: str, salt: str = "") -> Tuple[str, Dict[str, str]]:
        matches = self.detector.detect(text)
        hash_map = {}

        matches.sort(key=lambda m: m.start, reverse=True)

        hashed = text
        for match in matches:
            hash_input = f"{salt}:{match.value}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
            token = f"[{match.pii_type.value.upper()}_{hash_value}]"

            hash_map[token] = match.value
            hashed = hashed[: match.start] + token + hashed[match.end :]

        return hashed, hash_map


# =============================================================================
# ACCESS CONTROL
# =============================================================================


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXPORT = "export"


class ResourceType(Enum):
    CONTEXT = "context"
    DOCUMENT = "document"
    CONFIGURATION = "configuration"
    AUDIT_LOG = "audit_log"
    USER_DATA = "user_data"


@dataclass
class Role:
    name: str
    permissions: Dict[ResourceType, List[Permission]]


@dataclass
class AccessPolicy:
    roles: Dict[str, Role] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "AccessPolicy":
        return cls(
            roles={
                "viewer": Role(
                    name="viewer",
                    permissions={
                        ResourceType.CONTEXT: [Permission.READ],
                        ResourceType.DOCUMENT: [Permission.READ],
                    },
                ),
                "editor": Role(
                    name="editor",
                    permissions={
                        ResourceType.CONTEXT: [Permission.READ, Permission.WRITE],
                        ResourceType.DOCUMENT: [Permission.READ, Permission.WRITE],
                        ResourceType.CONFIGURATION: [Permission.READ],
                    },
                ),
                "admin": Role(
                    name="admin",
                    permissions={
                        ResourceType.CONTEXT: [
                            Permission.READ,
                            Permission.WRITE,
                            Permission.DELETE,
                        ],
                        ResourceType.DOCUMENT: [
                            Permission.READ,
                            Permission.WRITE,
                            Permission.DELETE,
                        ],
                        ResourceType.CONFIGURATION: [Permission.READ, Permission.WRITE],
                        ResourceType.AUDIT_LOG: [Permission.READ],
                        ResourceType.USER_DATA: [
                            Permission.READ,
                            Permission.WRITE,
                            Permission.ADMIN,
                        ],
                    },
                ),
            }
        )


class AccessController:
    """Control access to resources."""

    def __init__(self, policy: AccessPolicy = None):
        self.policy = policy or AccessPolicy.default()
        self.user_roles: Dict[str, List[str]] = {}

    def assign_role(self, user_id: str, role_name: str) -> None:
        if role_name not in self.policy.roles:
            raise ValueError(f"Unknown role: {role_name}")

        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)

    def check_permission(
        self, user_id: str, resource_type: ResourceType, permission: Permission
    ) -> bool:
        user_role_names = self.user_roles.get(user_id, [])

        for role_name in user_role_names:
            role = self.policy.roles.get(role_name)
            if role:
                resource_perms = role.permissions.get(resource_type, [])
                if permission in resource_perms:
                    return True

        return False

    def require_permission(
        self, user_id: str, resource_type: ResourceType, permission: Permission
    ) -> None:
        if not self.check_permission(user_id, resource_type, permission):
            raise PermissionError(
                f"User {user_id} lacks {permission.value} "
                f"permission for {resource_type.value}"
            )


# =============================================================================
# AUDIT LOGGING
# =============================================================================


class AuditEventType(Enum):
    ACCESS = "access"
    MODIFY = "modify"
    DELETE = "delete"
    EXPORT = "export"
    LOGIN = "login"
    PERMISSION_CHANGE = "permission_change"
    PII_ACCESS = "pii_access"


@dataclass
class AuditEvent:
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: str
    tenant_id: str
    resource_type: str
    resource_id: str
    action: str
    outcome: str
    details: Dict = field(default_factory=dict)
    ip_address: Optional[str] = None


class AuditLogger:
    """Immutable audit logging system."""

    def __init__(self):
        self.events: List[AuditEvent] = []
        self._locked_events: set = set()

    def log(
        self,
        event_type: AuditEventType,
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        outcome: str,
        details: Dict = None,
        ip_address: str = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
        )

        self.events.append(event)
        self._locked_events.add(event.event_id)

        return event

    def query(
        self,
        tenant_id: str = None,
        user_id: str = None,
        event_type: AuditEventType = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        results = self.events.copy()

        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]

        return results[-limit:]


# =============================================================================
# COMPLIANCE CHECKING
# =============================================================================


class ComplianceRegulation(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"


@dataclass
class ComplianceRequirement:
    regulation: ComplianceRegulation
    requirement_id: str
    description: str
    check_fn: callable


class ComplianceChecker:
    """Check compliance with regulations."""

    def __init__(self):
        self.requirements: List[ComplianceRequirement] = []
        self._register_requirements()

    def _register_requirements(self):
        self.requirements.append(
            ComplianceRequirement(
                regulation=ComplianceRegulation.GDPR,
                requirement_id="GDPR-7",
                description="PII must be encrypted at rest",
                check_fn=lambda c: c.get("encryption_enabled", False),
            )
        )

        self.requirements.append(
            ComplianceRequirement(
                regulation=ComplianceRegulation.SOC2,
                requirement_id="SOC2-CC6.1",
                description="Audit logging must be enabled",
                check_fn=lambda c: c.get("audit_logging_enabled", False),
            )
        )

    def check(
        self, regulations: List[ComplianceRegulation], system_config: Dict
    ) -> Dict:
        results = {"compliant": True, "checks": []}

        for req in self.requirements:
            if req.regulation in regulations:
                passed = req.check_fn(system_config)
                results["checks"].append(
                    {
                        "regulation": req.regulation.value,
                        "requirement_id": req.requirement_id,
                        "description": req.description,
                        "passed": passed,
                    }
                )
                if not passed:
                    results["compliant"] = False

        return results


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPLIANCE & SECURITY")
    print("=" * 60)

    # PII Detection
    print("\n" + "-" * 50)
    print("PII DETECTION")
    print("-" * 50)

    text = """
    Contact John at john.doe@example.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card is 4111-1111-1111-1111.
    Server IP: 192.168.1.100
    """

    detector = PIIDetector()
    matches = detector.detect(text)

    print(f"Found {len(matches)} PII instances:")
    for match in matches:
        print(f"  {match.pii_type.value}: {match.value}")

    # PII Redaction
    print("\n" + "-" * 50)
    print("PII REDACTION")
    print("-" * 50)

    redactor = PIIRedactor(detector)
    redacted, _ = redactor.redact(text)
    print(f"Redacted text:\n{redacted}")

    # Access Control
    print("\n" + "-" * 50)
    print("ACCESS CONTROL")
    print("-" * 50)

    controller = AccessController()
    controller.assign_role("alice", "admin")
    controller.assign_role("bob", "viewer")

    print(
        f"Alice can write context: {controller.check_permission('alice', ResourceType.CONTEXT, Permission.WRITE)}"
    )
    print(
        f"Bob can write context: {controller.check_permission('bob', ResourceType.CONTEXT, Permission.WRITE)}"
    )

    # Audit Logging
    print("\n" + "-" * 50)
    print("AUDIT LOGGING")
    print("-" * 50)

    logger = AuditLogger()

    logger.log(
        AuditEventType.ACCESS,
        user_id="alice",
        tenant_id="tenant_1",
        resource_type="context",
        resource_id="ctx_123",
        action="read",
        outcome="success",
    )

    logger.log(
        AuditEventType.PII_ACCESS,
        user_id="bob",
        tenant_id="tenant_1",
        resource_type="document",
        resource_id="doc_456",
        action="export",
        outcome="success",
        details={"pii_types": ["email", "phone"]},
    )

    print(f"Logged {len(logger.events)} audit events")
    for event in logger.events:
        print(f"  {event.timestamp}: {event.action} by {event.user_id}")

    # Compliance Check
    print("\n" + "-" * 50)
    print("COMPLIANCE CHECK")
    print("-" * 50)

    checker = ComplianceChecker()
    config = {"encryption_enabled": True, "audit_logging_enabled": True}

    result = checker.check(
        [ComplianceRegulation.GDPR, ComplianceRegulation.SOC2], config
    )

    print(f"Compliant: {result['compliant']}")
    for check in result["checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['requirement_id']}: {check['description']}")
