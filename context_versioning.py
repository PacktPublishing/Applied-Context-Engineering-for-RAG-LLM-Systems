"""
CONTEXT VERSIONING
Semantic versioning for schemas, prompts, and configs

Run: python context_versioning.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json
import re

# =============================================================================
# VERSION MODEL
# =============================================================================


@dataclass
class Version:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version: {version_str}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
        )

    def bump_major(self) -> "Version":
        return Version(self.major + 1, 0, 0)

    def bump_minor(self) -> "Version":
        return Version(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "Version":
        return Version(self.major, self.minor, self.patch + 1)

    def is_compatible_with(self, other: "Version") -> bool:
        """Check if this version is compatible with another (same major)."""
        return self.major == other.major


class ChangeType(Enum):
    BREAKING = "breaking"
    FEATURE = "feature"
    FIX = "fix"


@dataclass
class VersionChange:
    change_type: ChangeType
    description: str
    affected_fields: List[str] = field(default_factory=list)


# =============================================================================
# VERSIONED ARTIFACTS
# =============================================================================


@dataclass
class VersionedArtifact:
    """Base class for versioned content."""

    id: str
    version: Version
    content: Any
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    changes: List[VersionChange] = field(default_factory=list)


class ArtifactStore:
    """Store and retrieve versioned artifacts."""

    def __init__(self):
        self.artifacts: Dict[str, Dict[str, VersionedArtifact]] = {}

    def store(self, artifact: VersionedArtifact) -> None:
        if artifact.id not in self.artifacts:
            self.artifacts[artifact.id] = {}

        version_key = str(artifact.version)
        self.artifacts[artifact.id][version_key] = artifact

    def get(self, artifact_id: str, version: str = None) -> Optional[VersionedArtifact]:
        if artifact_id not in self.artifacts:
            return None

        versions = self.artifacts[artifact_id]

        if version:
            return versions.get(version)

        # Return latest version
        if not versions:
            return None

        latest_key = max(
            versions.keys(),
            key=lambda v: Version.parse(v).major * 10000
            + Version.parse(v).minor * 100
            + Version.parse(v).patch,
        )
        return versions[latest_key]

    def list_versions(self, artifact_id: str) -> List[str]:
        if artifact_id not in self.artifacts:
            return []
        # Sort by semantic version, not lexically, so 1.9.0 < 1.10.0
        return sorted(
            self.artifacts[artifact_id].keys(),
            key=lambda v: (
                Version.parse(v).major,
                Version.parse(v).minor,
                Version.parse(v).patch,
            ),
        )


# =============================================================================
# SCHEMA VERSIONING
# =============================================================================


@dataclass
class SchemaField:
    name: str
    field_type: str
    required: bool = True
    default: Any = None


@dataclass
class ContextSchema:
    name: str
    version: Version
    fields: List[SchemaField]
    description: str = ""

    def validate(self, data: Dict) -> Dict:
        """Validate data against schema."""
        errors = []
        warnings = []

        for field in self.fields:
            if field.name not in data:
                if field.required and field.default is None:
                    errors.append(f"Missing required field: {field.name}")
            else:
                # Basic type checking
                value = data[field.name]
                expected_type = {
                    "string": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                }.get(field.field_type)

                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Field {field.name}: expected {field.field_type}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# =============================================================================
# MIGRATIONS
# =============================================================================


@dataclass
class Migration:
    """Migrate data between schema versions."""

    from_version: Version
    to_version: Version
    description: str
    migrate_fn: Callable[[Dict], Dict]
    rollback_fn: Optional[Callable[[Dict], Dict]] = None


class MigrationManager:
    """Manage schema migrations."""

    def __init__(self):
        self.migrations: Dict[str, List[Migration]] = {}

    def register(self, schema_name: str, migration: Migration) -> None:
        if schema_name not in self.migrations:
            self.migrations[schema_name] = []
        self.migrations[schema_name].append(migration)

    def migrate(
        self, schema_name: str, data: Dict, from_version: Version, to_version: Version
    ) -> Dict:
        if schema_name not in self.migrations:
            raise ValueError(f"No migrations for schema: {schema_name}")

        migrations = self.migrations[schema_name]
        current_version = from_version
        result = data.copy()

        while str(current_version) != str(to_version):
            # Find next migration
            next_migration = None
            for m in migrations:
                if str(m.from_version) == str(current_version):
                    next_migration = m
                    break

            if not next_migration:
                raise ValueError(
                    f"No migration path from {current_version} to {to_version}"
                )

            result = next_migration.migrate_fn(result)
            current_version = next_migration.to_version

        return result

    def get_migration_path(
        self, schema_name: str, from_version: Version, to_version: Version
    ) -> List[Migration]:
        if schema_name not in self.migrations:
            return []

        path = []
        current = from_version
        migrations = self.migrations[schema_name]

        while str(current) != str(to_version):
            for m in migrations:
                if str(m.from_version) == str(current):
                    path.append(m)
                    current = m.to_version
                    break
            else:
                return []  # No path found

        return path


# =============================================================================
# PROMPT VERSIONING
# =============================================================================


@dataclass
class PromptVersion:
    version: Version
    template: str
    variables: List[str]
    created_at: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class PromptRegistry:
    """Version and manage prompts."""

    def __init__(self):
        self.prompts: Dict[str, List[PromptVersion]] = {}
        self.active_versions: Dict[str, str] = {}

    def register(self, name: str, prompt: PromptVersion) -> None:
        if name not in self.prompts:
            self.prompts[name] = []
            self.active_versions[name] = str(prompt.version)

        self.prompts[name].append(prompt)

    def get(self, name: str, version: str = None) -> Optional[PromptVersion]:
        if name not in self.prompts:
            return None

        if version:
            for p in self.prompts[name]:
                if str(p.version) == version:
                    return p
            return None

        # Return active version
        active = self.active_versions.get(name)
        if active:
            return self.get(name, active)

        return self.prompts[name][-1] if self.prompts[name] else None

    def set_active(self, name: str, version: str) -> None:
        if name not in self.prompts:
            raise ValueError(f"Unknown prompt: {name}")

        for p in self.prompts[name]:
            if str(p.version) == version:
                self.active_versions[name] = version
                return

        raise ValueError(f"Version {version} not found for prompt {name}")

    def record_metrics(
        self, name: str, version: str, metrics: Dict[str, float]
    ) -> None:
        prompt = self.get(name, version)
        if prompt:
            prompt.performance_metrics.update(metrics)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT VERSIONING")
    print("=" * 60)

    # -------------------------------------------------------------
    # Version parsing and bumping
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("VERSION OPERATIONS")
    print("-" * 50)

    v1 = Version.parse("1.2.3")
    print(f"Parsed version: {v1}")
    print(f"Bump major:  {v1.bump_major()}   (breaking change)")
    print(f"Bump minor:  {v1.bump_minor()}   (new feature)")
    print(f"Bump patch:  {v1.bump_patch()}   (fix)")

    v2 = Version.parse("1.5.0")
    v3 = Version.parse("2.0.0")
    print(f"\n{v1} compatible with {v2}: {v1.is_compatible_with(v2)}")
    print(f"{v1} compatible with {v3}: {v1.is_compatible_with(v3)}")

    # -------------------------------------------------------------
    # Artifact store: keep every version, pin the one you want
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("ARTIFACT STORE / VERSION HISTORY")
    print("-" * 50)

    store = ArtifactStore()
    for ver, text in [
        (Version(1, 0, 0), "You are a helpful assistant."),
        (Version(1, 1, 0), "You are a helpful assistant. Be concise."),
        (Version(2, 0, 0), "You are a support agent for {company}. Be concise."),
    ]:
        store.store(
            VersionedArtifact(
                id="system_prompt",
                version=ver,
                content=text,
                created_at=datetime.now(),
            )
        )

    print(f"Stored versions: {store.list_versions('system_prompt')}")
    latest = store.get("system_prompt")
    print(f"Latest (default): v{latest.version} -> {latest.content!r}")
    pinned = store.get("system_prompt", "1.1.0")
    print(f"Pinned  v1.1.0:   -> {pinned.content!r}")

    # -------------------------------------------------------------
    # Schema versioning
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("SCHEMA VERSIONING")
    print("-" * 50)

    schema_v1 = ContextSchema(
        name="UserContext",
        version=Version(1, 0, 0),
        fields=[
            SchemaField("user_id", "string"),
            SchemaField("preferences", "dict"),
        ],
    )

    data = {"user_id": "123", "preferences": {"theme": "dark"}}
    result = schema_v1.validate(data)
    print(f"Schema: {schema_v1.name} v{schema_v1.version}")
    print(f"Validation: {result['valid']}, errors: {result['errors']}")

    # -------------------------------------------------------------
    # Migrations: forward, then roll back
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("MIGRATIONS + ROLLBACK")
    print("-" * 50)

    manager = MigrationManager()

    manager.register(
        "UserContext",
        Migration(
            from_version=Version(1, 0, 0),
            to_version=Version(2, 0, 0),
            description="Add email field, rename preferences to settings",
            migrate_fn=lambda d: {
                "user_id": d["user_id"],
                "email": d.get("email", ""),
                "settings": d.get("preferences", {}),
            },
            rollback_fn=lambda d: {
                "user_id": d["user_id"],
                "preferences": d.get("settings", {}),
            },
        ),
    )

    path = manager.get_migration_path("UserContext", Version(1, 0, 0), Version(2, 0, 0))
    print(f"Migration path: {[m.description for m in path]}")

    migrated = manager.migrate("UserContext", data, Version(1, 0, 0), Version(2, 0, 0))
    print(f"Original:    {data}")
    print(f"Migrated:    {migrated}")

    # A bad new version? Roll back with the migration's rollback_fn.
    rolled_back = path[-1].rollback_fn(migrated)
    print(f"Rolled back: {rolled_back}")

    # -------------------------------------------------------------
    # Prompt versioning: register, pin, and record eval metrics
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PROMPT VERSIONING")
    print("-" * 50)

    registry = PromptRegistry()

    registry.register(
        "summarize",
        PromptVersion(
            version=Version(1, 0, 0),
            template="Summarize the following text:\n{text}",
            variables=["text"],
            created_at=datetime.now(),
        ),
    )

    registry.register(
        "summarize",
        PromptVersion(
            version=Version(1, 1, 0),
            template="Provide a {length} summary of:\n{text}",
            variables=["text", "length"],
            created_at=datetime.now(),
        ),
    )

    all_versions = [str(p.version) for p in registry.prompts["summarize"]]
    print(f"Registered versions: {all_versions}")

    prompt = registry.get("summarize")
    print(f"Active prompt: v{prompt.version} (variables: {prompt.variables})")

    # Pin v1.1.0 as active and attach a regression-eval score
    registry.set_active("summarize", "1.1.0")
    registry.record_metrics("summarize", "1.1.0", {"eval_score": 0.92})
    prompt = registry.get("summarize")
    print(f"After pinning: v{prompt.version} (variables: {prompt.variables})")
    print(f"Eval metrics for v1.1.0: {prompt.performance_metrics}")
