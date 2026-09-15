"""
REUSABLE CONTEXT LIBRARIES
Component registries and pipeline builders

Run: python context_library.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Protocol
from abc import ABC, abstractmethod
from datetime import datetime
import json

# =============================================================================
# COMPONENT INTERFACES
# =============================================================================


class ContextComponent(Protocol):
    """Protocol for all context pipeline components."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output."""
        ...

    @property
    def name(self) -> str:
        """Component identifier."""
        ...


@dataclass
class ComponentMetadata:
    """Metadata for registered components."""

    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    author: str = ""
    tags: List[str] = field(default_factory=list)


# =============================================================================
# STANDARD COMPONENTS
# =============================================================================


class ContextRetriever(ABC):
    """Base class for context retrievers."""

    @abstractmethod
    def retrieve(self, query: str, config: Dict[str, Any]) -> List[Dict]:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        config = input_data.get("step_config", {})
        results = self.retrieve(query, config)
        return {"documents": results, "query": query}


class SimpleRetriever(ContextRetriever):
    """Simple keyword-based retriever."""

    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []

    def retrieve(self, query: str, config: Dict[str, Any]) -> List[Dict]:
        limit = config.get("limit", 5)
        results = []

        # Keyword match: keep any doc containing at least one query term.
        query_terms = query.lower().split()
        for doc in self.documents:
            content = doc.get("content", "").lower()
            if any(term in content for term in query_terms):
                results.append(doc)
                if len(results) >= limit:
                    break

        return results


class ContextRanker(ABC):
    """Base class for context rankers."""

    @abstractmethod
    def rank(self, documents: List[Dict], query: str) -> List[Dict]:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = input_data.get("documents", [])
        query = input_data.get("query", "")
        ranked = self.rank(documents, query)
        return {"documents": ranked, "query": query}


class RelevanceRanker(ContextRanker):
    """Rank by keyword relevance."""

    def rank(self, documents: List[Dict], query: str) -> List[Dict]:
        query_terms = set(query.lower().split())

        def score(doc: Dict) -> float:
            content = doc.get("content", "").lower()
            content_terms = set(content.split())
            overlap = len(query_terms & content_terms)
            return overlap / max(len(query_terms), 1)

        scored = [(doc, score(doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [{**doc, "relevance_score": s} for doc, s in scored]


class ContextFilter(ABC):
    """Base class for context filters."""

    @abstractmethod
    def filter(self, documents: List[Dict], criteria: Dict) -> List[Dict]:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = input_data.get("documents", [])
        criteria = input_data.get("step_config", {})
        filtered = self.filter(documents, criteria)
        return {"documents": filtered}


class MetadataFilter(ContextFilter):
    """Filter by document metadata."""

    def filter(self, documents: List[Dict], criteria: Dict) -> List[Dict]:
        results = []

        for doc in documents:
            matches = True
            metadata = doc.get("metadata", {})

            for key, value in criteria.items():
                if metadata.get(key) != value:
                    matches = False
                    break

            if matches:
                results.append(doc)

        return results


class ContextFormatter(ABC):
    """Base class for context formatters."""

    @abstractmethod
    def format(self, documents: List[Dict], config: Dict) -> str:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = input_data.get("documents", [])
        config = input_data.get("step_config", {})
        formatted = self.format(documents, config)
        return {"formatted_context": formatted, "document_count": len(documents)}


class MarkdownFormatter(ContextFormatter):
    """Format context as markdown."""

    def format(self, documents: List[Dict], config: Dict) -> str:
        include_metadata = config.get("include_metadata", True)
        sections = []

        for i, doc in enumerate(documents, 1):
            section = f"## Document {i}\n\n"
            section += doc.get("content", "")

            if include_metadata:
                metadata = doc.get("metadata", {})
                if metadata:
                    section += "\n\n**Metadata:**\n"
                    for k, v in metadata.items():
                        section += f"- {k}: {v}\n"

            sections.append(section)

        return "\n\n---\n\n".join(sections)


# =============================================================================
# COMPONENT REGISTRY
# =============================================================================


class ComponentRegistry:
    """Registry for reusable components."""

    def __init__(self):
        self.components: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, ComponentMetadata] = {}

    def register(
        self, component: ContextComponent, metadata: ComponentMetadata = None
    ) -> None:
        name = component.name
        self.components[name] = {
            "instance": component,
            "registered_at": datetime.now().isoformat(),
        }

        if metadata:
            self.metadata[name] = metadata

    def get(self, name: str) -> Optional[ContextComponent]:
        entry = self.components.get(name)
        return entry["instance"] if entry else None

    def list_components(self, tag: str = None) -> List[str]:
        if not tag:
            return list(self.components.keys())

        return [name for name, meta in self.metadata.items() if tag in meta.tags]

    def get_metadata(self, name: str) -> Optional[ComponentMetadata]:
        return self.metadata.get(name)


# =============================================================================
# PIPELINE BUILDER
# =============================================================================


@dataclass
class PipelineStep:
    component_name: str
    config: Dict[str, Any] = field(default_factory=dict)


class ContextPipelineBuilder:
    """Build context pipelines from components."""

    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.steps: List[PipelineStep] = []

    def add(
        self, component_name: str, config: Dict[str, Any] = None
    ) -> "ContextPipelineBuilder":
        self.steps.append(
            PipelineStep(component_name=component_name, config=config or {})
        )
        return self

    def build(self) -> "ContextPipeline":
        components = []
        for step in self.steps:
            component = self.registry.get(step.component_name)
            if not component:
                raise ValueError(f"Unknown component: {step.component_name}")
            components.append((component, step.config))

        return ContextPipeline(components)

    def clear(self) -> "ContextPipelineBuilder":
        self.steps = []
        return self


class ContextPipeline:
    """Executable context pipeline."""

    def __init__(self, components: List[tuple]):
        self.components = components

    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        current = initial_input.copy()

        for component, config in self.components:
            # Hand this step its config under a single well-known key that every
            # component reads (input_data.get("step_config", {})).
            current["step_config"] = config
            current = component.process(current)

        return current

    def execute_with_trace(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with step-by-step tracing."""
        trace = []
        current = initial_input.copy()

        for component, config in self.components:
            current["step_config"] = config
            step_input = current.copy()
            current = component.process(current)

            trace.append(
                {
                    "component": component.name,
                    "input": step_input,
                    "output": current.copy(),
                }
            )

        return {"result": current, "trace": trace}


# =============================================================================
# PIPELINE TEMPLATES
# =============================================================================


class PipelineTemplate:
    """Reusable pipeline template."""

    def __init__(self, name: str, steps: List[PipelineStep]):
        self.name = name
        self.steps = steps

    def instantiate(self, registry: ComponentRegistry) -> ContextPipeline:
        builder = ContextPipelineBuilder(registry)
        for step in self.steps:
            builder.add(step.component_name, step.config)
        return builder.build()


class TemplateLibrary:
    """Library of pipeline templates."""

    def __init__(self):
        self.templates: Dict[str, PipelineTemplate] = {}

    def register(self, template: PipelineTemplate) -> None:
        self.templates[template.name] = template

    def get(self, name: str) -> Optional[PipelineTemplate]:
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REUSABLE CONTEXT LIBRARIES")
    print("=" * 60)

    # Create registry
    registry = ComponentRegistry()

    # Register components
    print("\n" + "-" * 50)
    print("REGISTERING COMPONENTS")
    print("-" * 50)

    documents = [
        {
            "id": "1",
            "content": "Python is a programming language",
            "metadata": {"topic": "programming"},
        },
        {
            "id": "2",
            "content": "Machine learning uses Python",
            "metadata": {"topic": "ML"},
        },
        {"id": "3", "content": "AI systems process data", "metadata": {"topic": "AI"}},
    ]

    retriever = SimpleRetriever(documents)
    ranker = RelevanceRanker()
    filter_comp = MetadataFilter()
    formatter = MarkdownFormatter()

    registry.register(
        retriever,
        ComponentMetadata(
            name="SimpleRetriever",
            version="1.0.0",
            description="Keyword-based document retrieval",
            input_schema={"query": "string"},
            output_schema={"documents": "array"},
            tags=["retrieval", "search"],
        ),
    )
    registry.register(ranker)
    registry.register(filter_comp)
    registry.register(formatter)

    print(f"Registered: {registry.list_components()}")

    # Build pipeline
    print("\n" + "-" * 50)
    print("BUILDING PIPELINE")
    print("-" * 50)

    pipeline = (
        ContextPipelineBuilder(registry)
        .add("SimpleRetriever", {"limit": 10})
        .add("RelevanceRanker")
        .add("MarkdownFormatter", {"include_metadata": True})
        .build()
    )

    # Execute pipeline
    print("\n" + "-" * 50)
    print("EXECUTING PIPELINE")
    print("-" * 50)

    result = pipeline.execute({"query": "Python programming"})
    print(f"Document count: {result['document_count']}")
    print(f"\nFormatted output:\n{result['formatted_context'][:500]}...")

    # Execute with trace
    print("\n" + "-" * 50)
    print("PIPELINE TRACING")
    print("-" * 50)

    traced = pipeline.execute_with_trace({"query": "machine learning"})
    print(f"Steps executed: {len(traced['trace'])}")
    for step in traced["trace"]:
        print(f"  {step['component']}")

    # Template library
    print("\n" + "-" * 50)
    print("PIPELINE TEMPLATES")
    print("-" * 50)

    library = TemplateLibrary()
    library.register(
        PipelineTemplate(
            name="basic_rag",
            steps=[
                PipelineStep("SimpleRetriever", {"limit": 5}),
                PipelineStep("RelevanceRanker"),
                PipelineStep("MarkdownFormatter"),
            ],
        )
    )

    print(f"Templates: {library.list_templates()}")

    template = library.get("basic_rag")
    if template:
        rag_pipeline = template.instantiate(registry)
        result = rag_pipeline.execute({"query": "AI data"})
        print(f"Template execution: {result['document_count']} documents")
