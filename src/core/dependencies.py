"""Metric dependency management and validation system."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from enum import Enum
import logging

from .types import MetricType, RAGInput
from .base_metric import BaseMetric
from .logging import get_logger


class DependencyType(Enum):
    """Types of metric dependencies."""
    HARD = "hard"  # Required dependency - metric cannot run without it
    SOFT = "soft"  # Optional dependency - metric can run but works better with it
    DATA = "data"  # Data dependency - metric needs certain input data
    COMPUTATION = "computation"  # Computational dependency - metric uses results from other metrics


class RequirementType(Enum):
    """Types of metric requirements."""
    LLM = "llm"  # Requires an LLM for evaluation
    CONTEXTS = "contexts"  # Requires retrieved contexts
    ANSWER = "answer"  # Requires a generated answer
    QUERY = "query"  # Requires a query
    METADATA = "metadata"  # Requires specific metadata
    PREVIOUS_RESULTS = "previous_results"  # Requires results from other metrics


@dataclass
class MetricRequirement:
    """Defines a specific requirement for a metric."""
    requirement_type: RequirementType
    required: bool = True  # Whether this requirement is mandatory
    description: str = ""
    validation_fn: Optional[callable] = None  # Custom validation function
    
    def validate(self, rag_input: RAGInput, available_results: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Validate if this requirement is met.
        
        Args:
            rag_input: The RAG input to validate
            available_results: Results from previously executed metrics
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.validation_fn:
            return self.validation_fn(rag_input, available_results)
        
        # Default validation logic
        if self.requirement_type == RequirementType.LLM:
            # This would be validated by the metric itself during initialization
            return True, ""
        elif self.requirement_type == RequirementType.CONTEXTS:
            valid = rag_input.retrieved_contexts is not None and len(rag_input.retrieved_contexts) > 0
            return valid, "No retrieved contexts provided" if not valid else ""
        elif self.requirement_type == RequirementType.ANSWER:
            valid = rag_input.answer is not None and rag_input.answer.strip() != ""
            return valid, "No answer provided" if not valid else ""
        elif self.requirement_type == RequirementType.QUERY:
            valid = rag_input.query is not None and rag_input.query.strip() != ""
            return valid, "No query provided" if not valid else ""
        elif self.requirement_type == RequirementType.METADATA:
            valid = rag_input.metadata is not None
            return valid, "No metadata provided" if not valid else ""
        elif self.requirement_type == RequirementType.PREVIOUS_RESULTS:
            if available_results is None:
                return False, "No previous results available"
            return True, ""
        
        return True, ""


@dataclass 
class MetricDependencyRule:
    """Defines a dependency relationship between metrics."""
    dependent_metric: str  # The metric that has the dependency
    dependency_metric: str  # The metric that is depended upon
    dependency_type: DependencyType
    description: str = ""
    condition_fn: Optional[callable] = None  # Optional condition function
    
    def is_applicable(self, context: Dict[str, Any] = None) -> bool:
        """Check if this dependency rule applies in the given context."""
        if self.condition_fn:
            return self.condition_fn(context or {})
        return True


@dataclass
class MetricProfile:
    """Complete profile of a metric including its dependencies and requirements."""
    metric_name: str
    metric_type: MetricType
    requirements: List[MetricRequirement] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # List of metric names this depends on
    provides: List[str] = field(default_factory=list)  # List of capabilities this metric provides
    computational_complexity: str = "medium"  # low, medium, high
    typical_execution_time: float = 1.0  # seconds
    
    def validate_requirements(self, rag_input: RAGInput, available_results: Dict[str, Any] = None) -> List[str]:
        """Validate all requirements for this metric.
        
        Returns:
            List of error messages (empty if all requirements are met)
        """
        errors = []
        
        for requirement in self.requirements:
            if requirement.required:
                is_valid, error_msg = requirement.validate(rag_input, available_results)
                if not is_valid:
                    errors.append(f"{self.metric_name}: {error_msg}")
        
        return errors


class MetricDependencyManager:
    """Manages metric dependencies and validates requirements."""
    
    def __init__(self):
        self.logger = get_logger("dependencies")
        
        # Registry of metric profiles
        self.metric_profiles: Dict[str, MetricProfile] = {}
        
        # Registry of dependency rules
        self.dependency_rules: List[MetricDependencyRule] = []
        
        # Initialize with built-in profiles and rules
        self._initialize_builtin_profiles()
        self._initialize_builtin_rules()
    
    def register_metric_profile(self, profile: MetricProfile):
        """Register a metric profile."""
        self.metric_profiles[profile.metric_name] = profile
        self.logger.debug(f"Registered profile for metric: {profile.metric_name}")
    
    def register_dependency_rule(self, rule: MetricDependencyRule):
        """Register a dependency rule."""
        self.dependency_rules.append(rule)
        self.logger.debug(f"Registered dependency rule: {rule.dependent_metric} -> {rule.dependency_metric}")
    
    def get_metric_dependencies(self, metric_name: str, context: Dict[str, Any] = None) -> List[str]:
        """Get all dependencies for a specific metric.
        
        Args:
            metric_name: Name of the metric
            context: Optional context for conditional dependencies
            
        Returns:
            List of metric names that this metric depends on
        """
        dependencies = set()
        
        # Get static dependencies from profile
        if metric_name in self.metric_profiles:
            dependencies.update(self.metric_profiles[metric_name].dependencies)
        
        # Get dynamic dependencies from rules
        for rule in self.dependency_rules:
            if rule.dependent_metric == metric_name and rule.is_applicable(context):
                dependencies.add(rule.dependency_metric)
        
        return list(dependencies)
    
    def validate_metric_requirements(self, metric_name: str, rag_input: RAGInput, 
                                   available_results: Dict[str, Any] = None) -> List[str]:
        """Validate requirements for a specific metric.
        
        Args:
            metric_name: Name of the metric to validate
            rag_input: The RAG input
            available_results: Results from previously executed metrics
            
        Returns:
            List of error messages (empty if all requirements are met)
        """
        if metric_name not in self.metric_profiles:
            return [f"No profile found for metric: {metric_name}"]
        
        profile = self.metric_profiles[metric_name]
        return profile.validate_requirements(rag_input, available_results)
    
    def get_execution_order(self, metric_names: List[str], context: Dict[str, Any] = None) -> List[List[str]]:
        """Get optimal execution order for a list of metrics.
        
        Args:
            metric_names: List of metric names to order
            context: Optional context for conditional dependencies
            
        Returns:
            List of execution levels (each level can be executed in parallel)
        """
        from collections import defaultdict, deque
        
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Initialize
        for metric_name in metric_names:
            in_degree[metric_name] = 0
        
        # Add dependencies
        for metric_name in metric_names:
            dependencies = self.get_metric_dependencies(metric_name, context)
            for dep in dependencies:
                if dep in metric_names:  # Only consider dependencies that are in our list
                    graph[dep].add(metric_name)
                    in_degree[metric_name] += 1
        
        # Topological sort using Kahn's algorithm
        levels = []
        queue = deque([name for name in metric_names if in_degree[name] == 0])
        
        while queue:
            current_level = list(queue)
            levels.append(current_level)
            queue.clear()
            
            for metric_name in current_level:
                for neighbor in graph[metric_name]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return levels
    
    def analyze_dependencies(self, metric_names: List[str]) -> Dict[str, Any]:
        """Analyze dependencies for a set of metrics.
        
        Args:
            metric_names: List of metric names to analyze
            
        Returns:
            Analysis results including dependency graph, execution order, etc.
        """
        analysis = {
            "metrics": metric_names,
            "dependency_graph": {},
            "execution_order": self.get_execution_order(metric_names),
            "complexity_analysis": {},
            "requirement_summary": {},
            "potential_issues": []
        }
        
        # Build dependency graph
        for metric_name in metric_names:
            dependencies = self.get_metric_dependencies(metric_name)
            analysis["dependency_graph"][metric_name] = dependencies
        
        # Analyze computational complexity
        total_complexity = 0
        complexity_map = {"low": 1, "medium": 2, "high": 3}
        
        for metric_name in metric_names:
            if metric_name in self.metric_profiles:
                profile = self.metric_profiles[metric_name]
                complexity = complexity_map.get(profile.computational_complexity, 2)
                total_complexity += complexity
                
                analysis["complexity_analysis"][metric_name] = {
                    "complexity": profile.computational_complexity,
                    "estimated_time": profile.typical_execution_time
                }
        
        analysis["complexity_analysis"]["total_complexity"] = total_complexity
        analysis["complexity_analysis"]["estimated_total_time"] = sum(
            p.typical_execution_time for p in self.metric_profiles.values() 
            if p.metric_name in metric_names
        )
        
        # Summarize requirements
        all_requirements = set()
        for metric_name in metric_names:
            if metric_name in self.metric_profiles:
                profile = self.metric_profiles[metric_name]
                for req in profile.requirements:
                    all_requirements.add(req.requirement_type.value)
                    
        analysis["requirement_summary"] = {
            "required_capabilities": list(all_requirements),
            "metrics_needing_llm": [m for m in metric_names if self._metric_needs_llm(m)],
            "metrics_needing_contexts": [m for m in metric_names if self._metric_needs_contexts(m)]
        }
        
        # Check for potential issues
        issues = []
        
        # Check for circular dependencies
        if len(analysis["execution_order"]) == 0:
            issues.append("Circular dependencies detected!")
        
        # Check for missing dependencies
        all_deps = set()
        for deps in analysis["dependency_graph"].values():
            all_deps.update(deps)
        
        missing_deps = all_deps - set(metric_names)
        if missing_deps:
            issues.append(f"Missing dependencies: {list(missing_deps)}")
        
        analysis["potential_issues"] = issues
        
        return analysis
    
    def suggest_metric_order(self, available_metrics: List[str], 
                           target_capabilities: List[str] = None) -> List[str]:
        """Suggest an optimal set of metrics for specific evaluation goals.
        
        Args:
            available_metrics: List of available metric names
            target_capabilities: List of desired evaluation capabilities
            
        Returns:
            Recommended list of metrics in execution order
        """
        if target_capabilities is None:
            # Default comprehensive evaluation
            target_capabilities = ["faithfulness", "relevance", "coherence", "completeness"]
        
        selected_metrics = []
        covered_capabilities = set()
        
        # Sort metrics by complexity (prefer simpler metrics first)
        complexity_order = {"low": 0, "medium": 1, "high": 2}
        sorted_metrics = sorted(
            available_metrics,
            key=lambda m: complexity_order.get(
                self.metric_profiles.get(m, MetricProfile("", MetricType.GENERATION)).computational_complexity,
                1
            )
        )
        
        # Select metrics that provide needed capabilities
        for metric_name in sorted_metrics:
            if metric_name in self.metric_profiles:
                profile = self.metric_profiles[metric_name]
                metric_capabilities = set(profile.provides)
                
                # Check if this metric provides any uncovered capabilities
                if metric_capabilities & (set(target_capabilities) - covered_capabilities):
                    selected_metrics.append(metric_name)
                    covered_capabilities.update(metric_capabilities)
                    
                    # Stop when we've covered all target capabilities
                    if covered_capabilities >= set(target_capabilities):
                        break
        
        # Return in execution order
        execution_levels = self.get_execution_order(selected_metrics)
        return [metric for level in execution_levels for metric in level]
    
    def _metric_needs_llm(self, metric_name: str) -> bool:
        """Check if a metric needs an LLM."""
        if metric_name in self.metric_profiles:
            profile = self.metric_profiles[metric_name]
            return any(req.requirement_type == RequirementType.LLM for req in profile.requirements)
        return True  # Default assumption
    
    def _metric_needs_contexts(self, metric_name: str) -> bool:
        """Check if a metric needs retrieved contexts."""
        if metric_name in self.metric_profiles:
            profile = self.metric_profiles[metric_name]
            return any(req.requirement_type == RequirementType.CONTEXTS for req in profile.requirements)
        return False  # Default assumption
    
    def _initialize_builtin_profiles(self):
        """Initialize built-in metric profiles."""
        
        # Generation metrics
        generation_metrics = [
            ("faithfulness", "Evaluates factual consistency with retrieved contexts", ["factual_consistency"]),
            ("answer_relevance", "Evaluates relevance of answer to the question", ["relevance"]),
            ("answer_correctness", "Evaluates correctness of the answer", ["correctness"]),
            ("completeness", "Evaluates completeness of the answer", ["completeness"]),
            ("coherence", "Evaluates logical coherence of the answer", ["coherence"]),
            ("helpfulness", "Evaluates helpfulness of the answer", ["helpfulness"])
        ]
        
        for name, desc, provides in generation_metrics:
            profile = MetricProfile(
                metric_name=name,
                metric_type=MetricType.GENERATION,
                requirements=[
                    MetricRequirement(RequirementType.LLM, required=True, description="Needs LLM for evaluation"),
                    MetricRequirement(RequirementType.QUERY, required=True, description="Needs query for evaluation"),
                    MetricRequirement(RequirementType.ANSWER, required=True, description="Needs answer for evaluation")
                ],
                provides=provides,
                computational_complexity="medium",
                typical_execution_time=2.0
            )
            
            # Add context requirement for faithfulness
            if name == "faithfulness":
                profile.requirements.append(
                    MetricRequirement(RequirementType.CONTEXTS, required=True, description="Needs contexts for consistency check")
                )
            
            self.register_metric_profile(profile)
        
        # Retrieval metrics
        retrieval_metrics = [
            ("context_relevance", "Evaluates relevance of retrieved contexts", ["context_quality"]),
            ("context_precision", "Evaluates precision of context retrieval", ["retrieval_precision"]),
            ("context_recall", "Evaluates recall of context retrieval", ["retrieval_recall"])
        ]
        
        for name, desc, provides in retrieval_metrics:
            profile = MetricProfile(
                metric_name=name,
                metric_type=MetricType.RETRIEVAL,
                requirements=[
                    MetricRequirement(RequirementType.LLM, required=True, description="Needs LLM for evaluation"),
                    MetricRequirement(RequirementType.QUERY, required=True, description="Needs query for evaluation"),
                    MetricRequirement(RequirementType.CONTEXTS, required=True, description="Needs contexts for evaluation")
                ],
                provides=provides,
                computational_complexity="medium",
                typical_execution_time=1.5
            )
            self.register_metric_profile(profile)
        
        # Composite metrics
        composite_metrics = [
            ("llm_judge", "Uses LLM as judge for comprehensive evaluation", ["comprehensive_evaluation"]),
            ("rag_certainty", "Evaluates certainty/confidence of RAG response", ["confidence_assessment"]),
            ("trust_score", "Evaluates trustworthiness through behavioral and factual consistency", ["trustworthiness"])
        ]
        
        for name, desc, provides in composite_metrics:
            profile = MetricProfile(
                metric_name=name,
                metric_type=MetricType.COMPOSITE,
                requirements=[
                    MetricRequirement(RequirementType.LLM, required=True, description="Needs LLM for evaluation"),
                    MetricRequirement(RequirementType.QUERY, required=True, description="Needs query for evaluation"),
                    MetricRequirement(RequirementType.ANSWER, required=True, description="Needs answer for evaluation"),
                    MetricRequirement(RequirementType.CONTEXTS, required=True, description="Needs contexts for evaluation")
                ],
                provides=provides,
                computational_complexity="high",
                typical_execution_time=5.0
            )
            self.register_metric_profile(profile)
    
    def _initialize_builtin_rules(self):
        """Initialize built-in dependency rules."""
        
        # Composite metrics can benefit from basic metrics
        composite_depends_on_generation = [
            ("llm_judge", "faithfulness", "LLM judge can leverage faithfulness assessment"),
            ("llm_judge", "answer_relevance", "LLM judge can leverage relevance assessment"),
            ("rag_certainty", "faithfulness", "Certainty assessment benefits from faithfulness check"),
            ("trust_score", "faithfulness", "Trust score incorporates faithfulness")
        ]
        
        for dependent, dependency, desc in composite_depends_on_generation:
            rule = MetricDependencyRule(
                dependent_metric=dependent,
                dependency_metric=dependency,
                dependency_type=DependencyType.SOFT,
                description=desc
            )
            self.register_dependency_rule(rule)
        
        # Composite metrics can benefit from retrieval metrics
        composite_depends_on_retrieval = [
            ("rag_certainty", "context_relevance", "Certainty assessment benefits from context quality"),
            ("trust_score", "context_precision", "Trust score incorporates retrieval quality")
        ]
        
        for dependent, dependency, desc in composite_depends_on_retrieval:
            rule = MetricDependencyRule(
                dependent_metric=dependent,
                dependency_metric=dependency,
                dependency_type=DependencyType.SOFT,
                description=desc
            )
            self.register_dependency_rule(rule)


# Global instance
_dependency_manager = None

def get_dependency_manager() -> MetricDependencyManager:
    """Get the global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = MetricDependencyManager()
    return _dependency_manager 