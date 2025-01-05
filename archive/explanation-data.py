import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from datetime import datetime


class ConfidenceScorer:
    """Calculates realistic confidence scores based on multiple factors"""
    
    def __init__(self):
        # Base weights for different factors affecting confidence
        self.factor_weights = {
            'data_quality': 0.3,
            'context_match': 0.25,
            'precedent_strength': 0.25,
            'complexity': 0.2
        }
        
        # Complexity penalties by domain
        self.domain_complexity = {
            'product_recommendation': 0.1,  # Relatively straightforward
            'career_advice': 0.15,
            'ethical_dilemmas': 0.25,      # More complex
            'argument_generation': 0.2,
            'mcdm': 0.18,
            'educational_support': 0.15,
            'policy_analysis': 0.22
        }

    def calculate_confidence(self, domain: str, factors: Dict[str, float]) -> float:
        """
        Calculate confidence score based on multiple factors
        Returns a score between 0 and 1
        """
        # Base score starting point
        base_score = 0.9  # Start high and apply penalties
        
        # Apply complexity penalty
        complexity_penalty = self.domain_complexity.get(domain, 0.2)
        base_score -= complexity_penalty
        
        # Calculate weighted factor scores
        weighted_score = sum(
            self.factor_weights[factor] * value 
            for factor, value in factors.items()
        )
        
        # Combine base score and weighted factors
        final_score = (base_score + weighted_score) / 2
        
        # Apply sigmoid to ensure score stays between 0 and 1
        final_score = 1 / (1 + np.exp(-10 * (final_score - 0.5)))
        
        # Convert to percentage and ensure reasonable bounds
        percentage = min(max(final_score * 100, 65), 98)
        
        return round(percentage, 1)

    def get_step_confidences(self, domain: str, complexity: float) -> Dict[str, float]:
        """Calculate confidence scores for each step of the explanation"""
        base_factors = {
            'data_quality': np.random.beta(8, 2),  # Usually high
            'context_match': np.random.beta(7, 3),
            'precedent_strength': np.random.beta(6, 2),
            'complexity': 1 - complexity
        }
        
        # Different steps have different base confidences
        confidences = {}
        
        # Input parsing usually has high confidence
        input_factors = base_factors.copy()
        input_factors['data_quality'] *= 1.1
        confidences['input_parsing'] = self.calculate_confidence(domain, input_factors)
        
        # Feature identification slightly lower
        feature_factors = base_factors.copy()
        feature_factors['complexity'] *= 0.95
        confidences['feature_identification'] = self.calculate_confidence(domain, feature_factors)
        
        # Logical inference most affected by complexity
        inference_factors = base_factors.copy()
        inference_factors['complexity'] *= 0.9
        confidences['logical_inference'] = self.calculate_confidence(domain, inference_factors)
        
        return confidences

@dataclass
class DomainTemplate:
    query_templates: List[str]
    response_templates: List[str]
    stakeholders: List[str]
    evidence_types: List[str]
    pros_templates: List[str]
    cons_templates: List[str]
    impact_areas: List[str]
    difficulty_weights: Dict[str, float] = field(default_factory=dict)
    certainty_modifiers: Dict[str, float] = field(default_factory=dict)

class ComprehensiveExplanationGenerator:
    def __init__(self):
        self.setup_domain_data()
        
    def setup_domain_data(self):
        """Initialize domain-specific templates and data"""
        self.domains = {
            "career_advice": DomainTemplate(
                query_templates=[
                    "Should I transition from {current_role} to {target_role}?",
                    "Is pursuing a career in {field} worth it in 2024?",
                    "How can I advance my career in {industry}?"
                ],
                response_templates=[
                    "Based on market analysis and your background, transitioning to {target_role} would be {assessment}.",
                    "A career in {field} shows {outlook} potential, with {key_factor} being the primary advantage.",
                    "Advancing in {industry} requires focusing on {skill_focus} and {strategy}."
                ],
                stakeholders=[
                    "job seekers", "employers", "industry experts",
                    "career counselors", "HR professionals", "mentors"
                ],
                evidence_types=[
                    "salary surveys", "employment statistics",
                    "industry growth projections", "skills demand data",
                    "professional testimonials", "market trend analysis"
                ],
                pros_templates=[
                    "strong market demand",
                    "competitive compensation",
                    "work-life balance",
                    "career growth potential",
                    "skill development opportunities"
                ],
                cons_templates=[
                    "high entry barriers",
                    "market saturation",
                    "intense competition",
                    "rapid technological changes",
                    "certification requirements"
                ],
                impact_areas=[
                    "career growth", "work satisfaction",
                    "skill development", "market value",
                    "professional network", "industry impact"
                ],
                difficulty_weights={
                    'market_analysis': 0.3,
                    'skill_assessment': 0.3,
                    'future_projection': 0.4
                }
            ),
            
            "product_recommendation": DomainTemplate(
                query_templates=[
                    "Which {product_category} is best for {use_case}?",
                    "Should I invest in {product_name} for {purpose}?",
                    "What's the best {product_type} under {budget}?"
                ],
                response_templates=[
                    "For {use_case}, the {product_name} offers the best value due to {features}.",
                    "Given your requirements, investing in {product_name} would be {assessment}.",
                    "Within your budget of {budget}, {product_name} provides optimal {benefits}."
                ],
                stakeholders=[
                    "consumers", "manufacturers", "experts",
                    "reviewers", "retailers", "industry analysts"
                ],
                evidence_types=[
                    "product reviews", "technical specifications",
                    "comparison tests", "user feedback",
                    "expert evaluations", "performance metrics"
                ],
                pros_templates=[
                    "superior performance",
                    "excellent value",
                    "reliable build quality",
                    "innovative features",
                    "user-friendly design"
                ],
                cons_templates=[
                    "premium pricing",
                    "limited availability",
                    "learning curve",
                    "maintenance requirements",
                    "compatibility issues"
                ],
                impact_areas=[
                    "performance", "value for money",
                    "user experience", "reliability",
                    "future-proofing", "ecosystem compatibility"
                ],
                difficulty_weights={
                    'feature_analysis': 0.25,
                    'price_comparison': 0.25,
                    'user_needs_match': 0.5
                }
            ),

            "ethical_dilemmas": DomainTemplate(
                query_templates=[
                    "Should we prioritize {option1} over {option2} in terms of funding?",
                    "Is it ethical to {action} when considering {context}?",
                    "How should we balance {value1} with {value2}?"
                ],
                response_templates=[
                    "Based on careful ethical analysis, prioritizing {option1} appears more justified because {reason}.",
                    "The ethical course would be to {recommendation} due to {justification}.",
                    "A balanced approach favoring {option1} while maintaining {option2} would be most ethical."
                ],
                stakeholders=[
                    "local communities", "future generations", "vulnerable populations",
                    "minority groups", "economic stakeholders", "global community"
                ],
                evidence_types=[
                    "ethical frameworks", "case studies", "impact assessments",
                    "expert opinions", "research studies", "historical precedents"
                ],
                pros_templates=[
                    "maximizes overall welfare",
                    "promotes equity and fairness",
                    "ensures long-term sustainability",
                    "protects vulnerable groups",
                    "maintains ethical integrity"
                ],
                cons_templates=[
                    "potential short-term challenges",
                    "implementation difficulties",
                    "resource allocation concerns",
                    "stakeholder resistance",
                    "unintended consequences"
                ],
                impact_areas=[
                    "social equity", "environmental sustainability",
                    "economic stability", "cultural preservation",
                    "public health", "educational access"
                ]
            ),
            
            "argument_generation": DomainTemplate(
                query_templates=[
                    "What are the arguments for and against {policy}?",
                    "Debate the merits of {system} versus {alternative_system}.",
                    "Analyze the implications of implementing {proposal}."
                ],
                response_templates=[
                    "Analysis reveals multiple perspectives on {policy}, with key considerations being {factors}.",
                    "The debate around {system} centers on {core_issues}, with evidence suggesting {conclusion}.",
                    "Implementing {proposal} would have {impact_type} implications, primarily affecting {areas}."
                ],
                stakeholders=[
                    "policy makers", "citizens", "businesses",
                    "workers", "economists", "social scientists"
                ],
                evidence_types=[
                    "economic studies", "social research",
                    "expert analyses", "comparative studies",
                    "statistical data", "policy papers"
                ],
                pros_templates=[
                    "economic efficiency gains",
                    "social welfare improvements",
                    "systemic stability",
                    "innovation potential",
                    "equity enhancement"
                ],
                cons_templates=[
                    "implementation costs",
                    "transition challenges",
                    "potential inequities",
                    "system disruption",
                    "resource constraints"
                ],
                impact_areas=[
                    "economic growth", "social cohesion",
                    "institutional stability", "market dynamics",
                    "public services", "private sector"
                ]
            ),
            
            "mcdm": DomainTemplate(
                query_templates=[
                    "Which {option_type} best balances {criterion1} and {criterion2}?",
                    "How should we evaluate {options} based on {criteria}?",
                    "What's the optimal choice for {context} considering {factors}?"
                ],
                response_templates=[
                    "Analysis of {criteria} suggests {recommendation} as the optimal choice.",
                    "Considering all factors, {option} provides the best balance of {benefits}.",
                    "Based on {methodology}, {choice} ranks highest in meeting the criteria."
                ],
                stakeholders=[
                    "investors", "managers", "analysts",
                    "stakeholders", "decision-makers", "experts"
                ],
                evidence_types=[
                    "quantitative analysis", "risk assessments",
                    "performance metrics", "comparative studies",
                    "expert evaluations", "historical data"
                ],
                pros_templates=[
                    "superior performance metrics",
                    "balanced risk-reward profile",
                    "strong historical track record",
                    "favorable market position",
                    "competitive advantages"
                ],
                cons_templates=[
                    "higher initial costs",
                    "implementation complexity",
                    "market uncertainties",
                    "resource requirements",
                    "operational challenges"
                ],
                impact_areas=[
                    "financial returns", "risk exposure",
                    "operational efficiency", "market position",
                    "strategic alignment", "resource utilization"
                ]
            ),
            
            "educational_support": DomainTemplate(
                query_templates=[
                    "Should I pursue {field1} or {field2} for better career prospects?",
                    "What educational path best prepares for {career}?",
                    "How should I choose between {program1} and {program2}?"
                ],
                response_templates=[
                    "Based on market trends and your interests, {recommendation} would be most beneficial.",
                    "Analysis suggests {path} as the optimal educational route for {goals}.",
                    "Considering industry demands, {choice} offers better long-term prospects."
                ],
                stakeholders=[
                    "students", "educators", "employers",
                    "industry experts", "career counselors", "alumni"
                ],
                evidence_types=[
                    "employment statistics", "salary data",
                    "industry projections", "alumni success stories",
                    "market demand analysis", "skill gap studies"
                ],
                pros_templates=[
                    "strong job market demand",
                    "competitive salary potential",
                    "career growth opportunities",
                    "skill transferability",
                    "industry recognition"
                ],
                cons_templates=[
                    "intensive study requirements",
                    "competitive field",
                    "rapid technological changes",
                    "initial salary constraints",
                    "work-life balance challenges"
                ],
                impact_areas=[
                    "career prospects", "skill development",
                    "industry relevance", "personal growth",
                    "professional network", "work satisfaction"
                ]
            ),
            
            "policy_analysis": DomainTemplate(
                query_templates=[
                    "What would be the impact of {policy} on {sector}?",
                    "Should the government implement {measure} to address {issue}?",
                    "How effective would {intervention} be in solving {problem}?"
                ],
                response_templates=[
                    "Analysis indicates that {policy} would {impact} with {confidence}% certainty.",
                    "Implementation of {measure} would likely result in {outcomes}.",
                    "Evidence suggests {intervention} would be {effectiveness} in addressing {problem}."
                ],
                stakeholders=[
                    "government officials", "citizens", "businesses",
                    "policy experts", "researchers", "affected communities"
                ],
                evidence_types=[
                    "policy research", "economic models",
                    "impact studies", "public opinion data",
                    "comparative analyses", "expert testimonies"
                ],
                pros_templates=[
                    "positive societal impact",
                    "economic benefits",
                    "environmental protection",
                    "social equity improvement",
                    "systemic efficiency"
                ],
                cons_templates=[
                    "implementation costs",
                    "administrative challenges",
                    "potential resistance",
                    "transition period effects",
                    "resource requirements"
                ],
                impact_areas=[
                    "public welfare", "economic stability",
                    "social equity", "environmental protection",
                    "institutional effectiveness", "market dynamics"
                ]
            )
        }

    def generate_explanation(self, domain: str, query: str, response: str) -> str:
        """Generate structured explanation for a given domain"""
        template = self.domains[domain]
        
        # Get confidence scores from the confidence scorer
        base_complexity = len(query.split()) / 50
        domain_complexity = self.confidence_scorer.domain_complexity.get(domain, 0.2)
        total_complexity = (base_complexity + domain_complexity) / 2
        confidence_scores = self.confidence_scorer.get_step_confidences(domain, total_complexity)
        
        explanation = f"""1. Input Parsing (Confidence: {confidence_scores['input_parsing']}%):
    * Query Analysis:
        - Primary focus: {random.choice(template.impact_areas)}
        - Context: {random.choice(template.stakeholders)}
        - Key considerations requested: {', '.join(random.sample(template.impact_areas, 2))}

    2. Feature Identification (Confidence: {confidence_scores['feature_identification']}%):
    * Key Stakeholders:
        - Primary: {', '.join(random.sample(template.stakeholders, 2))}
        - Secondary: {', '.join(random.sample(template.stakeholders, 2))}
    * Evidence Base:
        - {random.choice(template.evidence_types)}: Supporting data shows {random.choice(template.pros_templates)}
        - {random.choice(template.evidence_types)}: Indicates {random.choice(template.impact_areas)}

    3. Logical Inference (Confidence: {confidence_scores['logical_inference']}%):
    * Arguments For:
        - {random.choice(template.pros_templates)}
        - {random.choice(template.pros_templates)}
    * Arguments Against:
        - {random.choice(template.cons_templates)}
        - {random.choice(template.cons_templates)}
    * Impact Assessment:
        - Primary effects: {random.choice(template.impact_areas)}
        - Secondary effects: {random.choice(template.impact_areas)}
    * Recommendation Basis:
        - Key factor: {random.choice(template.pros_templates)}
        - Supporting evidence: {random.choice(template.evidence_types)}
        - Confidence level: {confidence_scores['logical_inference']}%"""
        
        return explanation

def generate_dataset(self, num_examples: int, output_file: str):
    """Generate complete dataset with examples from all domains"""
    dataset = []
    domains = list(self.domains.keys())
    examples_per_domain = num_examples // len(domains)
    
    for domain in tqdm(domains, desc="Generating domain examples"):
        template = self.domains[domain]
        
        for _ in range(examples_per_domain):
            query = random.choice(template.query_templates)
            response = random.choice(template.response_templates)
            
            # Fill in template variables
            variables = {
                'option1': random.choice(template.impact_areas),
                'option2': random.choice(template.impact_areas),
                'action': random.choice(['implement', 'regulate', 'support', 'restrict']),
                'context': random.choice(template.impact_areas),
                'value1': random.choice(template.pros_templates),
                'value2': random.choice(template.pros_templates),
                'reason': random.choice(template.pros_templates),
                'recommendation': random.choice(['prioritize', 'balance', 'focus on']),
                'justification': random.choice(template.pros_templates),
                'field1': random.choice(['data science', 'AI', 'cybersecurity', 'cloud computing']),
                'field2': random.choice(['software engineering', 'machine learning', 'web development']),
                'product_category': random.choice(['laptop', 'smartphone', 'camera', 'headphones']),
                'use_case': random.choice(['professional work', 'gaming', 'content creation', 'daily use']),
                'current_role': random.choice(['developer', 'analyst', 'manager', 'consultant']),
                'target_role': random.choice(['data scientist', 'product manager', 'solution architect', 'team lead']),
                'industry': random.choice(['tech', 'finance', 'healthcare', 'education']),
                'assessment': random.choice(['highly recommended', 'worth considering', 'challenging but rewarding']),
                'outlook': random.choice(['positive', 'promising', 'strong']),
                'key_factor': random.choice(['market demand', 'growth potential', 'skill alignment']),
                'skill_focus': random.choice(['technical skills', 'leadership abilities', 'domain expertise']),
                'strategy': random.choice(['continuous learning', 'networking', 'specialization'])
            }
            
            # Format query and response with variables
            query = query.format(**variables)
            response = response.format(**variables)
            
            example = {
                "domain": domain,
                "query": query,
                "response": response,
                "explanation": self.generate_explanation(domain, query, response),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "domain_specific_factors": random.sample(template.impact_areas, 3),
                    "evidence_types_used": random.sample(template.evidence_types, 2)
                }
            }
            
            dataset.append(example)
    
    # Save dataset
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} examples and saved to {output_file}")
    
    # Print statistics
    self._print_dataset_statistics(dataset)

def _print_dataset_statistics(self, dataset: List[Dict]):
    """Print useful statistics about the generated dataset"""
    print("\nDataset Statistics:")
    print(f"Total examples: {len(dataset)}")
    
    # Domain distribution
    print("\nDomain Distribution:")
    domain_counts = {}
    for example in dataset:
        domain_counts[example['domain']] = domain_counts.get(example['domain'], 0) + 1
    
    for domain, count in domain_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"{domain}: {count} examples ({percentage:.1f}%)")

if __name__ == "__main__":
    # Generate comprehensive dataset
    generator = ComprehensiveExplanationGenerator()
    generator.generate_dataset(30000, "comprehensive_explanation_data.json")
    
    # Print a sample example
    with open("comprehensive_explanation_data.json", 'r') as f:
        data = json.load(f)
        print("\nSample Example:")
        print(json.dumps(data[0], indent=2))
