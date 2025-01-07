from typing import List, Dict, Any, Tuple, Optional
import random
import spacy
from textblob import TextBlob
from collections import defaultdict
import numpy as np

class EvidenceBase:
    """Manages evidence and citations for explanations"""
    
    def __init__(self):
        self.evidence_database = {
            'ethical_dilemmas': {
                'research_papers': [
                    ('Ethics Review 2024', 'Study on ethical decision-making in resource allocation'),
                    ('Journal of Applied Ethics', 'Impact assessment of environmental vs educational funding'),
                    ('Social Policy Quarterly', 'Long-term effects of public policy choices')
                ],
                'case_studies': [
                    ('Nordic Model', 'Successful balance of social welfare and economic growth'),
                    ('Singapore Education', 'Educational investment impact on economic development'),
                    ('Costa Rica Environment', 'Environmental protection economic benefits')
                ],
                'statistics': [
                    ('World Bank 2023', 'Environmental degradation costs: 5-10% of GDP annually'),
                    ('OECD Education', 'Every $1 in education returns $8-12 in economic growth'),
                    ('UN Development', 'Sustainable development impact metrics')
                ]
            },
            'product_recommendation': {
                'benchmarks': [
                    ('TechBench 2024', 'Comprehensive performance metrics'),
                    ('Consumer Labs', 'Reliability and durability testing'),
                    ('Efficiency Ratings', 'Energy consumption and cost analysis')
                ],
                'user_studies': [
                    ('User Experience Lab', 'Long-term usage patterns'),
                    ('Professional Reviews', 'Expert evaluation metrics'),
                    ('Market Research', 'Customer satisfaction data')
                ],
                'comparative_data': [
                    ('Industry Standards', 'Performance benchmarks by category'),
                    ('Price Analysis', 'Value for money metrics'),
                    ('Feature Matrix', 'Comprehensive comparison data')
                ]
            },
            'mcdm': {
                'analytical_frameworks': [
                    ('Decision Theory Journal', 'Multi-criteria optimization methods'),
                    ('Risk Analysis', 'Quantitative risk assessment models'),
                    ('Investment Analytics', 'Portfolio optimization metrics')
                ],
                'market_data': [
                    ('Market Intelligence', 'Sector-specific performance data'),
                    ('Industry Reports', 'Growth and trend analysis'),
                    ('Economic Indicators', 'Market condition metrics')
                ]
            },
            'argument_generation': {
                'academic_research': [
                    ('Policy Research', 'Evidence-based policy analysis'),
                    ('Economic Review', 'Impact assessment studies'),
                    ('Social Science Data', 'Empirical research findings')
                ],
                'expert_opinions': [
                    ('Expert Panel', 'Professional consensus views'),
                    ('Industry Leaders', 'Professional insights'),
                    ('Academic Consensus', 'Scholarly agreement points')
                ]
            }
        }
        
    def get_relevant_evidence(self, domain: str, context: str, num_pieces: int = 2) -> List[Dict]:
        """Get contextually relevant evidence"""
        if domain not in self.evidence_database:
            return []
            
        evidence_pool = []
        for category, items in self.evidence_database[domain].items():
            for source, description in items:
                relevance_score = self._calculate_relevance(context, description)
                evidence_pool.append({
                    'source': source,
                    'description': description,
                    'category': category,
                    'relevance': relevance_score
                })
        
        # Sort by relevance and return top pieces
        evidence_pool.sort(key=lambda x: x['relevance'], reverse=True)
        return evidence_pool[:num_pieces]
    
    def _calculate_relevance(self, context: str, evidence: str) -> float:
        """Calculate relevance score between context and evidence"""
        # Simple word overlap scoring - could be  with embeddings
        context_words = set(context.lower().split())
        evidence_words = set(evidence.lower().split())
        overlap = len(context_words.intersection(evidence_words))
        return overlap / len(context_words)

class CounterfactualGenerator:
    """ counterfactual generation with more  logic"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.setup_domain_templates()
        
    def setup_domain_templates(self):
        """Setup  domain-specific templates and modifiers"""
        self.domain_templates = {
            'ethical_dilemmas': {
                'stakeholder_impact': {
                    'template': "What if {stakeholder} were disproportionately affected by {decision}?",
                    'variables': {
                        'stakeholder': [
                            'vulnerable populations',
                            'future generations',
                            'local communities',
                            'minority groups',
                            'developing nations'
                        ],
                        'decision': [
                            'this policy',
                            'the resource allocation',
                            'the proposed solution',
                            'the implementation timeline'
                        ]
                    }
                },
                'resource_constraints': {
                    'template': "How would the decision change with {constraint} in {timeframe}?",
                    'variables': {
                        'constraint': [
                            'limited budget',
                            'resource scarcity',
                            'time pressure',
                            'technological limitations'
                        ],
                        'timeframe': [
                            'immediate term',
                            'medium term',
                            'long term'
                        ]
                    }
                }
            },
            'product_recommendation': {
                'use_case_variation': {
                    'template': "Consider using {product} in {context} with {requirement}",
                    'variables': {
                        'context': [
                            'professional environment',
                            'educational setting',
                            'home office',
                            'travel scenario'
                        ],
                        'requirement': [
                            'high performance demands',
                            'limited infrastructure',
                            'strict budget constraints',
                            'specific compatibility needs'
                        ]
                    }
                },
                'feature_priority': {
                    'template': "What if {feature} was the primary consideration over {other_feature}?",
                    'variables': {
                        'feature': [
                            'reliability',
                            'performance',
                            'cost-effectiveness',
                            'user experience'
                        ],
                        'other_feature': [
                            'initial cost',
                            'brand reputation',
                            'advanced features',
                            'aesthetic design'
                        ]
                    }
                }
            }
        }
        
    def generate_counterfactuals(self, 
                               query: str, 
                               domain: str,
                               context: Dict[str, Any] = None) -> List[Dict]:
        """Generate  counterfactuals based on query analysis"""
        counterfactuals = []
        
        # Parse query to understand key elements
        doc = self.nlp(query)
        key_elements = self._extract_key_elements(doc)
        
        # Get domain-specific templates
        domain_temps = self.domain_templates.get(domain, {})
        
        for template_type, template_data in domain_temps.items():
            # Generate base counterfactual
            cf = self._generate_base_counterfactual(
                template_data['template'],
                template_data['variables'],
                key_elements
            )
            
            # Add contextual variations
            variations = self._generate_contextual_variations(cf, context)
            counterfactuals.extend(variations)
        
        return counterfactuals
    
    def _extract_key_elements(self, doc) -> Dict[str, List[str]]:
        """Extract key elements from query using spaCy"""
        elements = {
            'entities': [],
            'key_phrases': [],
            'actions': []
        }
        
        for ent in doc.ents:
            elements['entities'].append(ent.text)
            
        for chunk in doc.noun_chunks:
            elements['key_phrases'].append(chunk.text)
            
        for token in doc:
            if token.pos_ == "VERB":
                elements['actions'].append(token.text)
                
        return elements
    
    def _generate_base_counterfactual(self, 
                                    template: str,
                                    variables: Dict[str, List[str]],
                                    key_elements: Dict[str, List[str]]) -> str:
        """Generate base counterfactual from template"""
        filled_template = template
        
        for var_name, var_values in variables.items():
            if '{' + var_name + '}' in template:
                # Try to find contextually relevant value
                relevant_value = self._find_relevant_value(
                    var_values,
                    key_elements
                )
                filled_template = filled_template.replace(
                    '{' + var_name + '}',
                    relevant_value or random.choice(var_values)
                )
                
        return filled_template
    
    def _find_relevant_value(self, 
                           values: List[str],
                           key_elements: Dict[str, List[str]]) -> Optional[str]:
        """Find most relevant value based on key elements"""
        max_score = 0
        best_value = None
        
        for value in values:
            score = 0
            for element_list in key_elements.values():
                for element in element_list:
                    if element.lower() in value.lower():
                        score += 1
            
            if score > max_score:
                max_score = score
                best_value = value
                
        return best_value
    
    def _generate_contextual_variations(self, 
                                     base_cf: str,
                                     context: Optional[Dict[str, Any]]) -> List[Dict]:
        """Generate variations based on context"""
        variations = [{'counterfactual': base_cf, 'type': 'base'}]
        
        if context:
            # Add context-specific variations
            if 'constraints' in context:
                variations.append({
                    'counterfactual': f"Under {context['constraints']}, {base_cf}",
                    'type': 'constraint_based'
                })
                
            if 'objectives' in context:
                variations.append({
                    'counterfactual': f"If prioritizing {context['objectives']}, {base_cf}",
                    'type': 'objective_based'
                })
                
        return variations

class Prober:
    """ probing question generator with  analysis"""
    
    def __init__(self, evidence_base: EvidenceBase):
        self.nlp = spacy.load('en_core_web_sm')
        self.evidence_base = evidence_base
        self.setup_probe_patterns()
        
    def setup_probe_patterns(self):
        """Setup  probing patterns and templates"""
        self.probe_patterns = {
            'assumption_testing': {
                'template': "What if the assumption about {assumption} doesn't hold due to {factor}?",
                'triggers': ['assumes', 'based on', 'given that', 'considering']
            },
            'evidence_probing': {
                'template': "How reliable is the evidence from {source} regarding {aspect}?",
                'triggers': ['research shows', 'studies indicate', 'data suggests']
            },
            'impact_analysis': {
                'template': "What would be the {timeframe} impact on {stakeholder} if {condition}?",
                'timeframes': ['immediate', 'short-term', 'long-term'],
                'triggers': ['affects', 'impacts', 'influences']
            },
            'alternative_exploration': {
                'template': "Have you considered {alternative} as a way to {objective}?",
                'triggers': ['could', 'might', 'potentially']
            }
        }
        
        self.domain_specific_probes = {
            'ethical_dilemmas': [
                "How does this balance competing rights between {stakeholder1} and {stakeholder2}?",
                "What precedents exist for similar ethical decisions in {context}?",
                "How might this decision affect social equity in {timeframe}?"
            ],
            'product_recommendation': [
                "What reliability data exists for {feature} in {context}?",
                "How does {product} compare to emerging alternatives in terms of {aspect}?",
                "What are the hidden costs or requirements for {feature}?"
            ],
            'mcdm': [
                "How sensitive is this recommendation to changes in {factor}?",
                "What risk mitigation strategies exist for {risk_factor}?",
                "How does this align with {stakeholder}'s long-term objectives?"
            ]
        }
        
    def generate_probing_questions(self,
                                 explanation: str,
                                 domain: str,
                                 depth: int = 3) -> List[Dict[str, Any]]:
        """Generate  probing questions"""
        questions = []
        
        # Parse explanation
        doc = self.nlp(explanation)
        
        # Extract key components
        components = self._extract_components(doc)
        
        # Generate different types of probes
        questions.extend(self._generate_assumption_probes(components, domain))
        questions.extend(self._generate_evidence_probes(components, domain))
        questions.extend(self._generate_impact_probes(components, domain))
        
        # Add domain-specific probes
        if domain in self.domain_specific_probes:
            questions.extend(self._generate_domain_probes(
                components,
                self.domain_specific_probes[domain]
            ))
        
        # Sort by relevance and complexity
        questions = self._rank_questions(questions, explanation)
        
        return questions[:depth]
    
    def _extract_components(self, doc) -> Dict[str, List[str]]:
        """Extract key components from explanation"""
        components = {
            'assumptions': [],
            'evidence': [],
            'stakeholders': [],
            'impacts': [],
            'factors': []
        }
        
        # Extract assumptions
        for sent in doc.sents:
            if any(trigger in sent.text.lower() for trigger in self.probe_patterns['assumption_testing']['triggers']):
                components['assumptions'].append(sent.text)
                
        # Extract evidence mentions
        for sent in doc.sents:
            if any(trigger in sent.text.lower() for trigger in self.probe_patterns['evidence_probing']['triggers']):
                components['evidence'].append(sent.text)
                
        # Extract entities and categorize
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'NORP']:
                components['stakeholders'].append(ent.text)
            elif ent.label_ in ['EVENT', 'FAC']:
                components['impacts'].append(ent.text)
                
        return components
    
    def _generate_assumption_probes(self,
                                  components: Dict[str, List[str]],
                                  domain: str) -> List[Dict[str, Any]]:
        """Generate questions probing assumptions"""
        probes = []
        
        for assumption in components['assumptions']:
            evidence = self.evidence_base.get_relevant_evidence(domain, assumption)
            
            for evidence_piece in evidence:
                probe = {
                    'question': self.probe_patterns['assumption_testing']['template'].format(
                        assumption=assumption,
                        factor=evidence_piece['description']),
                    'type': 'assumption_probe',
                    'evidence': evidence_piece,
                    'confidence': self._calculate_probe_confidence(assumption, evidence_piece)
                }
                probes.append(probe)
                
        return probes
    
    def _generate_evidence_probes(self,
                                components: Dict[str, List[str]],
                                domain: str) -> List[Dict[str, Any]]:
        """Generate questions probing evidence validity"""
        probes = []
        
        for evidence_mention in components['evidence']:
            # Get contrasting or supporting evidence
            related_evidence = self.evidence_base.get_relevant_evidence(domain, evidence_mention)
            
            for rel_evidence in related_evidence:
                probe = {
                    'question': self.probe_patterns['evidence_probing']['template'].format(
                        source=rel_evidence['source'],
                        aspect=self._extract_key_aspect(evidence_mention)
                    ),
                    'type': 'evidence_probe',
                    'related_evidence': rel_evidence,
                    'confidence': self._calculate_probe_confidence(evidence_mention, rel_evidence)
                }
                probes.append(probe)
                
        return probes
    
    def _generate_impact_probes(self,
                              components: Dict[str, List[str]],
                              domain: str) -> List[Dict[str, Any]]:
        """Generate questions about potential impacts"""
        probes = []
        
        for impact in components['impacts']:
            for timeframe in self.probe_patterns['impact_analysis']['timeframes']:
                for stakeholder in components['stakeholders']:
                    probe = {
                        'question': self.probe_patterns['impact_analysis']['template'].format(
                            timeframe=timeframe,
                            stakeholder=stakeholder,
                            condition=impact
                        ),
                        'type': 'impact_probe',
                        'timeframe': timeframe,
                        'confidence': self._calculate_temporal_confidence(timeframe)
                    }
                    probes.append(probe)
                    
        return probes
    
    def _generate_domain_probes(self,
                              components: Dict[str, List[str]],
                              templates: List[str]) -> List[Dict[str, Any]]:
        """Generate domain-specific probing questions"""
        probes = []
        
        for template in templates:
            # Fill template with relevant components
            filled_template = template
            
            for component_type, items in components.items():
                if items:  # If we have components of this type
                    placeholder = f"{{{component_type}}}"
                    if placeholder in template:
                        filled_template = filled_template.replace(placeholder, random.choice(items))
            
            if '{' not in filled_template:  # Only add if all placeholders were filled
                probes.append({
                    'question': filled_template,
                    'type': 'domain_specific_probe',
                    'confidence': 0.85  # Domain-specific probes have high confidence
                })
                
        return probes
    
    def _calculate_probe_confidence(self,
                                  context: str,
                                  evidence: Dict[str, Any]) -> float:
        """Calculate confidence score for a probe"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on evidence relevance
        if evidence.get('relevance', 0) > 0.8:
            confidence += 0.2
        elif evidence.get('relevance', 0) > 0.5:
            confidence += 0.1
            
        # Adjust based on source category
        if evidence.get('category') == 'research_papers':
            confidence += 0.1
        elif evidence.get('category') == 'case_studies':
            confidence += 0.05
            
        return min(confidence, 1.0)
    
    def _calculate_temporal_confidence(self, timeframe: str) -> float:
        """Calculate confidence based on temporal distance"""
        confidence_map = {
            'immediate': 0.9,
            'short-term': 0.8,
            'long-term': 0.6
        }
        return confidence_map.get(timeframe, 0.7)
    
    def _extract_key_aspect(self, text: str) -> str:
        """Extract key aspect from text using NLP"""
        doc = self.nlp(text)
        
        # Try to find a noun phrase
        for chunk in doc.noun_chunks:
            return chunk.text
            
        # Fallback to first noun
        for token in doc:
            if token.pos_ == "NOUN":
                return token.text
                
        return text
    
    def _rank_questions(self,
                       questions: List[Dict[str, Any]],
                       context: str) -> List[Dict[str, Any]]:
        """Rank probing questions by relevance and complexity"""
        # Calculate relevance scores
        for question in questions:
            relevance = self._calculate_relevance(question['question'], context)
            complexity = self._calculate_complexity(question['question'])
            
            # Combined score favoring relevant but not too complex questions
            question['rank_score'] = (relevance * 0.7) + ((1 - complexity) * 0.3)
            
        # Sort by rank score
        return sorted(questions, key=lambda x: x['rank_score'], reverse=True)
    
    def _calculate_relevance(self, question: str, context: str) -> float:
        """Calculate relevance of question to context"""
        # Use TextBlob for simple similarity
        question_blob = TextBlob(question)
        context_blob = TextBlob(context)
        
        # Calculate word overlap and sentiment similarity
        word_overlap = len(set(question_blob.words) & set(context_blob.words)) / len(set(question_blob.words))
        sentiment_diff = abs(question_blob.sentiment.polarity - context_blob.sentiment.polarity)
        
        return (word_overlap * 0.7) + ((1 - sentiment_diff) * 0.3)
    
    def _calculate_complexity(self, question: str) -> float:
        """Calculate question complexity"""
        doc = self.nlp(question)
        
        # Factors contributing to complexity
        num_tokens = len(doc)
        num_entities = len(doc.ents)
        depth = len([token for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
        
        # Normalize and combine
        normalized_complexity = (
            (num_tokens / 20) * 0.4 +  # Length factor
            (num_entities / 3) * 0.3 +  # Entity complexity
            (depth / 2) * 0.3           # Syntactic complexity
        )
        
        return min(normalized_complexity, 1.0)

def create_evidence_based_explanation(query: str, 
                                    domain: str,
                                    counterfactual_gen: CounterfactualGenerator,
                                    prober: Prober,
                                    evidence_base: EvidenceBase) -> Dict[str, Any]:
    """Create a complete explanation with evidence, counterfactuals, and probes"""
    
    # Generate base explanation components
    explanation = {
        'query': query,
        'domain': domain,
        'timestamp': datetime.now().isoformat(),
        'components': {
            'input_parsing': None,
            'feature_identification': None,
            'logical_inference': None
        },
        'evidence': [],
        'counterfactuals': [],
        'probing_questions': []
    }
    
    # Get relevant evidence
    evidence = evidence_base.get_relevant_evidence(domain, query)
    explanation['evidence'] = evidence
    
    # Generate counterfactuals
    counterfactuals = counterfactual_gen.generate_counterfactuals(query, domain)
    explanation['counterfactuals'] = counterfactuals
    
    # Generate probing questions
    probing_questions = prober.generate_probing_questions(str(explanation), domain)
    explanation['probing_questions'] = probing_questions
    
    return explanation