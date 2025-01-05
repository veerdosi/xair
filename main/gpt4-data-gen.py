from openai import OpenAI
import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(
    filename='data_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GPT4Generator:
    def __init__(self, api_key: str, batch_size: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def create_domain_prompt(self, domain: str, template: DomainTemplate) -> str:
        """Create sophisticated prompt for GPT-4 based on domain"""
        
        prompts = {
            "ethical_dilemmas": f"""You are an expert in ethical reasoning and decision-making.
    Context: Generate a complex ethical dilemma and a well-reasoned response.
    Key Considerations:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Impact Areas: {', '.join(template.impact_areas)}
    - Evidence Types: {', '.join(template.evidence_types)}

    The query should present a nuanced ethical challenge without an obvious answer.
    The response should:
    - Consider multiple perspectives
    - Weigh competing values
    - Reference ethical frameworks
    - Provide concrete examples
    - Consider long-term implications

    Example Format:
    {{
        "query": "How should we balance privacy rights with public health monitoring during a pandemic?",
        "response": "This complex issue requires carefully weighing individual privacy against collective well-being..."
    }}""",

            "argument_generation": f"""You are an expert in critical thinking and argumentation.
    Context: Generate a complex policy or social issue query and a balanced analytical response.
    Key Elements:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Impact Areas: {', '.join(template.impact_areas)}
    - Evidence Types: {', '.join(template.evidence_types)}

    The query should address a substantive policy or social issue.
    The response should:
    - Present multiple valid arguments
    - Cite specific evidence
    - Consider counterarguments
    - Analyze trade-offs
    - Draw reasoned conclusions

    Example Format:
    {{
        "query": "What are the key arguments for and against implementing a universal basic income?",
        "response": "The debate around universal basic income centers on several key considerations..."
    }}""",

            "mcdm": f"""You are an expert in multi-criteria decision making and analysis.
    Context: Generate a complex decision-making scenario requiring evaluation of multiple criteria.
    Decision Framework:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Evaluation Criteria: {', '.join(template.impact_areas)}
    - Evidence Sources: {', '.join(template.evidence_types)}

    The query should present a decision scenario with multiple competing factors.
    The response should:
    - Break down decision criteria
    - Quantify trade-offs where possible
    - Use decision matrices or frameworks
    - Consider risk factors
    - Provide clear methodology

    Example Format:
    {{
        "query": "Which renewable energy investment strategy best balances environmental impact, cost efficiency, and community benefits?",
        "response": "To evaluate this investment decision, let's analyze each option across our key criteria..."
    }}""",

            "educational_support": f"""You are an expert educational counselor and career advisor.
    Context: Generate a detailed educational guidance query and comprehensive advisory response.
    Key Areas:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Development Areas: {', '.join(template.impact_areas)}
    - Supporting Data: {', '.join(template.evidence_types)}

    The query should address real educational or career development challenges.
    The response should:
    - Analyze market trends
    - Consider skill development paths
    - Include industry insights
    - Address practical constraints
    - Provide actionable steps

    Example Format:
    {{
        "query": "Should I pursue a specialized Masters in AI or a broader Computer Science degree for a career in tech?",
        "response": "Let's analyze this choice based on current industry trends and your career objectives..."
    }}""",

            "policy_analysis": f"""You are an expert policy analyst and researcher.
    Context: Generate a sophisticated policy analysis question and detailed evaluation.
    Analysis Framework:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Policy Impacts: {', '.join(template.impact_areas)}
    - Evidence Base: {', '.join(template.evidence_types)}

    The query should address complex policy challenges requiring detailed analysis.
    The response should:
    - Evaluate policy effectiveness
    - Consider implementation challenges
    - Assess stakeholder impacts
    - Analyze cost-benefit ratio
    - Project long-term outcomes

    Example Format:
    {{
        "query": "What would be the socioeconomic impacts of implementing a carbon pricing policy in developing economies?",
        "response": "The implementation of carbon pricing in developing economies requires careful consideration of multiple factors..."
    }}""",

            "career_advice": f"""You are an expert career counselor and industry analyst.
    Context: Generate realistic career guidance queries and detailed professional advice.
    Career Framework:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Career Factors: {', '.join(template.impact_areas)}
    - Industry Data: {', '.join(template.evidence_types)}

    The query should reflect real career challenges and transitions.
    The response should:
    - Analyze market opportunities
    - Evaluate skill requirements
    - Consider growth potential
    - Address practical barriers
    - Provide actionable steps

    Example Format:
    {{
        "query": "How should I transition from a traditional software developer role to a machine learning specialist?",
        "response": "This career transition requires a strategic approach considering current market demands and skill gaps..."
    }}""",

            "product_recommendation": f"""You are an expert product analyst and consumer advisor.
    Context: Generate detailed product recommendation queries and comprehensive analysis.
    Evaluation Framework:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Product Factors: {', '.join(template.impact_areas)}
    - Evidence Sources: {', '.join(template.evidence_types)}

    The query should address real consumer decision-making challenges.
    The response should:
    - Compare key features
    - Analyze price-performance ratio
    - Consider use case alignment
    - Evaluate long-term value
    - Address practical limitations

    Example Format:
    {{
        "query": "Which high-end laptop best suits a professional video editor working with 4K content?",
        "response": "For professional 4K video editing, we need to consider several critical factors..."
    }}"""
        }
        
        return prompts.get(domain, self._create_default_prompt(domain, template))

    def _create_default_prompt(self, domain: str, template: DomainTemplate) -> str:
        """Create a default prompt for domains without specific templates"""
        return f"""You are an expert advisor in {domain}.
    Generate a realistic query and detailed response following these guidelines:

    Domain Knowledge:
    - Stakeholders: {', '.join(template.stakeholders)}
    - Key Areas: {', '.join(template.impact_areas)}
    - Evidence Types: {', '.join(template.evidence_types)}

    Requirements:
    1. Query should be specific and realistic
    2. Response should:
    - Provide detailed analysis
    - Reference specific evidence
    - Consider multiple factors
    - Give concrete recommendations
    - Include relevant examples

    Format as JSON:
    {{
        "query": "detailed user question",
        "response": "comprehensive expert response"
    }}"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_batch(self, domain: str, template: DomainTemplate, batch_size: int) -> List[Tuple[str, str]]:
        """Generate a batch of query-response pairs using GPT-4"""
        
        system_prompt = self.create_domain_prompt(domain, template)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Changed from "gpt-4"
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate {batch_size} diverse query-response pairs."}
                    ],
                    temperature=0.8,
                    max_tokens=2000,
                    n=batch_size
                )
            )
            
            pairs = []
            for choice in response.choices:
                try:
                    content = choice.message.content
                    data = json.loads(content)
                    if self.validate_content(data["query"], data["response"], domain, template):
                        pairs.append((data["query"], data["response"]))
                except json.JSONDecodeError:
                    continue
                    
            return pairs
            
        except Exception as e:
            print(f"Error in batch generation: {e}")
            return []

    def validate_content(self, query: str, response: str, domain: str, template: DomainTemplate) -> bool:
        """Enhanced validation of generated content"""
        
        # Basic length checks
        if len(query.split()) < 5 or len(response.split()) < 30:
            return False
            
        # Query structure validation
        if not any(char in query for char in "?"):
            return False
            
        # Domain-specific validation
        validation_criteria = {
            "ethical_dilemmas": {
                "required_phrases": ["ethical", "moral", "value", "impact"],
                "min_response_length": 100,
                "required_components": ["perspective", "consideration", "implication"]
            },
            "argument_generation": {
                "required_phrases": ["argument", "evidence", "analysis"],
                "min_response_length": 150,
                "required_components": ["pros", "cons", "conclusion"]
            },
            "mcdm": {
                "required_phrases": ["criteria", "analysis", "trade-off", "decision"],
                "min_response_length": 150,
                "required_components": ["evaluation", "comparison", "recommendation"]
            },
            "educational_support": {
                "required_phrases": ["career", "skills", "opportunities", "development"],
                "min_response_length": 120,
                "required_components": ["analysis", "pathway", "requirements"]
            },
            "policy_analysis": {
                "required_phrases": ["policy", "impact", "implementation", "stakeholders"],
                "min_response_length": 200,
                "required_components": ["assessment", "implications", "recommendations"]
            },
            "career_advice": {
                "required_phrases": ["market", "skills", "growth", "industry"],
                "min_response_length": 120,
                "required_components": ["analysis", "requirements", "steps"]
            },
            "product_recommendation": {
                "required_phrases": ["features", "performance", "value", "comparison"],
                "min_response_length": 100,
                "required_components": ["features", "comparison", "recommendation"]
            }
        }
        
        criteria = validation_criteria.get(domain, {
            "required_phrases": [],
            "min_response_length": 50,
            "required_components": []
        })
        
        # Check for domain-specific phrases
        if not any(phrase in response.lower() for phrase in criteria["required_phrases"]):
            return False
            
        # Check for evidence citations
        if not any(evidence_type.lower() in response.lower() for evidence_type in template.evidence_types):
            return False
            
        # Check for stakeholder consideration
        if not any(stakeholder.lower() in response.lower() for stakeholder in template.stakeholders):
            return False
            
        # Check for proper structure
        if not self.validate_response_structure(response):
            return False
            
        return True

    def validate_response_structure(self, response: str) -> bool:
        """Validate the structure of the response"""
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) < 3:
            return False
            
        # Check for proper paragraph structure
        paragraphs = response.split('\n\n')
        if len(paragraphs) < 1:
            return False
            
        # Check for coherent flow (basic check)
        transition_words = ['however', 'therefore', 'moreover', 'furthermore']
        has_transitions = any(word in response.lower() for word in transition_words)
        if not has_transitions:
            return False
            
        return True

class ComprehensiveExplanationGenerator:
    def __init__(self, openai_api_key: str):
        self.confidence_scorer = ConfidenceScorer()
        self.gpt4_generator = GPT4Generator(openai_api_key)
        self.setup_domain_data()
        
    async def generate_dataset(self, num_examples: int, output_file: str):
        """Generate dataset with batch processing"""
        dataset = []
        domains = list(self.domains.keys())
        examples_per_domain = num_examples // len(domains)
        batch_size = self.gpt4_generator.batch_size

        # Add to generate_dataset method
        total_generated = 0
        failed_attempts = 0
    
        
        for domain in tqdm(domains, desc="Generating domain examples"):
            logging.info(f"Starting generation for domain: {domain}")
            template = self.domains[domain]
            examples_generated = 0
            
            while examples_generated < examples_per_domain:
                # Generate batch
                batch_pairs = await self.gpt4_generator.generate_batch(
                    domain, template, min(batch_size, examples_per_domain - examples_generated)
                )
                
                for query, response in batch_pairs:
                    example = {
                        "domain": domain,
                        "query": query,
                        "response": response,
                        "explanation": self.generate_explanation(domain, query, response),
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "domain_specific_factors": random.sample(template.impact_areas, 3),
                            "evidence_types_used": random.sample(template.evidence_types, 2),
                            "generation_method": "gpt-4",
                            "validation_passed": True
                        }
                    }
                    
                    dataset.append(example)
                    examples_generated += 1
                
                # Add delay between batches
                await asyncio.sleep(2)
            logging.info(f"Completed {examples_generated}/{examples_per_domain} for {domain}")
        
        # Save dataset
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated {len(dataset)} examples and saved to {output_file}")
        self._print_dataset_statistics(dataset)

if __name__ == "__main__":
    import os
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Generate dataset using asyncio
    generator = ComprehensiveExplanationGenerator(openai_api_key)
    asyncio.run(generator.generate_dataset(15000, "data.json"))
