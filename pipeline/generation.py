from typing import List, Dict
from models.vector_store import Document

class GenerationPipeline:
    """Handles prompt construction and answer generation"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def construct_prompt(
        self, 
        query: str, 
        context_docs: List[Document],
        prompt_template: str = None
    ) -> str:
        """Create optimized prompt with retrieved context"""
        # Build context from documents
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.content}" 
            for i, doc in enumerate(context_docs)
        ])
        
        # Use custom template or default
        if prompt_template is None:
            prompt_template = """Answer the question based on the context provided below. If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {query}

Answer:"""
        
        prompt = prompt_template.format(context=context, query=query)
        return prompt
    
    def generate_answer(
        self, 
        query: str, 
        context_docs: List[Document],
        max_tokens: int = 500
    ) -> str:
        """Generate answer using LLM"""
        prompt = self.construct_prompt(query, context_docs)
        answer = self.llm.generate(prompt, max_tokens=max_tokens)
        return answer

