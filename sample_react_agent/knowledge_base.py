import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from models import RAGResult

class KnowledgeBase:
    def __init__(self, collection_name: str = "company_docs"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
            self._populate_knowledge_base()
    
    def _populate_knowledge_base(self):
        """Populate the knowledge base with sample documents"""
        
        documents = [
            {
                "id": "company_policy_1",
                "content": "Our company return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days after we receive the returned item. Customers are responsible for return shipping costs unless the item was defective.",
                "metadata": {"category": "policy", "topic": "returns"}
            },
            {
                "id": "company_policy_2", 
                "content": "We offer free shipping on orders over $50 within the continental United States. Standard shipping takes 3-5 business days, while expedited shipping takes 1-2 business days. International shipping is available for an additional fee and typically takes 7-14 business days.",
                "metadata": {"category": "policy", "topic": "shipping"}
            },
            {
                "id": "product_info_1",
                "content": "Our smartphones come with a 2-year manufacturer warranty covering defects in materials and workmanship. The warranty does not cover damage from accidents, misuse, or normal wear and tear. Water damage is not covered unless specifically stated as water-resistant.",
                "metadata": {"category": "product", "topic": "warranty"}
            },
            {
                "id": "product_info_2",
                "content": "All laptops in our inventory feature SSD storage for faster boot times and application loading. Most models include 8GB or 16GB of RAM and are suitable for both work and entertainment. Gaming laptops feature dedicated graphics cards for enhanced performance.",
                "metadata": {"category": "product", "topic": "specifications"}
            },
            {
                "id": "customer_service_1",
                "content": "Our customer service team is available Monday through Friday from 9 AM to 6 PM EST. You can reach us by phone at 1-800-SUPPORT, email at support@company.com, or through live chat on our website. We typically respond to emails within 24 hours.",
                "metadata": {"category": "service", "topic": "contact"}
            },
            {
                "id": "customer_service_2",
                "content": "If you're experiencing issues with your order, please have your order number ready when contacting support. Common issues include tracking problems, damaged items, or incorrect orders. We're committed to resolving all issues quickly and to your satisfaction.",
                "metadata": {"category": "service", "topic": "support"}
            },
            {
                "id": "account_info_1",
                "content": "Creating an account allows you to track orders, save favorite items, and receive exclusive offers. Your personal information is protected with industry-standard encryption. You can update your account information, including shipping addresses and payment methods, at any time.",
                "metadata": {"category": "account", "topic": "benefits"}
            },
            {
                "id": "payment_info_1",
                "content": "We accept all major credit cards, PayPal, and Apple Pay. All payments are processed securely through encrypted connections. For large orders over $1000, we also accept bank transfers and purchase orders from verified business customers.",
                "metadata": {"category": "payment", "topic": "methods"}
            },
            {
                "id": "promotion_info_1",
                "content": "We regularly offer seasonal promotions and discounts. Sign up for our newsletter to receive exclusive coupon codes and early access to sales. Students and military personnel are eligible for a 10% discount with valid ID verification.",
                "metadata": {"category": "promotion", "topic": "discounts"}
            },
            {
                "id": "technical_support_1",
                "content": "For technical issues with electronic products, our technical support team can help with setup, troubleshooting, and software questions. We provide phone support, email assistance, and downloadable user manuals. Most technical issues can be resolved within one support session.",
                "metadata": {"category": "technical", "topic": "support"}
            }
        ]
        
        # Add documents to the collection
        for doc in documents:
            self.collection.add(
                documents=[doc["content"]],
                metadatas=[doc["metadata"]],
                ids=[doc["id"]]
            )
        
        print(f"Knowledge base populated with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> RAGResult:
        """Search the knowledge base and return relevant documents"""
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Convert distances to similarity scores (lower distance = higher similarity)
            scores = [1.0 - dist for dist in distances] if distances else []
            
            # Generate an answer based on the retrieved documents
            if documents:
                answer = self._generate_answer(query, documents[:3])  # Use top 3 documents
            else:
                answer = "I couldn't find relevant information in the knowledge base."
            
            return RAGResult(
                documents=documents,
                scores=scores,
                answer=answer
            )
            
        except Exception as e:
            return RAGResult(
                documents=[],
                scores=[],
                answer=f"Error searching knowledge base: {str(e)}"
            )
    
    def _generate_answer(self, query: str, documents: List[str]) -> str:
        """Generate an answer based on retrieved documents"""
        
        # Simple answer generation - in a real system, you'd use an LLM here
        relevant_info = []
        
        query_lower = query.lower()
        
        for doc in documents:
            # Find sentences that contain query terms
            sentences = doc.split('. ')
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_lower.split()):
                    relevant_info.append(sentence.strip())
        
        if relevant_info:
            # Remove duplicates and join
            unique_info = list(dict.fromkeys(relevant_info))
            answer = '. '.join(unique_info[:3])  # Limit to 3 most relevant sentences
            if not answer.endswith('.'):
                answer += '.'
            return answer
        else:
            return "Based on the available information: " + documents[0][:200] + "..."

def create_knowledge_base():
    """Initialize and populate the knowledge base"""
    kb = KnowledgeBase()
    return kb

if __name__ == "__main__":
    kb = create_knowledge_base()
    
    # Test search
    result = kb.search("What is the return policy?")
    print("Search Result:")
    print(f"Answer: {result.answer}")
    print(f"Documents found: {len(result.documents)}")