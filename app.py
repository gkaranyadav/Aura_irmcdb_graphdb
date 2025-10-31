# app.py - IrMC AURA with Neo4j GraphDB (Simplified)

import streamlit as st
import tempfile
import os
import re
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
from streamlit.components.v1 import html
from neo4j import GraphDatabase

# --------------------------------------------------------------------------------------
# CONFIG

class Config:
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")

# --------------------------------------------------------------------------------------
# NEO4J SERVICE

class Neo4jService:
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                Neo4jConfig.URI,
                auth=(Neo4jConfig.USERNAME, Neo4jConfig.PASSWORD)
            )
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            st.sidebar.success("✅ Connected to Neo4j")
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j connection failed: {str(e)}")
            self.driver = None
    
    def create_document_graph(self, document_name, pages_data):
        if not self.driver:
            return False
            
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (d:Document {name: $doc_name}) DETACH DELETE d", doc_name=document_name)
                
                # Create Document
                session.run("CREATE (d:Document {name: $doc_name, upload_date: datetime()})", doc_name=document_name)
                
                # Create Pages and Paragraphs
                for page_num, page_data in pages_data.items():
                    session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (p:Page {number: $page_num})
                    CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num)
                    
                    for para_idx, paragraph in enumerate(page_data['paragraphs']):
                        # Extract simple entities
                        entities = self._extract_simple_entities(paragraph)
                        
                        session.run("""
                        MATCH (p:Page {number: $page_num})<-[:HAS_PAGE]-(d:Document {name: $doc_name})
                        CREATE (para:Paragraph {content: $content, chunk_id: $chunk_id})
                        CREATE (p)-[:HAS_PARAGRAPH]->(para)
                        """, doc_name=document_name, page_num=page_num, 
                           content=paragraph, chunk_id=f"page_{page_num}_para_{para_idx}")
                        
                        # Create entity relationships
                        for entity in entities:
                            session.run("""
                            MERGE (e:Entity {name: $name})
                            WITH e
                            MATCH (para:Paragraph {chunk_id: $chunk_id})
                            MERGE (para)-[:MENTIONS]->(e)
                            """, name=entity, chunk_id=f"page_{page_num}_para_{para_idx}")
                
                st.sidebar.success(f"✅ Graph created for {document_name}")
                return True
                
        except Exception as e:
            st.error(f"❌ Graph creation failed: {str(e)}")
            return False
    
    def _extract_simple_entities(self, text):
        """Simple entity extraction using regex patterns"""
        entities = []
        
        # Patterns for common entities
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Company|Ltd|Systems|Technologies)\b',
            'TECH': r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Neural Network|Deep Learning|Transformer|GPT|LLM)\b',
            'ACRONYM': r'\b[A-Z]{2,}\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(set(entities))[:10]  # Return top 10 unique entities
    
    def search_relationships(self, query_terms, document_name):
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                results = []
                
                for term in query_terms:
                    # Find paragraphs mentioning the term
                    paragraphs_result = session.run("""
                    MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
                    WHERE para.content CONTAINS $term
                    RETURN para.content AS content, p.number AS page
                    LIMIT 3
                    """, doc_name=document_name, term=term)
                    
                    for record in paragraphs_result:
                        results.append({
                            "content": record["content"],
                            "page": record["page"],
                            "concept": term,
                            "type": "DIRECT_MENTION"
                        })
                
                return results
                
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            return []

# ------------------------------------------------------------------------------------------------------
# LLM Service (Simplified)

class LLMService:
    def __init__(self):
        self.client = None
        self.neo4j = Neo4jService()
        try:
            import groq
            api_key = st.secrets.get("GROQ_API_KEY", "")
            if api_key:
                self.client = groq.Groq(api_key=api_key)
        except:
            pass

    def generate_answer(self, question, document_name):
        if not document_name:
            return "Please upload and process a PDF first.", 0.0
            
        query_terms = self._extract_query_terms(question)
        relationship_data = self.neo4j.search_relationships(query_terms, document_name)
        
        if not relationship_data:
            return f"No information found about '{question}'. Try different terms.", 0.0
        
        confidence = min(len(relationship_data) / 5.0, 1.0)
        
        if self.client:
            return self._groq_analysis(question, relationship_data, confidence)
        else:
            return self._fallback_analysis(relationship_data, confidence)

    def _extract_query_terms(self, question):
        words = question.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:4]

    def _groq_analysis(self, question, relationship_data, confidence):
        context_text = "\n\n".join([
            f"Page {item['page']}: {item['content'][:300]}..."
            for item in relationship_data
        ])
        
        user_prompt = f"QUESTION: {question}\n\nDOCUMENT EXCERPTS:\n{context_text}\n\nProvide a concise answer:"
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=800
            )
            answer = response.choices[0].message.content
            return answer + self._format_sources(relationship_data), confidence
        except:
            return self._fallback_analysis(relationship_data, confidence)

    def _fallback_analysis(self, relationship_data, confidence):
        answer = "**Based on the document:**\n\n"
        for i, item in enumerate(relationship_data[:3], 1):
            answer += f"{i}. Page {item['page']}: {item['content'][:200]}...\n"
        return answer + self._format_sources(relationship_data), confidence

    def _format_sources(self, relationship_data):
        sources = "\n\n**📚 Sources:**\n"
        for i, item in enumerate(relationship_data[:3], 1):
            sources += f"{i}. Page {item['page']}\n"
        return sources

# ---------------------------------------------------------------------------------------------------------------------
# DOCUMENT PROCESSOR

class DocumentProcessor:
    def __init__(self):
        self.neo4j = Neo4jService()

    def process_pdf(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            pages_data = self._extract_pdf_text(pdf_path)
            if not pages_data:
                st.error("❌ No text extracted from PDF.")
                return 0
                
            success = self.neo4j.create_document_graph(uploaded_file.name, pages_data)
            return len(pages_data) if success else 0
                
        finally:
            if os.path.exists(pdf_path):
                os.unlink(path)

    def _extract_pdf_text(self, pdf_path):
        reader = PdfReader(pdf_path)
        pages_data = {}
        
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                paragraphs = self._split_into_paragraphs(text)
                pages_data[i] = {'paragraphs': paragraphs}
        
        return pages_data

    def _split_into_paragraphs(self, text):
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        final_paragraphs = []
        
        for para in paragraphs:
            if len(para) > Config.CHUNK_SIZE:
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < Config.CHUNK_SIZE:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            final_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk.strip():
                    final_paragraphs.append(current_chunk.strip())
            else:
                final_paragraphs.append(para)
                
        return final_paragraphs

# ---------------------------------------------------------------------------------------------------------------------
# STREAMLIT APP

class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_service = LLMService()
        self.setup_session_state()
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide")

    def setup_session_state(self):
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA (GraphDB)")
        
        if self.doc_processor.neo4j.driver:
            st.sidebar.success("✅ Neo4j Connected")
        else:
            st.sidebar.error("❌ Neo4j Disconnected")
        
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and not st.session_state.pdf_processed:
            if st.sidebar.button("Process Document to Graph"):
                with st.spinner("Processing PDF and building graph..."):
                    page_count = self.doc_processor.process_pdf(uploaded_file)
                    if page_count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.current_document = uploaded_file.name
                        st.rerun()
        
        return 3

    def render_chat(self, top_k):
        st.title("Aura PDF QA ⚡ (GraphDB)")
        
        if not st.session_state.pdf_processed:
            st.info("📁 Upload a PDF from the sidebar to build a knowledge graph!")
            return
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        question = st.chat_input("Ask about relationships in your document...")
            
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.chat_message("assistant"):
                with st.spinner("Exploring graph relationships..."):
                    answer, confidence = self.llm_service.generate_answer(question, st.session_state.current_document)
                    st.markdown(answer)
                    
            st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
