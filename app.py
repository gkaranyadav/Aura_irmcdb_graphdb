# app.py - IrMC AURA with Neo4j GraphDB

# Importing necessary libraries for the app
import streamlit as st  # For creating web app interface
import tempfile  # For creating temporary files
import os  # For operating system operations
import time  # For time-related functions
from PyPDF2 import PdfReader  # For reading PDF files
from sentence_transformers import SentenceTransformer  # For text embeddings
import faiss  # For similarity search
import numpy as np  # For numerical operations
import pytesseract  # For OCR text extraction
from PIL import Image  # For image processing
import pdf2image  # For converting PDF to images
from streamlit.components.v1 import html  # For custom HTML components
from neo4j import GraphDatabase  # Neo4j graph database
import networkx as nx  # For graph operations
import re  # For regular expressions
import spacy  # For NLP entity extraction

# --------------------------------------------------------------------------------------
# CONFIG - App configuration settings

class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Model for text embeddings
    TOP_K_CHUNKS = 5  # Number of top chunks to retrieve
    CHUNK_SIZE = 1000  # Size of text chunks
    MIN_PARAGRAPH_LENGTH = 50  # Minimum length for paragraphs

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")
    DATABASE = "neo4j"

# --------------------------------------------------------------------------------------
# NEO4J SERVICE - Graph Database Operations

class Neo4jService:
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                Neo4jConfig.URI,
                auth=(Neo4jConfig.USERNAME, Neo4jConfig.PASSWORD)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            st.sidebar.success("✅ Connected to Neo4j")
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j connection failed: {str(e)}")
            self.driver = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def create_document_graph(self, document_name, pages_data):
        """Create graph structure from PDF document"""
        if not self.driver:
            return False
            
        try:
            with self.driver.session() as session:
                # Clear existing data for this document
                session.run("""
                MATCH (d:Document {name: $doc_name}) 
                DETACH DELETE d
                """, doc_name=document_name)
                
                # Create Document node
                session.run("""
                CREATE (d:Document {name: $doc_name, upload_date: datetime()})
                RETURN d
                """, doc_name=document_name)
                
                # Create Pages and Paragraphs
                for page_num, page_data in pages_data.items():
                    # Create Page node
                    session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (p:Page {number: $page_num, preview: $preview})
                    CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num, 
                       preview=page_data['preview'][:200] + "...")
                    
                    # Create Paragraph nodes
                    for para_idx, paragraph in enumerate(page_data['paragraphs']):
                        session.run("""
                        MATCH (p:Page {number: $page_num})<-[:HAS_PAGE]-(d:Document {name: $doc_name})
                        CREATE (para:Paragraph {content: $content, chunk_id: $chunk_id})
                        CREATE (p)-[:HAS_PARAGRAPH]->(para)
                        """, doc_name=document_name, page_num=page_num, 
                           content=paragraph, chunk_id=f"page_{page_num}_para_{para_idx}")
                
                st.sidebar.success(f"✅ Graph created for {document_name}")
                return True
                
        except Exception as e:
            st.error(f"❌ Graph creation failed: {str(e)}")
            return False
    
    def extract_and_link_entities(self, document_name, pages_data):
        """Extract entities and create relationships"""
        if not self.driver:
            return False
            
        try:
            # Load spaCy model for entity extraction
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.error("❌ spaCy model not found. Please install: python -m spacy download en_core_web_sm")
                return False
            
            with self.driver.session() as session:
                for page_num, page_data in pages_data.items():
                    for para_idx, paragraph in enumerate(page_data['paragraphs']):
                        # Extract entities using spaCy
                        doc = nlp(paragraph)
                        
                        for ent in doc.ents:
                            entity_type = self._map_entity_type(ent.label_)
                            if entity_type in ["PERSON", "ORG", "GPE", "TECH"]:
                                # Create Entity node
                                session.run("""
                                MERGE (e:Entity {name: $name, type: $type})
                                """, name=ent.text, type=entity_type)
                                
                                # Link to Paragraph
                                session.run("""
                                MATCH (para:Paragraph {chunk_id: $chunk_id})
                                MATCH (e:Entity {name: $name, type: $type})
                                MERGE (para)-[:MENTIONS_ENTITY]->(e)
                                """, chunk_id=f"page_{page_num}_para_{para_idx}", 
                                   name=ent.text, type=entity_type)
                        
                        # Extract concepts (simple keyword-based for now)
                        concepts = self._extract_concepts(paragraph)
                        for concept in concepts:
                            session.run("""
                            MERGE (c:Concept {name: $name})
                            """, name=concept)
                            
                            session.run("""
                            MATCH (para:Paragraph {chunk_id: $chunk_id})
                            MATCH (c:Concept {name: $name})
                            MERGE (para)-[:MENTIONS_CONCEPT]->(c)
                            """, chunk_id=f"page_{page_num}_para_{para_idx}", name=concept)
                
                st.sidebar.success("✅ Entities and concepts extracted")
                return True
                
        except Exception as e:
            st.error(f"❌ Entity extraction failed: {str(e)}")
            return False
    
    def _map_entity_type(self, spacy_label):
        """Map spaCy entity types to custom types"""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORG", 
            "GPE": "GPE",
            "PRODUCT": "TECH",
            "WORK_OF_ART": "CONCEPT"
        }
        return mapping.get(spacy_label, "OTHER")
    
    def _extract_concepts(self, text):
        """Extract key concepts from text"""
        concepts = []
        # Simple pattern matching for technical terms
        patterns = [
            r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b',
            r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Neural Network|Deep Learning)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            concepts.extend([match.strip() for match in matches if len(match) > 3])
        
        return list(set(concepts))[:5]  # Return top 5 unique concepts
    
    def search_relationships(self, query_terms, document_name):
        """Search for relationships between concepts"""
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                results = []
                
                for term in query_terms:
                    # Find paragraphs mentioning the concept
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
                
                # Find related concepts
                if len(query_terms) >= 2:
                    for i, term1 in enumerate(query_terms):
                        for term2 in query_terms[i+1:]:
                            relationship_result = session.run("""
                            MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
                            WHERE para.content CONTAINS $term1 AND para.content CONTAINS $term2
                            RETURN para.content AS content, p.number AS page
                            LIMIT 2
                            """, doc_name=document_name, term1=term1, term2=term2)
                            
                            for record in relationship_result:
                                results.append({
                                    "content": record["content"],
                                    "page": record["page"], 
                                    "concept": f"{term1} & {term2}",
                                    "type": "RELATIONSHIP"
                                })
                
                return results
                
        except Exception as e:
            st.error(f"❌ Relationship search failed: {str(e)}")
            return []

# ------------------------------------------------------------------------------------------------------
# LLM Groq

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
        """Generate answer using GraphDB relationships"""
        if not document_name:
            return self._no_info_response(question), 0.0
            
        # Extract key terms from question
        query_terms = self._extract_query_terms(question)
        
        # Search relationships in Neo4j
        relationship_data = self.neo4j.search_relationships(query_terms, document_name)
        
        if not relationship_data:
            return self._no_info_response(question), 0.0
        
        # Calculate confidence based on matches
        confidence = min(len(relationship_data) / 5.0, 1.0)
        
        # Generate answer using LLM
        if self.client:
            return self._groq_analysis(question, relationship_data, confidence)
        else:
            return self._fallback_analysis(relationship_data, confidence)

    def _extract_query_terms(self, question):
        """Extract key terms from question for graph search"""
        # Simple extraction - can be enhanced with NLP
        words = question.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:4]  # Limit to 4 terms

    def _groq_analysis(self, question, relationship_data, confidence):
        """Use Groq LLM for analysis with graph data"""
        system_prompt = """You are a professional document analyst. Use the relationship data from the graph database to provide insightful answers. Explain how concepts are connected and reference specific pages."""
        
        context_text = "\n\n".join([
            f"Page {item['page']} ({item['type']}): {item['content'][:300]}..."
            for item in relationship_data
        ])
        
        user_prompt = f"QUESTION: {question}\n\nRELATIONSHIP DATA:\n{context_text}\n\nProvide a detailed answer explaining relationships:"
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            answer = response.choices[0].message.content
            return answer + self._format_graph_sources(relationship_data), confidence
        except:
            return self._fallback_analysis(relationship_data, confidence)

    def _fallback_analysis(self, relationship_data, confidence):
        """Fallback analysis without API"""
        answer = "**Based on document relationships:**\n\n"
        
        # Group by relationship type
        direct_mentions = [item for item in relationship_data if item['type'] == 'DIRECT_MENTION']
        relationships = [item for item in relationship_data if item['type'] == 'RELATIONSHIP']
        
        if direct_mentions:
            answer += "**Key Mentions:**\n"
            for item in direct_mentions[:3]:
                answer += f"• {item['concept']} (Page {item['page']})\n"
        
        if relationships:
            answer += "\n**Relationships Found:**\n"
            for item in relationships[:2]:
                answer += f"• {item['concept']} (Page {item['page']})\n"
        
        return answer + self._format_graph_sources(relationship_data), confidence

    def _format_graph_sources(self, relationship_data):
        """Format sources from graph data"""
        if not relationship_data:
            return ""
            
        sources = "\n\n**🔗 Relationship Sources:**\n"
        for i, item in enumerate(relationship_data[:3], 1):
            sources += f"{i}. Page {item['page']} | {item['type']} | {item['concept']}\n"
        return sources

    def _no_info_response(self, question):
        return f"No relationship information found for '{question}'. Try asking about specific concepts or entities.", 0.0

# ---------------------------------------------------------------------------------------------------------------------
# DOCUMENT PROCESSOR - Handles PDF processing and graph creation

class DocumentProcessor:
    def __init__(self):
        self.neo4j = Neo4jService()

    def process_pdf(self, uploaded_file):
        """Process PDF and create graph in Neo4j"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # Extract text from PDF
            pages_data = self._extract_pdf_text(pdf_path)
            
            if not pages_data:
                st.error("❌ No text extracted from PDF.")
                return 0
                
            # Create graph in Neo4j
            success = self.neo4j.create_document_graph(uploaded_file.name, pages_data)
            
            if success:
                # Extract entities and relationships
                self.neo4j.extract_and_link_entities(uploaded_file.name, pages_data)
                return len(pages_data)
            else:
                return 0
                
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF using appropriate method"""
        # Determine PDF type
        pdf_type = self._analyze_pdf_type(pdf_path)
        
        if pdf_type == "text_based":
            return self._extract_text_direct(pdf_path)
        else:
            return self._extract_text_ocr(pdf_path)

    def _analyze_pdf_type(self, pdf_path):
        """Determine if PDF is text-based or scanned"""
        reader = PdfReader(pdf_path)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    def _extract_text_direct(self, pdf_path):
        """Extract text directly from PDF"""
        reader = PdfReader(pdf_path)
        pages_data = {}
        
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                paragraphs = self._split_into_paragraphs(text)
                pages_data[i] = {
                    'preview': text[:500],
                    'paragraphs': paragraphs
                }
        
        return pages_data

    def _extract_text_ocr(self, pdf_path):
        """Extract text using OCR"""
        images = pdf2image.convert_from_path(pdf_path, dpi=200)
        pages_data = {}
        
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                paragraphs = self._split_into_paragraphs(text)
                pages_data[i+1] = {
                    'preview': text[:500],
                    'paragraphs': paragraphs
                }
        
        return pages_data

    def _split_into_paragraphs(self, text):
        """Split text into paragraphs"""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Further split large paragraphs
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > Config.CHUNK_SIZE:
                # Split by sentences for large paragraphs
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
# VOICE SERVICE (Browser Native)

def speak_text_js(text):
    escaped_text = text.replace('"', '\\"')
    html_code = f"""
    <script>
    const msg = new SpeechSynthesisUtterance("{escaped_text}");
    msg.lang = "en-US";
    msg.rate = 1;
    window.speechSynthesis.speak(msg);
    </script>
    """
    html(html_code)

def get_voice_input():
    js_code = """
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.start();
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.dispatchEvent(new CustomEvent("voice_input", {detail: transcript}));
    };
    </script>
    """
    html(js_code)
    voice_text = st.text_input("Voice Input (Speak now):", key="voice_text")
    return voice_text

# ---------------------------------------------------------------------------------------------------------------------
# STREAMLIT APP - Main application class

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
        if 'input_mode' not in st.session_state:
            st.session_state.input_mode = "Text"
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA (GraphDB)")
        
        # Neo4j Connection Status
        st.sidebar.markdown("### Database Status")
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
                        st.experimental_rerun()
                    
        # App settings
        st.sidebar.markdown("### Settings")
        st.session_state.input_mode = st.sidebar.radio("Input mode", ["Text", "Voice"], index=0)
        auto_voice = st.sidebar.checkbox("Auto Voice Response", True)
        st.session_state.auto_voice = auto_voice
        
        # Document overview button
        if st.session_state.pdf_processed:
            if st.sidebar.button("Document Graph Overview"):
                self.show_graph_overview()
                
        return 3  # Default top_k for compatibility

    def show_graph_overview(self):
        """Show graph statistics and overview"""
        try:
            with self.doc_processor.neo4j.driver.session() as session:
                # Get graph statistics
                stats = session.run("""
                MATCH (d:Document {name: $doc_name})
                OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:HAS_PARAGRAPH]->(para:Paragraph)
                OPTIONAL MATCH (para)-[:MENTIONS_ENTITY]->(e:Entity)
                OPTIONAL MATCH (para)-[:MENTIONS_CONCEPT]->(c:Concept)
                RETURN 
                    count(p) AS page_count,
                    count(para) AS paragraph_count,
                    count(DISTINCT e) AS entity_count,
                    count(DISTINCT c) AS concept_count
                """, doc_name=st.session_state.current_document).single()
                
                overview = f"""
**📊 Graph Overview - {st.session_state.current_document}**

- **Pages**: {stats['page_count']}
- **Paragraphs**: {stats['paragraph_count']}  
- **Entities**: {stats['entity_count']}
- **Concepts**: {stats['concept_count']}

**💡 Ask questions about relationships between concepts and entities!**
                """
                
                st.session_state.messages.append({"role": "assistant", "content": overview})
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Failed to get graph overview: {str(e)}")

    def render_chat(self, top_k):
        st.title("Aura PDF QA ⚡ (GraphDB)")
        
        if not st.session_state.pdf_processed:
            st.info("📁 Upload a PDF from the sidebar to build a knowledge graph and start asking relationship-based questions!")
            return
            
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Get user input
        if st.session_state.input_mode == "Voice":
            question = get_voice_input()
        else:
            question = st.chat_input("Ask about relationships in your document...")
            
        if question:
            self.process_question(question)

    def process_question(self, question):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("Exploring graph relationships..."):
                answer, confidence = self.llm_service.generate_answer(question, st.session_state.current_document)
                st.markdown(answer)
                
                if st.session_state.auto_voice:
                    speak_text_js(answer)
                    
        st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

# ---------------------------------------------------------------------------------------------------------------------
# MAIN - App entry point

if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
