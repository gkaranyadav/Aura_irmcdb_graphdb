# app.py - IrMC AURA with Neo4j GraphDB + Groq (Fixed Neo4j Connection)
import streamlit as st
import tempfile
import os
import re
import io
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
from neo4j import GraphDatabase
import groq
import speech_recognition as sr
from gtts import gTTS

# --------------------------------------------------------------------------------------
# CONFIG
class Config:
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50

class Neo4jConfig:
    # FIX: Use bolt:// or neo4j+s:// instead of https://
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")

class GroqConfig:
    API_KEY = st.secrets.get("GROQ_API_KEY", "")
    MODEL = "llama3-70b-8192"

# --------------------------------------------------------------------------------------
# NEO4J SERVICE
class Neo4jService:
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        try:
            # Fix URI format if needed
            uri = Neo4jConfig.URI
            if uri.startswith('https://'):
                uri = uri.replace('https://', 'bolt://')
            elif uri.startswith('http://'):
                uri = uri.replace('http://', 'bolt://')
            
            self.driver = GraphDatabase.driver(
                uri,
                auth=(Neo4jConfig.USERNAME, Neo4jConfig.PASSWORD)
            )
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            st.sidebar.success("✅ Connected to Neo4j")
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j: {str(e)}")
            self.driver = None
    
    def create_document_graph(self, document_name, pages_data):
        if not self.driver:
            return False
            
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)
                
                # Create Document node
                session.run("CREATE (d:Document {name: $name})", name=document_name)
                
                # Create Pages and Paragraphs
                for page_num, page_data in pages_data.items():
                    session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (p:Page {number: $page_num})
                    CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num)
                    
                    # Create Paragraph nodes
                    for para_idx, paragraph in enumerate(page_data['paragraphs']):
                        # Extract simple entities using regex
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
                            MERGE (e:Entity {name: $name, type: $type})
                            WITH e
                            MATCH (para:Paragraph {chunk_id: $chunk_id})
                            MERGE (para)-[:MENTIONS]->(e)
                            """, name=entity['name'], type=entity['type'], chunk_id=f"page_{page_num}_para_{para_idx}")
                
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
            'ORG': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Company|Ltd)\b',
            'TECH': r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Neural Network|Deep Learning)\b',
            'CONCEPT': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3:
                    entities.append({"name": match, "type": entity_type})
        
        return entities[:8]
    
    def search_relationships(self, query_terms, document_name):
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
                
                # Find relationships between concepts
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
    
    def get_graph_statistics(self, document_name):
        if not self.driver:
            return {}
            
        try:
            with self.driver.session() as session:
                stats = session.run("""
                MATCH (d:Document {name: $doc_name})
                OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:HAS_PARAGRAPH]->(para:Paragraph)
                OPTIONAL MATCH (para)-[:MENTIONS]->(e:Entity)
                RETURN 
                    count(p) AS page_count,
                    count(para) AS paragraph_count,
                    count(DISTINCT e) AS entity_count
                """, doc_name=document_name).single()
                
                return {
                    "pages": stats["page_count"] or 0,
                    "paragraphs": stats["paragraph_count"] or 0,
                    "entities": stats["entity_count"] or 0
                }
        except Exception as e:
            return {"pages": 0, "paragraphs": 0, "entities": 0}

# --------------------------------------------------------------------------------------
# GROQ LLM SERVICE
class GroqService:
    def __init__(self):
        self.client = None
        self.neo4j = Neo4jService()
        if GroqConfig.API_KEY:
            try:
                self.client = groq.Groq(api_key=GroqConfig.API_KEY)
                st.sidebar.success("✅ Groq Connected")
            except Exception as e:
                st.sidebar.error(f"❌ Groq: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Groq API key not set")

    def generate_answer(self, question, document_name):
        if not document_name:
            return self._no_info_response(question), 0.0
            
        query_terms = self._extract_query_terms(question)
        relationship_data = self.neo4j.search_relationships(query_terms, document_name)
        
        if not relationship_data:
            return self._no_info_response(question), 0.0
        
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
        system_prompt = """You are a professional document analyst. Use the relationship data from the graph database to provide insightful answers. Explain how concepts are connected and reference specific pages."""
        
        context_text = "\n\n".join([
            f"Page {item['page']} ({item['type']}): {item['content'][:300]}..."
            for item in relationship_data
        ])
        
        user_prompt = f"QUESTION: {question}\n\nRELATIONSHIP DATA:\n{context_text}\n\nProvide a detailed answer explaining relationships:"
        
        try:
            response = self.client.chat.completions.create(
                model=GroqConfig.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            return answer + self._format_graph_sources(relationship_data), confidence
        except Exception as e:
            st.error(f"Groq error: {str(e)}")
            return self._fallback_analysis(relationship_data, confidence)

    def _fallback_analysis(self, relationship_data, confidence):
        answer = "**Based on document relationships:**\n\n"
        
        for i, item in enumerate(relationship_data[:3], 1):
            answer += f"{i}. **Page {item['page']}** ({item['type']}): {item['content'][:200]}...\n\n"
        
        return answer + self._format_graph_sources(relationship_data), confidence

    def _format_graph_sources(self, relationship_data):
        if not relationship_data:
            return ""
            
        sources = "\n\n**🔗 Sources:**\n"
        for i, item in enumerate(relationship_data[:3], 1):
            sources += f"{i}. Page {item['page']} | {item['type']}\n"
        return sources

    def _no_info_response(self, question):
        return f"No relationship information found for '{question}'. Try asking about specific concepts.", 0.0

    def analyze_document_overview(self, document_name):
        stats = self.neo4j.get_graph_statistics(document_name)
        
        overview = f"""
**📊 Graph Overview - {document_name}**

- **Pages**: {stats.get('pages', 0)}
- **Paragraphs**: {stats.get('paragraphs', 0)}  
- **Entities**: {stats.get('entities', 0)}

**💡 Ask questions about relationships between concepts!**
        """
        return overview

# --------------------------------------------------------------------------------------
# VOICE SERVICE
class VoiceService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def text_to_speech(self, text):
        """Convert text to speech using gTTS"""
        try:
            # Limit text length for audio
            text = text[:1000] + "..." if len(text) > 1000 else text
            
            tts = gTTS(text=text, lang='en', slow=False)
            audio_file = io.BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            return audio_file
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")
            return None
    
    def speech_to_text(self):
        """Convert speech to text using microphone"""
        try:
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.error("⏰ Listening timeout. Please try again.")
            return None
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio. Please try again.")
            return None
        except Exception as e:
            st.error(f"🎤 Speech recognition error: {str(e)}")
            return None

# --------------------------------------------------------------------------------------
# DOCUMENT PROCESSOR
class DocumentProcessor:
    def __init__(self):
        self.neo4j = Neo4jService()

    def process_pdf(self, uploaded_file):
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
                return len(pages_data)
            else:
                return 0
                
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_pdf_text(self, pdf_path):
        # Determine PDF type
        pdf_type = self._analyze_pdf_type(pdf_path)
        
        if pdf_type == "text_based":
            return self._extract_text_direct(pdf_path)
        else:
            return self._extract_text_ocr(pdf_path)

    def _analyze_pdf_type(self, pdf_path):
        reader = PdfReader(pdf_path)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    def _extract_text_direct(self, pdf_path):
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

# --------------------------------------------------------------------------------------
# STREAMLIT APP
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_service = GroqService()
        self.voice_service = VoiceService()
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
        if 'auto_voice' not in st.session_state:
            st.session_state.auto_voice = True

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA (GraphDB + Groq)")
        
        # Connection Status
        if self.doc_processor.neo4j.driver:
            st.sidebar.success("✅ Neo4j Connected")
        else:
            st.sidebar.error("❌ Neo4j Disconnected")
            
        if self.llm_service.client:
            st.sidebar.success("✅ Groq Connected")
        else:
            st.sidebar.warning("⚠️ Groq Disconnected")
        
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and not st.session_state.pdf_processed:
            if st.sidebar.button("Process Document to Graph"):
                with st.spinner("Processing PDF and building graph..."):
                    page_count = self.doc_processor.process_pdf(uploaded_file)
                    if page_count > 0:
                        st.session_state.pdf_processed = True
                        st.session_state.current_document = uploaded_file.name
                        st.rerun()
                    
        # App settings
        st.sidebar.markdown("### Settings")
        st.session_state.input_mode = st.sidebar.radio("Input mode", ["Text", "Voice"], index=0)
        st.session_state.auto_voice = st.sidebar.checkbox("Auto Voice Response", True)
        
        # Document overview button
        if st.session_state.pdf_processed:
            if st.sidebar.button("Document Graph Overview"):
                overview = self.llm_service.analyze_document_overview(st.session_state.current_document)
                st.session_state.messages.append({"role": "assistant", "content": overview})
                st.rerun()
        
        # Voice input button
        if st.session_state.input_mode == "Voice":
            if st.sidebar.button("🎤 Start Voice Input"):
                question = self.voice_service.speech_to_text()
                if question:
                    self.process_question(question)
        
        return 3

    def render_chat(self, top_k):
        st.title("Aura PDF QA ⚡ (GraphDB + Groq)")
        
        if not st.session_state.pdf_processed:
            st.info("📁 Upload a PDF from the sidebar to build a knowledge graph and start asking relationship-based questions!")
            return
            
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Get user input based on mode
        if st.session_state.input_mode == "Text":
            question = st.chat_input("Ask about relationships in your document...")
            if question:
                self.process_question(question)

    def process_question(self, question):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("Exploring graph relationships with Groq..."):
                answer, confidence = self.llm_service.generate_answer(question, st.session_state.current_document)
                st.markdown(answer)
                
                # Voice response if enabled
                if st.session_state.auto_voice:
                    audio_file = self.voice_service.text_to_speech(answer)
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                    
        st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        top_k = self.render_sidebar()
        self.render_chat(top_k)

# --------------------------------------------------------------------------------------
# MAIN
if __name__ == "__main__":
    app = AuraPDFQAApp()
    app.run()
