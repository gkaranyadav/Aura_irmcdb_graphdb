# aura_pdf_qa_app.py - Single-file working version

# --------------------------- IMPORTS ---------------------------
import streamlit as st
import tempfile, os, re, time
from PyPDF2 import PdfReader
import pdf2image
import pytesseract
from PIL import Image
from neo4j import GraphDatabase
import spacy
from streamlit.components.v1 import html

# ------------------------ CONFIG ------------------------
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50
    TOP_K_CHUNKS = 5

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")
    DATABASE = "neo4j"

# ------------------------ NEO4J SERVICE ------------------------
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
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            st.sidebar.success("✅ Connected to Neo4j")
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j connection failed: {str(e)}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def create_document_graph(self, document_name, pages_data):
        if not self.driver:
            return False
        try:
            with self.driver.session() as session:
                # Clear existing document nodes
                session.run("MATCH (d:Document {name:$doc_name}) DETACH DELETE d", doc_name=document_name)
                # Create Document node
                session.run("CREATE (d:Document {name:$doc_name, upload_date: datetime()})", doc_name=document_name)
                # Create Pages & Paragraphs
                for page_num, page_data in pages_data.items():
                    session.run("""
                        MATCH (d:Document {name:$doc_name})
                        CREATE (p:Page {number:$page_num, preview:$preview})
                        CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num, preview=page_data['preview'][:200]+"...")
                    for idx, para in enumerate(page_data['paragraphs']):
                        session.run("""
                            MATCH (p:Page {number:$page_num})<-[:HAS_PAGE]-(d:Document {name:$doc_name})
                            CREATE (para:Paragraph {content:$content, chunk_id:$chunk_id})
                            CREATE (p)-[:HAS_PARAGRAPH]->(para)
                        """, doc_name=document_name, page_num=page_num,
                           content=para, chunk_id=f"page_{page_num}_para_{idx}")
            st.sidebar.success(f"✅ Graph created for {document_name}")
            return True
        except Exception as e:
            st.error(f"❌ Graph creation failed: {str(e)}")
            return False

    def extract_entities(self, document_name, pages_data):
        if not self.driver:
            return False
        try:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.error("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
                return False
            with self.driver.session() as session:
                for page_num, page_data in pages_data.items():
                    for idx, para in enumerate(page_data['paragraphs']):
                        doc = nlp(para)
                        for ent in doc.ents:
                            session.run("MERGE (e:Entity {name:$name, type:$type})",
                                        name=ent.text, type=ent.label_)
                            session.run("""
                                MATCH (para:Paragraph {chunk_id:$chunk_id})
                                MATCH (e:Entity {name:$name, type:$type})
                                MERGE (para)-[:MENTIONS_ENTITY]->(e)
                            """, chunk_id=f"page_{page_num}_para_{idx}",
                               name=ent.text, type=ent.label_)
            st.sidebar.success("✅ Entities extracted")
            return True
        except Exception as e:
            st.error(f"❌ Entity extraction failed: {str(e)}")
            return False

    def search_relationships(self, query_terms, document_name):
        if not self.driver:
            return []
        try:
            results = []
            with self.driver.session() as session:
                for term in query_terms:
                    paras = session.run("""
                        MATCH (d:Document {name:$doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
                        WHERE para.content CONTAINS $term
                        RETURN para.content AS content, p.number AS page LIMIT 3
                    """, doc_name=document_name, term=term)
                    for r in paras:
                        results.append({"content": r["content"], "page": r["page"], "concept": term, "type":"DIRECT"})
            return results
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            return []

# ------------------------ DOCUMENT PROCESSOR ------------------------
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
                st.error("❌ No text extracted")
                return 0
            success = self.neo4j.create_document_graph(uploaded_file.name, pages_data)
            if success:
                self.neo4j.extract_entities(uploaded_file.name, pages_data)
                return len(pages_data)
            return 0
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_pdf_text(self, pdf_path):
        reader = PdfReader(pdf_path)
        pages_data = {}
        for i, page in enumerate(reader.pages,1):
            text = page.extract_text() or ""
            if text.strip():
                paras = [p.strip() for p in text.split("\n\n") if len(p.strip())>=Config.MIN_PARAGRAPH_LENGTH]
                pages_data[i] = {"preview": text[:500], "paragraphs": paras}
        if not pages_data:
            # fallback OCR
            images = pdf2image.convert_from_path(pdf_path,dpi=200)
            for i,img in enumerate(images,1):
                text = pytesseract.image_to_string(img)
                if text.strip():
                    paras = [p.strip() for p in text.split("\n\n") if len(p.strip())>=Config.MIN_PARAGRAPH_LENGTH]
                    pages_data[i] = {"preview": text[:500], "paragraphs": paras}
        return pages_data

# ------------------------ VOICE ------------------------
def speak_text_js(text):
    escaped_text = text.replace('"','\\"')
    html_code=f"""
    <script>
    const msg = new SpeechSynthesisUtterance("{escaped_text}");
    msg.lang = "en-US"; msg.rate=1; window.speechSynthesis.speak(msg);
    </script>
    """
    html(html_code)

def get_voice_input():
    js="""
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang='en-US'; recognition.interimResults=false;
    recognition.maxAlternatives=1; recognition.start();
    recognition.onresult=(event)=>{
        const transcript=event.results[0][0].transcript;
        document.dispatchEvent(new CustomEvent("voice_input",{detail:transcript}));
    };
    </script>
    """
    html(js)
    return st.text_input("Voice Input:", key="voice_text")

# ------------------------ APP ------------------------
class AuraPDFQAApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.neo4j = self.doc_processor.neo4j
        st.set_page_config(page_title="Aura PDF QA ⚡", layout="wide")
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed=False
            st.session_state.messages=[]
            st.session_state.current_document=None
        if 'input_mode' not in st.session_state:
            st.session_state.input_mode="Text"
        if 'auto_voice' not in st.session_state:
            st.session_state.auto_voice=True

    def render_sidebar(self):
        st.sidebar.title("🎯 Aura PDF QA")
        uploaded_file=st.sidebar.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and not st.session_state.pdf_processed:
            if st.sidebar.button("Process PDF"):
                with st.spinner("Processing..."):
                    count=self.doc_processor.process_pdf(uploaded_file)
                    if count>0:
                        st.session_state.pdf_processed=True
                        st.session_state.current_document=uploaded_file.name
                        st.experimental_rerun()
        st.session_state.input_mode=st.sidebar.radio("Input Mode", ["Text","Voice"])
        st.session_state.auto_voice=st.sidebar.checkbox("Auto Voice Response", True)

    def render_chat(self):
        st.title("Aura PDF QA ⚡")
        if not st.session_state.pdf_processed:
            st.info("Upload a PDF from sidebar to build knowledge graph")
            return
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if st.session_state.input_mode=="Voice":
            question=get_voice_input()
        else:
            question=st.chat_input("Ask about relationships in your document...")
        if question:
            self.process_question(question)

    def process_question(self, question):
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("assistant"):
            with st.spinner("Searching relationships..."):
                terms=[w for w in question.lower().split() if len(w)>2]
                results=self.neo4j.search_relationships(terms, st.session_state.current_document)
                if not results:
                    answer="No relationship info found."
                else:
                    answer="**Found Relationships:**\n"
                    for r in results[:5]:
                        answer+=f"- {r['concept']} (Page {r['page']})\n"
                st.markdown(answer)
                if st.session_state.auto_voice:
                    speak_text_js(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})

    def run(self):
        self.render_sidebar()
        self.render_chat()

# ------------------------ MAIN ------------------------
if __name__=="__main__":
    AuraPDFQAApp().run()
