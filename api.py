import os
import logging
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Optional
import hashlib
import json

from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from twilio.rest import Client

# Configurer la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFRAGService:
    """Service professionnel de RAG pour PDF avec mise en cache et optimisation"""
    
    def __init__(self, api_key: str, pdf_directory: str = "pdfs"):
        self.api_key = api_key
        self.pdf_directory = pdf_directory
        self.vector_store = None
        self.embeddings = None
        self.chain = None
        self.pdf_metadata = {}
        
        # Initialiser les composants
        self._initialize_embeddings()
        self._load_and_process_pdfs()
        self._initialize_chain()
    
    def _initialize_embeddings(self):
        """Initialiser les embeddings Google AI"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=self.api_key
            )
            logger.info("Embeddings initialisés avec succès")
        except Exception as e:
            logger.error(f"Échec de l'initialisation des embeddings : {str(e)}")
            raise
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extraire le texte d'un seul fichier PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte de {pdf_path} : {str(e)}")
            return ""
    
    def _get_text_chunks(self, text: str) -> List[str]:
        """Diviser le texte en morceaux pour le traitement"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, 
            chunk_overlap=1000
        )
        return text_splitter.split_text(text)
    
    def _load_and_process_pdfs(self):
        """Charger tous les PDF et JSON du répertoire et créer le magasin de vecteurs"""
        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            logger.warning(f"Répertoire PDF créé : {self.pdf_directory}")
            return

        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
        json_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.json')]

        if not pdf_files and not json_files:
            logger.warning(f"Aucun fichier PDF ou JSON trouvé dans {self.pdf_directory}")
            return

        logger.info(f"{len(pdf_files)} fichiers PDF et {len(json_files)} fichiers JSON trouvés à traiter")

        all_chunks = []
        chunk_metadata = []

        # Traitement des PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Traitement de {pdf_file}...")

            # Extraire le texte et créer des morceaux
            text = self._extract_pdf_text(pdf_path)
            if text:
                chunks = self._get_text_chunks(text)
                all_chunks.extend(chunks)

                # Stocker les métadonnées
                self.pdf_metadata[pdf_file] = {
                    'path': pdf_path,
                    'chunk_count': len(chunks),
                    'processed_at': datetime.now().isoformat()
                }

                # Créer des métadonnées pour chaque morceau
                for i, chunk in enumerate(chunks):
                    chunk_metadata.append({
                        'source': pdf_file,
                        'chunk_id': i,
                        'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
                    })

        # Traitement des JSON
        for json_file in json_files:
            json_path = os.path.join(self.pdf_directory, json_file)
            logger.info(f"Traitement de {json_file}...")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                    # Fonction récursive pour aplatir le JSON en chunks de texte
                    def flatten_json(obj, prefix=""):
                        chunks = []
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                new_prefix = f"{prefix}{k}: " if prefix else f"{k}: "
                                chunks.extend(flatten_json(v, new_prefix))
                        elif isinstance(obj, list):
                            for item in obj:
                                chunks.extend(flatten_json(item, prefix))
                        else:
                            # Feuille : on ajoute le texte
                            text = f"{prefix}{obj}"
                            chunks.append(text)
                        return chunks

                    json_chunks = flatten_json(json_data)
                    for i, chunk in enumerate(json_chunks):
                        all_chunks.append(chunk)
                        chunk_metadata.append({
                            'source': json_file,
                            'chunk_id': i,
                            'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
                        })
            except Exception as e:
                logger.error(f"Erreur lors du traitement du JSON {json_file} : {str(e)}")

        if all_chunks:
            try:
                # Créer le magasin de vecteurs
                self.vector_store = FAISS.from_texts(
                    all_chunks,
                    embedding=self.embeddings,
                    metadatas=chunk_metadata
                )

                # Sauvegarder le magasin de vecteurs pour une utilisation future
                vector_store_path = "faiss_index_api"
                self.vector_store.save_local(vector_store_path)

                logger.info(f"Magasin de vecteurs créé avec {len(all_chunks)} morceaux à partir de {len(pdf_files)} PDFs et {len(json_files)} JSONs")

            except Exception as e:
                logger.error(f"Échec de la création du magasin de vecteurs : {str(e)}")
                raise
        else:
            logger.error("Aucun texte extrait des fichiers PDF ou JSON")
            raise ValueError("Aucun contenu PDF ou JSON valide trouvé")
    
    def _initialize_chain(self):
        """Initialiser la chaîne conversationnelle"""
        prompt_template = """
        Répondez aux questions en fournissant autant de contexte que possible. Si vous ne connaissez pas la réponse, 
        dites-leur de contacter Penpen Berboss ou d'ecrire sur le site de ECEMA pour plus d'informations. Répondez dans la même langue que la question de l'utilisateur.

        Contexte :\n{context}\n
        Question :\n{question}\n

        Réponse :
        """
        
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.3, 
                google_api_key=self.api_key
            )
            
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            self.chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            logger.info("Chaîne conversationnelle initialisée avec succès")
            
        except Exception as e:
            logger.error(f"Échec de l'initialisation de la chaîne : {str(e)}")
            raise
    
    @lru_cache(maxsize=128)
    def _get_cached_similarity_search(self, question_hash: str, question: str, k: int = 4):
        """Recherche de similarité mise en cache pour éviter les embeddings répétés"""
        if not self.vector_store:
            raise ValueError("Magasin de vecteurs non initialisé")
        
        return self.vector_store.similarity_search(question, k=k)
    
    def query(self, question: str) -> Dict:
        """Traiter une requête et retourner la réponse"""
        if not question or not question.strip():
            raise BadRequest("La question ne peut pas être vide")
        
        if not self.vector_store or not self.chain:
            raise InternalServerError("Système RAG non correctement initialisé")
        
        try:
            # Créer un hash pour la mise en cache
            question_hash = hashlib.md5(question.encode()).hexdigest()
            
            # Obtenir les documents pertinents
            docs = self._get_cached_similarity_search(question_hash, question)
            
            # Générer la réponse
            response = self.chain(
                {"input_documents": docs, "question": question}, 
                return_only_outputs=True
            )
            
            # Préparer la réponse avec les métadonnées
            result = {
                "question": question,
                "answer": response['output_text'],
                "timestamp": datetime.now().isoformat(),
                "sources_used": len(docs),
                "pdf_sources": list(set([doc.metadata.get('source', 'unknown') for doc in docs if hasattr(doc, 'metadata')]))
            }
            
            logger.info(f"Requête traitée avec succès : {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête : {str(e)}")
            raise InternalServerError(f"Échec du traitement de la requête : {str(e)}")
    
    def get_system_info(self) -> Dict:
        """Obtenir des informations sur les PDF chargés et l'état du système"""
        return {
            "status": "actif",
            "pdfs_loaded": len(self.pdf_metadata),
            "pdf_files": list(self.pdf_metadata.keys()),
            "pdf_details": self.pdf_metadata,
            "vector_store_ready": self.vector_store is not None,
            "chain_ready": self.chain is not None
        }

# Application Flask
app = Flask(__name__)
CORS(app)  # Autorise toutes les origines (pour le développement)

# Configurer la mise en cache
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
cache = Cache(app)

# Instance globale du service RAG
rag_service = None

def initialize_rag_service():
    """Initialiser le service RAG au démarrage"""
    global rag_service
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("La variable d'environnement GOOGLE_API_KEY est requise")
    
    pdf_directory = os.getenv('PDF_DIRECTORY', 'pdfs')
    
    try:
        rag_service = PDFRAGService(api_key, pdf_directory)
        logger.info("Service RAG initialisé avec succès")
    except Exception as e:
        logger.error(f"Échec de l'initialisation du service RAG : {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Point de terminaison de vérification de santé"""
    if rag_service is None:
        return jsonify({"status": "erreur", "message": "Service RAG non initialisé"}), 503
    
    return jsonify({"status": "sain", "timestamp": datetime.now().isoformat()})

@app.route('/info', methods=['GET'])
def system_info():
    """Obtenir les informations du système"""
    if rag_service is None:
        return jsonify({"error": "Service RAG non initialisé"}), 503
    
    try:
        info = rag_service.get_system_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des informations du système : {str(e)}")
        return jsonify({"error": "Échec de l'obtention des informations du système"}), 500

@app.route('/query', methods=['POST'])
# @cache.cached(timeout=300, key_prefix='query')
def query_pdfs():
    """Interroger les PDF"""
    if rag_service is None:
        return jsonify({"error": "Service RAG non initialisé"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "La requête doit contenir le champ 'question'"}), 400
        
        question = data['question']
        result = rag_service.query(question)
        
        logger.info(f"Réponse brute de l'API RAG : {data}")

        return jsonify(result)
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except InternalServerError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Erreur inattendue dans le point de terminaison query : {str(e)}")
        return jsonify({"error": "Une erreur inattendue s'est produite"}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Vider le cache de l'application"""
    try:
        # cache.clear()
        return jsonify({"message": "Cache vidé avec succès"})
    except Exception as e:
        logger.error(f"Erreur lors du vidage du cache : {str(e)}")
        return jsonify({"error": "Échec du vidage du cache"}), 500

@app.route('/whatsapp', methods=['POST'])
def send_whatsapp_message():
    try:
        data = request.get_json()
        to_number = data.get('to')
        content_sid = data.get('content_sid')
        content_variables = data.get('content_variables', '{}')

        account_sid = 'AC4661b5fc371a756e4612313b05a94797'
        auth_token = '[AuthToken]'  # Remplace par ton vrai token sécurisé
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            from_='whatsapp:+14155238886',
            content_sid=content_sid,
            content_variables=content_variables,
            to=f'whatsapp:{to_number}'
        )

        return jsonify({"status": "success", "sid": message.sid}), 200
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi WhatsApp : {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Point de terminaison non trouvé"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == '__main__':
    try:
        load_dotenv()
        initialize_rag_service()
        
        # Démarrer l'application Flask
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Démarrage de l'application Flask sur le port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Échec du démarrage de l'application : {str(e)}")
        exit(1)