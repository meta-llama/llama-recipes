"""Ingestor module for NotebookLlama supporting multiple input formats"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import warnings
import requests

# Core PDF support - required
import PyPDF2

# Optional format support
try:
    from langchain.document_loaders import WebBaseLoader, YoutubeLoader
except ImportError:
    WebBaseLoader = YoutubeLoader = None

import whisper

class BaseIngestor(ABC):
    """Base class for all ingestors"""
    
    @abstractmethod
    def validate(self, source: str) -> bool:
        """Validate if source is valid"""
        pass
    
    @abstractmethod
    def extract_text(self, source: str, max_chars: int = 100000) -> Optional[str]:
        """Extract text from source"""
        pass

class PDFIngestor(BaseIngestor):
    """PDF ingestion - core functionality"""
    
    def validate(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not file_path.lower().endswith('.pdf'):
            print("Error: File is not a PDF")
            return False
        return True

    def extract_text(self, file_path: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                
                extracted_text = []
                total_chars = 0
                
                for page_num in range(num_pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:
                        total_chars += len(page_text)
                        if total_chars > max_chars:
                            remaining_chars = max_chars - total_chars
                            extracted_text.append(page_text[:remaining_chars])
                            print(f"Reached {max_chars} character limit at page {page_num + 1}")
                            break
                        extracted_text.append(page_text)
                        print(f"Processed page {page_num + 1}/{num_pages}")
                
                return "\n".join(extracted_text)
                
        except PyPDF2.PdfReadError:
            print("Error: Invalid or corrupted PDF file")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None

class WebsiteIngestor(BaseIngestor):
    """Website ingestion using LangChain's WebBaseLoader"""

    def validate(self, url: str) -> bool:
        if WebBaseLoader is None:
            print("Error: langchain is not installed. Please install it to use WebsiteIngestor.")
            return False
        if not url.startswith(('http://', 'https://')):
            print("Error: Invalid URL format")
            return False
        return True

    def extract_text(self, url: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(url):
            return None
        
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in documents])
            if len(extracted_text) > max_chars:
                extracted_text = extracted_text[:max_chars]
                print(f"Truncated extracted text to {max_chars} characters")
            print(f"Extracted text from website: {url}")
            return extracted_text
        except Exception as e:
            print(f"An error occurred while extracting from website: {str(e)}")
            return None

class YouTubeIngestor(BaseIngestor):
    """YouTube ingestion using LangChain's YoutubeLoader"""

    def validate(self, youtube_url: str) -> bool:
        if YoutubeLoader is None:
            print("Error: langchain is not installed. Please install it to use YouTubeIngestor.")
            return False
        if not youtube_url.startswith(('http://', 'https://')):
            print("Error: Invalid URL format")
            return False
        return True

    def extract_text(self, youtube_url: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(youtube_url):
            return None
        
        try:
            loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
            documents = loader.load()
            transcript = "\n".join([doc.page_content for doc in documents])
            if len(transcript) > max_chars:
                transcript = transcript[:max_chars]
                print(f"Truncated transcript to {max_chars} characters")
            print(f"Extracted transcript from YouTube video: {youtube_url}")
            return transcript
        except Exception as e:
            print(f"An error occurred while extracting from YouTube: {str(e)}")
            return None

class AudioIngestor(BaseIngestor):
    """Audio ingestion using OpenAI's Whisper model"""

    def __init__(self, model_type: str = "base"):
        self.model_type = model_type
        self.model = whisper.load_model(self.model_type)

    def validate(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found at path: {audio_file}")
            return False
        if not audio_file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            print("Error: Unsupported audio format. Supported formats are .mp3, .wav, .flac, .m4a")
            return False
        return True

    def extract_text(self, audio_file: str, max_chars: int = 100000) -> Optional[str]:
        if not self.validate(audio_file):
            return None
        
        try:
            result = self.model.transcribe(audio_file)
            transcription = result["text"]
            if len(transcription) > max_chars:
                transcription = transcription[:max_chars]
                print(f"Truncated transcription to {max_chars} characters")
            print(f"Transcribed audio file: {audio_file}")
            return transcription
        except Exception as e:
            print(f"An error occurred during audio transcription: {str(e)}")
            return None

class IngestorFactory:
    """Factory to create appropriate ingestor based on input type"""

    @staticmethod
    def get_ingestor(input_type: str, **kwargs) -> Optional[BaseIngestor]:
        """
        Retrieve the appropriate ingestor based on input type.

        Args:
            input_type (str): The type of input ('pdf', 'website', 'youtube', 'audio').

        Returns:
            BaseIngestor: An instance of a concrete Ingestor or None if unsupported.
        """
        input_type = input_type.lower()
        if input_type == "pdf":
            return PDFIngestor()
        elif input_type == "website":
            return WebsiteIngestor()
        elif input_type == "youtube":
            return YouTubeIngestor()
        elif input_type == "audio":
            return AudioIngestor(**kwargs)
        else:
            print(f"Unsupported input type: {input_type}")
            return None

def get_ingestor(source: str) -> BaseIngestor:
    """Factory function to get appropriate ingestor based on source type"""
    if source.lower().endswith('.pdf'):
        return PDFIngestor()
    elif source.startswith('http'):
        if 'youtube.com' in source or 'youtu.be' in source:
            return YouTubeIngestor()
        return WebsiteIngestor()
    raise ValueError(f"Unsupported source type: {source}")

def ingest_content(input_type: str, source: str, max_chars: int = 100000, **kwargs) -> Optional[str]:
    """
    Ingest content from various sources based on the input type.

    Args:
        input_type (str): Type of the input ('pdf', 'website', 'youtube', 'audio').
        source (str): Path to the file or URL.
        max_chars (int, optional): Maximum number of characters to extract. Defaults to 100000.
        **kwargs: Additional arguments for specific ingestors.

    Returns:
        Optional[str]: Extracted text or None if ingestion fails.
    """
    ingestor = IngestorFactory.get_ingestor(input_type, **kwargs)
    if not ingestor:
        print(f"Failed to get ingestor for input type: {input_type}")
        return None
    return ingestor.extract_text(source, max_chars)

