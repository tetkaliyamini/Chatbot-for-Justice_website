The DOJ AI Chatbot is an AI-powered conversational assistant designed to improve access to legal information on the Department of Justice (DoJ) website. This project leverages natural language processing (NLP) and optical character recognition (OCR) to provide users with accurate responses to both text and image-based queries. Built using Flask, SentenceTransformer, and easyocr, the chatbot facilitates user interaction with legal resources through a seamless, real-time conversational interface.

Features
Text-Based Query Processing: Uses SentenceTransformer to understand user intent and match queries to relevant information in a preloaded legal dataset.
Image-Based Query Processing: Integrates easyocr to extract text from uploaded images, such as legal documents, and process it as a query.
Responsive UI: A simple, intuitive interface built with HTML, CSS, and JavaScript allows users to interact with the chatbot effortlessly.
Scalability: Designed to handle multiple concurrent queries, making it suitable for real-time applications.
Technologies Used
Flask: Lightweight Python web framework for backend routing and API management.
SentenceTransformer: NLP model for understanding and matching user text queries.
easyocr: OCR tool for extracting text from uploaded images.
HTML/CSS/JavaScript: Frontend interface for user interaction.
