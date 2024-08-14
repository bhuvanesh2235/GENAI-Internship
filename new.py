from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from PyPDF2 import PdfReader
import spacy
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# Initialize Flask app and Spacy model for NLP
app = Flask(__name__)  # Fixed the Flask initialization with __name__
app.secret_key = os.getenv("SECRET_KEY", "default_secret")  # Use a default secret key for session management
nlp = spacy.load("en_core_web_sm")

# Function to get current timestamp
def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Define the interview bot class
class InterviewBot:
    def __init__(self):  # Fixed the constructor method
        self.name = None
        self.personal_details = None
        self.skills = []
        self.current_step = 0
        self.questions = []
        self.prompt_template = """
        You are an interviewer. Based on the candidate's skills: {skills}, generate a list of interview questions.
        Use the skills as a guide to create relevant and insightful interview questions.
        Make sure to use "wh" words or other probing question types to ask questions that are specific, relevant,
        and that probe the candidate's depth of knowledge and experience in each skill. Ask one question at a time.

        Skills: {skills}
        For each skill, generate one question:
        """
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["skills"])
        self.chain = LLMChain(prompt=self.prompt, llm=self.model)

    def extract_skills(self, resume_text):
        doc = nlp(resume_text)
        self.skills = [ent.text for ent in doc.ents if ent.label_ in ("SKILL", "ORG")]
        if not self.skills:
            self.skills = ["general skills"]  # Fallback if no skills are found

    def generate_questions(self):
        if not self.skills:
            self.questions = ["Can you describe your general technical skills?"]
            return

        try:
            raw_questions = self.chain.run({"skills": ', '.join(self.skills)})
            self.questions = [q.strip() for q in raw_questions.split('\n') if q.strip()]
        except Exception as e:
            print(f"Error generating questions: {e}")
            self.questions = ["Unable to generate questions."]

    def analyze_response(self, user_response):
        # Analyze the user response using NLP
        doc = nlp(user_response)
        response_quality = "adequate"

        # Simple analysis: Check if response is relevant to the skills mentioned
        relevant_terms = [ent.text for ent in doc.ents if ent.label_ in ("SKILL", "ORG", "PERSON")]
        if any(skill in relevant_terms for skill in self.skills):
            response_quality = "good"
        else:
            response_quality = "poor"

        return response_quality

    def get_next_question(self, response_quality="adequate"):
        if self.current_step == 0:
            self.current_step += 1
            return "What is your name?"

        elif self.current_step == 1:
            self.current_step += 1
            return "Can you provide some personal details?"

        elif self.current_step == 2:
            self.current_step += 1
            if self.questions:
                question = self.questions.pop(0)
                if response_quality == "poor":
                    question = f"I noticed your previous answer might need some more detail. {question}"
                return f"Let's move on to your skills from the resume. Here is the first question: {question}"
            else:
                return "I don't have more questions about your skills. Can you share more about your experience?"

        elif self.current_step > 2 and self.questions:
            question = self.questions.pop(0)
            if response_quality == "poor":
                question = f"Your last answer could be expanded. {question}"
            return question

        else:
            return "No more questions."

    def handle_unknown_answer(self):
        return "I see. Let's move on to the next topic."

# Initialize the InterviewBot
bot = InterviewBot()

# Function to read PDF content
def extract_resume_text(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        resume_file = request.files['resume']
        resume_text = extract_resume_text(resume_file)
        if not resume_text:
            return jsonify({"error": "Failed to extract text from resume."}), 500

        bot.extract_skills(resume_text)
        bot.generate_questions()
        session['current_step'] = bot.current_step
        session['skills'] = bot.skills
        session['resume_uploaded'] = True  # Mark resume as uploaded
        next_question = bot.get_next_question()
        timestamp = get_timestamp()  # Get current timestamp
        return jsonify({"message": "Resume uploaded and skills extracted successfully.", "question": next_question, "timestamp": timestamp})
    except Exception as e:
        app.logger.error(f"Error in /upload_resume: {e}")
        return jsonify({"error": "Failed to process resume."}), 500

@app.route('/chatbot')
def chatbot():
    if not session.get('resume_uploaded'):
        return redirect(url_for('index'))  # Redirect to upload page if no resume uploaded
    return render_template('mindex.html')

@app.route('/answer', methods=['POST'])
def answer():
    try:
        user_response = request.json.get("response")

        if not user_response:
            return jsonify({"error": "No response provided."}), 400

        # Initialize InterviewBot from session if not already done
        bot.current_step = session.get('current_step', 0)
        bot.skills = session.get('skills', [])

        if bot.current_step == 1:
            bot.name = user_response
        elif bot.current_step == 2:
            bot.personal_details = user_response

        response_quality = bot.analyze_response(user_response)

        if "don't know" in user_response.lower() or "i don't know" in user_response.lower():
            next_question = bot.handle_unknown_answer()
        else:
            next_question = bot.get_next_question(response_quality)

        session['current_step'] = bot.current_step

        timestamp = get_timestamp()  # Get current timestamp
        return jsonify({"question": next_question, "timestamp": timestamp})
    except Exception as e:
        app.logger.error(f"Error in /answer: {e}")
        return jsonify({"error": "Failed to process response."}), 500

if __name__ == '__main__':  # Fixed the main guard with __name__
    app.run(debug=True)
