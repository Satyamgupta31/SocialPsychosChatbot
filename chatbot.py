import re
import json
import random
import datetime
import csv
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os
from collections import defaultdict, Counter

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"

class Language(Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"

class ConversationState(Enum):
    GREETING = "greeting"
    PHQ9_ASSESSMENT = "phq9_assessment"
    GAD7_ASSESSMENT = "gad7_assessment"
    GENERAL_SUPPORT = "general_support"
    CRISIS_INTERVENTION = "crisis_intervention"
    FOLLOWUP = "followup"

@dataclass
class AssessmentResponse:
    question_id: int
    question: str
    response: int
    timestamp: str

@dataclass
class UserSession:
    user_id: str
    name: str
    language: Language
    conversation_history: List[Dict]
    conversation_state: ConversationState
    current_question_index: int = 0
    phq9_responses: List[AssessmentResponse] = None
    gad7_responses: List[AssessmentResponse] = None
    phq9_score: int = 0
    gad7_score: int = 0
    risk_score: int = 0
    session_start: str = None
    assessment_mandatory: bool = True
    
    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.datetime.now().isoformat()
        if self.phq9_responses is None:
            self.phq9_responses = []
        if self.gad7_responses is None:
            self.gad7_responses = []

class LightweightMentalHealthChatbot:
    def __init__(self, dataset_path: Optional[str] = None):
        self.sessions = {}
        self.training_data = []
        self.risk_patterns = self._initialize_risk_patterns()
        
        # Crisis keywords (expanded)
        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "want to die", "hurt myself",
            "self harm", "ending it all", "can't go on", "hopeless forever",
            "worthless", "better off dead", "harm", "cut myself", "overdose",
            "jump off", "hang myself", "no point living", "everyone better without me",
            "want to disappear", "end it all", "nothing matters", "give up completely"
        ]
        
        # PHQ-9 Questions (standardized clinical version)
        self.phq9_questions = [
            {
                "id": 1,
                "question": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
                "domain": "anhedonia"
            },
            {
                "id": 2,
                "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
                "domain": "depressed_mood"
            },
            {
                "id": 3,
                "question": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
                "domain": "sleep_disturbance"
            },
            {
                "id": 4,
                "question": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
                "domain": "fatigue"
            },
            {
                "id": 5,
                "question": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
                "domain": "appetite"
            },
            {
                "id": 6,
                "question": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
                "domain": "self_worth"
            },
            {
                "id": 7,
                "question": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
                "domain": "concentration"
            },
            {
                "id": 8,
                "question": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed - or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
                "domain": "psychomotor"
            },
            {
                "id": 9,
                "question": "Over the last 2 weeks, how often have you had thoughts that you would be better off dead, or of hurting yourself in some way?",
                "domain": "suicidal_ideation"
            }
        ]
        
        # GAD-7 Questions (standardized clinical version)
        self.gad7_questions = [
            {
                "id": 1,
                "question": "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge?",
                "domain": "nervousness"
            },
            {
                "id": 2,
                "question": "Over the last 2 weeks, how often have you been bothered by not being able to stop or control worrying?",
                "domain": "uncontrollable_worry"
            },
            {
                "id": 3,
                "question": "Over the last 2 weeks, how often have you been bothered by worrying too much about different things?",
                "domain": "excessive_worry"
            },
            {
                "id": 4,
                "question": "Over the last 2 weeks, how often have you been bothered by trouble relaxing?",
                "domain": "relaxation_difficulty"
            },
            {
                "id": 5,
                "question": "Over the last 2 weeks, how often have you been bothered by being so restless that it is hard to sit still?",
                "domain": "restlessness"
            },
            {
                "id": 6,
                "question": "Over the last 2 weeks, how often have you been bothered by becoming easily annoyed or irritable?",
                "domain": "irritability"
            },
            {
                "id": 7,
                "question": "Over the last 2 weeks, how often have you been bothered by feeling afraid, as if something awful might happen?",
                "domain": "catastrophic_worry"
            }
        ]
        
        # Response scale
        self.response_scale = {
            0: "Not at all",
            1: "Several days", 
            2: "More than half the days",
            3: "Nearly every day"
        }
        
        # Multilingual responses
        self.responses = {
            Language.ENGLISH: {
                "greeting": [
                    "Hello! I'm your mental health support companion. Before we chat, I need to ask you some important questions about how you've been feeling lately. This helps me better understand and support you.",
                    "Hi there! I'm here to provide you with personalized mental health support. To get started, I'll need to conduct a brief assessment to understand your current wellbeing.",
                    "Welcome! I'm glad you're here. To provide you with the best support, I'll start with a standardized assessment. This is completely confidential and will help me tailor our conversation to your needs."
                ],
                "phq9_intro": [
                    "I'll start with some questions about depression symptoms. Please answer honestly - this information is confidential and will help me support you better.",
                    "First, I need to assess for depression symptoms using a standardized questionnaire. Your honest responses will help me provide appropriate support."
                ],
                "gad7_intro": [
                    "Now I'll ask about anxiety symptoms. Again, please respond honestly based on how you've been feeling over the past 2 weeks.",
                    "Next, let's assess anxiety symptoms. These questions help identify if anxiety is affecting your daily life."
                ],
                "assessment_complete": [
                    "Thank you for completing the assessments. Based on your responses, I can now provide personalized support and recommendations.",
                    "Assessment complete! I now have a better understanding of how you're feeling. Let's discuss your results and how I can help."
                ],
                "crisis_detected": [
                    "I'm very concerned about what you've shared, especially regarding thoughts of self-harm. Your safety is the top priority. Please contact emergency services (112) or reach out to a mental health professional immediately.",
                    "What you've described indicates you may be in crisis. Please don't hesitate to seek immediate professional help. You don't have to go through this alone."
                ]
            },
            Language.HINDI: {
                "greeting": [
                    "नमस्ते! मैं आपका मानसिक स्वास्थ्य सहायक हूँ। बातचीत शुरू करने से पहले, मुझे आपकी भावनाओं के बारे में कुछ महत्वपूर्ण प्रश्न पूछने होंगे।",
                    "हैलो! मैं यहाँ आपको व्यक्तिगत मानसिक स्वास्थ्य सहायता प्रदान करने के लिए हूँ। शुरुआत के लिए, मैं एक संक्षिप्त मूल्यांकन करूंगा।"
                ],
                "phq9_intro": [
                    "मैं अवसाद के लक्षणों के बारे में कुछ प्रश्न पूछूंगा। कृपया ईमानदारी से उत्तर दें - यह जानकारी गोपनीय है।"
                ],
                "gad7_intro": [
                    "अब मैं चिंता के लक्षणों के बारे में पूछूंगा। कृपया पिछले 2 हफ्तों के आधार पर ईमानदारी से जवाब दें।"
                ]
            }
        }
        
        # Emergency contacts
        self.emergency_contacts = {
            "KIRAN Helpline": "1800-599-0019",
            "NIMHANS Helpline": "080-26995000", 
            "Emergency Services": "112",
            "Student Helpline": "1800-XXX-XXXX",
            "Suicide Prevention": "9152987821"
        }
        
        # Load dataset if provided
        if dataset_path and os.path.exists(dataset_path):
            self.load_dataset(dataset_path)
        else:
            self.create_sample_dataset()

    def _initialize_risk_patterns(self) -> Dict[str, List[str]]:
        """Initialize risk assessment patterns"""
        return {
            "crisis": [
                "suicide", "kill myself", "end my life", "want to die", "hurt myself",
                "self harm", "ending it all", "better off dead", "no point living",
                "want to disappear", "overdose", "jump off", "hang myself"
            ],
            "high": [
                "hopeless", "worthless", "failure", "can't cope", "desperate",
                "panic attacks", "can't breathe", "falling apart", "breaking down",
                "nothing matters", "empty inside", "completely alone", "giving up"
            ],
            "moderate": [
                "stressed", "worried", "anxious", "overwhelmed", "tired", "sad",
                "frustrated", "angry", "confused", "lost", "struggling", "difficult",
                "pressure", "burden", "exhausted", "drained", "unmotivated"
            ],
            "low": [
                "good", "fine", "okay", "better", "happy", "content", "peaceful",
                "optimistic", "hopeful", "grateful", "blessed", "improving", "positive"
            ]
        }

    def create_sample_dataset(self):
        """Create sample training dataset for demonstration"""
        sample_data = [
            # Crisis samples
            {"text": "I want to end my life", "risk_level": "crisis"},
            {"text": "I'm thinking about suicide", "risk_level": "crisis"},
            {"text": "I want to hurt myself", "risk_level": "crisis"},
            {"text": "Everyone would be better off without me", "risk_level": "crisis"},
            {"text": "I can't go on living like this", "risk_level": "crisis"},
            {"text": "I want to disappear forever", "risk_level": "crisis"},
            
            # High risk samples
            {"text": "I feel hopeless about everything", "risk_level": "high"},
            {"text": "I can't seem to enjoy anything anymore", "risk_level": "high"},
            {"text": "I feel like a complete failure", "risk_level": "high"},
            {"text": "Nothing seems to matter anymore", "risk_level": "high"},
            {"text": "I feel so alone and isolated", "risk_level": "high"},
            {"text": "I'm having panic attacks regularly", "risk_level": "high"},
            {"text": "I can't cope with life anymore", "risk_level": "high"},
            {"text": "I'm falling apart completely", "risk_level": "high"},
            
            # Moderate risk samples  
            {"text": "I'm feeling a bit stressed about exams", "risk_level": "moderate"},
            {"text": "Having some trouble sleeping lately", "risk_level": "moderate"},
            {"text": "Feeling overwhelmed with assignments", "risk_level": "moderate"},
            {"text": "I'm worried about my grades this semester", "risk_level": "moderate"},
            {"text": "Having difficulty concentrating on studies", "risk_level": "moderate"},
            {"text": "Feeling tired all the time", "risk_level": "moderate"},
            {"text": "I'm anxious about my future career", "risk_level": "moderate"},
            {"text": "The pressure is getting to me", "risk_level": "moderate"},
            
            # Low risk samples
            {"text": "I'm feeling good today, had a great class", "risk_level": "low"},
            {"text": "Everything is going well with my studies", "risk_level": "low"},
            {"text": "I'm happy with my progress this semester", "risk_level": "low"},
            {"text": "Having a peaceful day, feeling content", "risk_level": "low"},
            {"text": "I feel optimistic about my future", "risk_level": "low"},
            {"text": "Things are looking up for me", "risk_level": "low"}
        ]
        
        self.training_data = sample_data
        print(f"✅ Created sample dataset with {len(sample_data)} examples")
        
    def load_dataset(self, dataset_path: str):
        """Load custom dataset from CSV file"""
        try:
            self.training_data = []
            
            with open(dataset_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    if 'text' in row and 'risk_level' in row:
                        self.training_data.append({
                            'text': row['text'].strip(),
                            'risk_level': row['risk_level'].strip().lower()
                        })
                    
            print(f"✅ Loaded dataset with {len(self.training_data)} samples from {dataset_path}")
            
            # Update risk patterns based on training data
            self._update_risk_patterns_from_data()
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("Creating sample dataset instead...")
            self.create_sample_dataset()

    def _update_risk_patterns_from_data(self):
        """Update risk patterns based on training data"""
        risk_words = defaultdict(list)
        
        for item in self.training_data:
            words = re.findall(r'\b\w+\b', item['text'].lower())
            risk_level = item['risk_level']
            
            if risk_level in ['crisis', 'high', 'moderate', 'low']:
                risk_words[risk_level].extend(words)
        
        # Update patterns with most common words for each risk level
        for level, words in risk_words.items():
            word_counts = Counter(words)
            # Keep top 20 most common words for each level
            common_words = [word for word, count in word_counts.most_common(20)]
            self.risk_patterns[level].extend([w for w in common_words if w not in self.risk_patterns[level]])

    def predict_risk_level(self, text: str) -> Tuple[str, float]:
        """Predict risk level using pattern matching and scoring"""
        text_lower = text.lower()
        
        # First check for crisis keywords
        if self.detect_crisis(text):
            return "crisis", 1.0
        
        # Score each risk level
        scores = {}
        
        for level, patterns in self.risk_patterns.items():
            score = 0
            words_found = []
            
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
                    words_found.append(pattern)
            
            # Weight scores based on risk level importance
            if level == "crisis":
                score *= 4
            elif level == "high":
                score *= 3
            elif level == "moderate":
                score *= 2
            elif level == "low":
                score *= 1
                
            scores[level] = score
        
        # Determine highest scoring risk level
        if max(scores.values()) == 0:
            return "moderate", 0.5  # Default if no patterns match
            
        predicted_level = max(scores, key=scores.get)
        confidence = min(scores[predicted_level] / 10.0, 1.0)  # Normalize to 0-1
        
        return predicted_level, confidence

    def detect_crisis(self, text: str) -> bool:
        """Enhanced crisis detection"""
        text_lower = text.lower()
        
        # Direct keyword matching
        direct_match = any(keyword in text_lower for keyword in self.crisis_keywords)
        
        # Pattern matching for suicidal ideation
        patterns = [
            r'\bi want to die\b',
            r'\bkill me\b', 
            r'\bend my life\b',
            r'\bbetter off dead\b',
            r'\bno reason to live\b',
            r'\bwant to disappear\b',
            r'\bending it all\b',
            r'\bcan\'t go on\b'
        ]
        
        pattern_match = any(re.search(pattern, text_lower) for pattern in patterns)
        
        return direct_match or pattern_match

    def create_session(self, user_id: str, name: str, language: Language = Language.ENGLISH) -> UserSession:
        """Create a new user session with mandatory assessments"""
        session = UserSession(
            user_id=user_id,
            name=name,
            language=language,
            conversation_history=[],
            conversation_state=ConversationState.GREETING
        )
        self.sessions[user_id] = session
        return session

    def process_message(self, user_id: str, message: str) -> Dict:
        """Process user message based on conversation state"""
        if user_id not in self.sessions:
            return {"error": "Session not found. Please start a new conversation."}
        
        session = self.sessions[user_id]
        
        # Handle different conversation states
        if session.conversation_state == ConversationState.GREETING:
            return self.handle_greeting(user_id, message)
        elif session.conversation_state == ConversationState.PHQ9_ASSESSMENT:
            return self.handle_phq9_assessment(user_id, message)
        elif session.conversation_state == ConversationState.GAD7_ASSESSMENT:
            return self.handle_gad7_assessment(user_id, message)
        elif session.conversation_state == ConversationState.GENERAL_SUPPORT:
            return self.handle_general_support(user_id, message)
        elif session.conversation_state == ConversationState.CRISIS_INTERVENTION:
            return self.handle_crisis_followup(user_id, message)
        else:
            return self.handle_general_support(user_id, message)

    def handle_greeting(self, user_id: str, message: str) -> Dict:
        """Handle initial greeting and start PHQ-9"""
        session = self.sessions[user_id]
        language = session.language
        
        # Log the initial message
        self.log_conversation(user_id, message, "user")
        
        # Start PHQ-9 assessment
        session.conversation_state = ConversationState.PHQ9_ASSESSMENT
        session.current_question_index = 0
        
        intro_message = random.choice(self.responses[language]["phq9_intro"])
        first_question = self.phq9_questions[0]["question"]
        
        response_text = f"{intro_message}\n\n**Question 1 of 9:**\n{first_question}\n\n**Please respond with:**\n0 = Not at all\n1 = Several days\n2 = More than half the days\n3 = Nearly every day"
        
        self.log_conversation(user_id, response_text, "bot")
        
        return {
            "user_id": user_id,
            "message": response_text,
            "conversation_state": session.conversation_state.value,
            "assessment_progress": "PHQ-9: 1/9",
            "requires_numeric_response": True,
            "valid_responses": [0, 1, 2, 3],
            "response_scale": self.response_scale
        }

    def handle_phq9_assessment(self, user_id: str, message: str) -> Dict:
        """Handle PHQ-9 assessment responses"""
        session = self.sessions[user_id]
        
        # Validate numeric response
        try:
            response_value = int(message.strip())
            if response_value not in [0, 1, 2, 3]:
                raise ValueError("Invalid response")
        except ValueError:
            return {
                "user_id": user_id,
                "message": "Please provide a valid response (0, 1, 2, or 3).\n\n**Response Scale:**\n0 = Not at all\n1 = Several days\n2 = More than half the days\n3 = Nearly every day",
                "conversation_state": session.conversation_state.value,
                "error": "Invalid input",
                "requires_numeric_response": True,
                "valid_responses": [0, 1, 2, 3]
            }
        
        # Store response
        current_question = self.phq9_questions[session.current_question_index]
        assessment_response = AssessmentResponse(
            question_id=current_question["id"],
            question=current_question["question"],
            response=response_value,
            timestamp=datetime.datetime.now().isoformat()
        )
        session.phq9_responses.append(assessment_response)
        
        # Check if PHQ-9 is complete
        if session.current_question_index < 8:
            # Continue with next PHQ-9 question
            session.current_question_index += 1
            next_question = self.phq9_questions[session.current_question_index]
            
            response_text = f"**Question {session.current_question_index + 1} of 9:**\n{next_question['question']}\n\n**Please respond with:**\n0 = Not at all\n1 = Several days\n2 = More than half the days\n3 = Nearly every day"
            
            return {
                "user_id": user_id,
                "message": response_text,
                "conversation_state": session.conversation_state.value,
                "assessment_progress": f"PHQ-9: {session.current_question_index + 1}/9",
                "requires_numeric_response": True,
                "valid_responses": [0, 1, 2, 3]
            }
        else:
            # PHQ-9 complete, calculate score and start GAD-7
            session.phq9_score = sum(resp.response for resp in session.phq9_responses)
            session.conversation_state = ConversationState.GAD7_ASSESSMENT
            session.current_question_index = 0
            
            # Check for suicidal ideation (PHQ-9 question 9)
            if session.phq9_responses[8].response > 0:
                return self.handle_crisis_intervention(user_id)
            
            language = session.language
            intro_message = random.choice(self.responses[language]["gad7_intro"])
            first_gad7_question = self.gad7_questions[0]["question"]
            
            response_text = f"✅ **PHQ-9 completed!** Now let's assess anxiety symptoms.\n\n{intro_message}\n\n**Question 1 of 7:**\n{first_gad7_question}\n\n**Please respond with:**\n0 = Not at all\n1 = Several days\n2 = More than half the days\n3 = Nearly every day"
            
            return {
                "user_id": user_id,
                "message": response_text,
                "conversation_state": session.conversation_state.value,
                "assessment_progress": "GAD-7: 1/7",
                "phq9_score": session.phq9_score,
                "requires_numeric_response": True,
                "valid_responses": [0, 1, 2, 3]
            }

    def handle_gad7_assessment(self, user_id: str, message: str) -> Dict:
        """Handle GAD-7 assessment responses"""
        session = self.sessions[user_id]
        
        # Validate numeric response
        try:
            response_value = int(message.strip())
            if response_value not in [0, 1, 2, 3]:
                raise ValueError("Invalid response")
        except ValueError:
            return {
                "user_id": user_id,
                "message": "Please provide a valid response (0, 1, 2, or 3).\n\n**Response Scale:**\n0 = Not at all\n1 = Several days\n2 = More than half the days\n3 = Nearly every day",
                "conversation_state": session.conversation_state.value,
                "error": "Invalid input",
                "requires_numeric_response": True,
                "valid_responses": [0, 1, 2, 3]
            }
        
        # Store response
        current_question = self.gad7_questions[session.current_question_index]
        assessment_response = AssessmentResponse(
            question_id=current_question["id"],
            question=current_question["question"],
            response=response_value,
            timestamp=datetime.datetime.now().isoformat()
        )
        session.gad7_responses.append(assessment_response)
        
        # Check if GAD-7 is complete
        if session.current_question_index < 6:
            # Continue with next GAD-7 question
            session.current_question_index += 1
            next_question = self.gad7_questions[session.current_question_index]
            
            response_text = f"**Question {session.current_question_index + 1} of 7:**\n{next_question['question']}\n\n**Please respond with:**\n0 = Not at all\n1 = Several days\n2 = More than half the days\n3 = Nearly every day"
            
            return {
                "user_id": user_id,
                "message": response_text,
                "conversation_state": session.conversation_state.value,
                "assessment_progress": f"GAD-7: {session.current_question_index + 1}/7",
                "requires_numeric_response": True,
                "valid_responses": [0, 1, 2, 3]
            }
        else:
            # GAD-7 complete, calculate scores and provide results
            session.gad7_score = sum(resp.response for resp in session.gad7_responses)
            session.conversation_state = ConversationState.GENERAL_SUPPORT
            
            return self.provide_assessment_results(user_id)

    def handle_crisis_intervention(self, user_id: str) -> Dict:
        """Handle crisis intervention when suicidal ideation is detected"""
        session = self.sessions[user_id]
        session.conversation_state = ConversationState.CRISIS_INTERVENTION
        language = session.language
        
        crisis_message = random.choice(self.responses[language]["crisis_detected"])
        
        emergency_info = "\n\n🚨 **IMMEDIATE HELP AVAILABLE 24/7:**\n"
        for service, number in self.emergency_contacts.items():
            emergency_info += f"📞 {service}: {number}\n"
        
        response_text = f"{crisis_message}{emergency_info}\n\n❗ **Please reach out to one of these services RIGHT NOW.**\n\nWould you like me to help you connect with a counselor or continue talking? (Type 'help' for counselor or 'talk' to continue)"
        
        self.log_conversation(user_id, response_text, "bot")
        
        return {
            "user_id": user_id,
            "message": response_text,
            "conversation_state": session.conversation_state.value,
            "crisis_level": "HIGH",
            "requires_immediate_attention": True,
            "emergency_contacts": self.emergency_contacts,
            "phq9_score": session.phq9_score,
            "suicidal_ideation_detected": True,
            "alert_counselor": True
        }

    def handle_crisis_followup(self, user_id: str, message: str) -> Dict:
        """Handle followup after crisis intervention"""
        session = self.sessions[user_id]
        message_lower = message.lower()
        
        if "help" in message_lower:
            response_text = "🏥 **Connecting you with professional help:**\n\n"
            response_text += "I'm notifying our counseling team about your situation. Someone will contact you within 15 minutes.\n\n"
            response_text += "In the meantime, please:\n"
            response_text += "• Stay with someone you trust\n"
            response_text += "• Call emergency services (112) if you feel unsafe\n"
            response_text += "• Remember: You are valuable and help is available\n\n"
            response_text += "Is there someone with you right now?"
            
            return {
                "user_id": user_id,
                "message": response_text,
                "counselor_alert": True,
                "immediate_followup": True,
                "crisis_level": "HIGH"
            }
        else:
            # Continue conversation with crisis awareness
            session.conversation_state = ConversationState.GENERAL_SUPPORT
            return self.handle_general_support(user_id, message)

    def provide_assessment_results(self, user_id: str) -> Dict:
        """Provide comprehensive assessment results and recommendations"""
        session = self.sessions[user_id]
        language = session.language
        
        # Calculate interpretations
        phq9_interpretation = self.interpret_phq9_score(session.phq9_score)
        gad7_interpretation = self.interpret_gad7_score(session.gad7_score)
        
        # Generate comprehensive report
        assessment_complete_msg = random.choice(self.responses[language]["assessment_complete"])
        
        results_text = f"🎉 {assessment_complete_msg}\n\n"
        results_text += f"📊 **YOUR ASSESSMENT RESULTS:**\n\n"
        results_text += f"🧠 **Depression Screening (PHQ-9):**\n"
        results_text += f"• Score: {session.phq9_score}/27\n"
        results_text += f"• Level: {phq9_interpretation['severity']}\n"
        results_text += f"• Recommendation: {phq9_interpretation['recommendation']}\n\n"
        results_text += f"😰 **Anxiety Screening (GAD-7):**\n"
        results_text += f"• Score: {session.gad7_score}/21\n"
        results_text += f"• Level: {gad7_interpretation['severity']}\n"
        results_text += f"• Recommendation: {gad7_interpretation['recommendation']}\n\n"
        
        # Overall risk assessment
        overall_risk = self.calculate_overall_risk(session.phq9_score, session.gad7_score)
        results_text += f"⚠️ **Overall Risk Level: {overall_risk['level'].upper()}**\n"
        results_text += f"🎯 **Priority Actions:** {overall_risk['actions']}\n\n"
        
        # Personalized coping strategies
        coping_strategies = self.get_personalized_coping_strategies(session.phq9_score, session.gad7_score)
        results_text += f"💡 **Personalized Coping Strategies:**\n"
        for i, strategy in enumerate(coping_strategies, 1):
            results_text += f"{i}. {strategy}\n"
        
        results_text += f"\n💬 **Now, how are you feeling right now? I'm here to support you through whatever you're experiencing.**"
        
        self.log_conversation(user_id, results_text, "bot")
        
        return {
            "user_id": user_id,
            "message": results_text,
            "conversation_state": session.conversation_state.value,
            "assessment_results": {
                "phq9": {
                    "score": session.phq9_score,
                    "max_score": 27,
                    "severity": phq9_interpretation['severity'],
                    "recommendation": phq9_interpretation['recommendation']
                },
                "gad7": {
                    "score": session.gad7_score,
                    "max_score": 21,
                    "severity": gad7_interpretation['severity'],
                    "recommendation": gad7_interpretation['recommendation']
                },
                "overall_risk": overall_risk
            },
            "coping_strategies": coping_strategies,
            "requires_followup": overall_risk['level'] in ['moderate', 'high', 'crisis'],
            "assessment_completed": True
        }

    def handle_general_support(self, user_id: str, message: str) -> Dict:
        """Handle general support conversation after assessments"""
        session = self.sessions[user_id]
        
        # Log user message
        self.log_conversation(user_id, message, "user")
        
        # Use risk assessment to evaluate current message
        risk_level, confidence = self.predict_risk_level(message)
        
        # Check for crisis in current message
        if self.detect_crisis(message) or risk_level == "crisis":
            return self.handle_crisis_intervention(user_id)
        
        # Generate contextual response based on assessment results and current message
        response = self.generate_contextual_response(user_id, message, risk_level, confidence)
        
        self.log_conversation(user_id, response["message"], "bot")
        
        return response

    def generate_contextual_response(self, user_id: str, message: str, risk_level: str, confidence: float) -> Dict:
        """Generate contextual response based on assessments and current message"""
        session = self.sessions[user_id]
        
        # Base supportive response
        supportive_responses = [
            "Thank you for sharing that with me. I can hear that you're going through a difficult time.",
            "It takes courage to open up about these feelings. I'm here to listen and support you.",
            "Your feelings are completely valid. Many students experience similar challenges.",
            "I want you to know that you're not alone in feeling this way.",
            "I appreciate you trusting me with your thoughts and feelings."
        ]
        
        base_response = random.choice(supportive_responses)
        
        # Add risk-specific guidance
        if risk_level == "high" or session.phq9_score >= 15 or session.gad7_score >= 15:
            guidance = "\n\n🔴 Based on your assessment results and what you've shared, I'm concerned about your wellbeing. I strongly recommend reaching out to a professional counselor. Would you like me to help you schedule an appointment?"
            urgent = True
        elif risk_level == "moderate" or session.phq9_score >= 10 or session.gad7_score >= 10:
            guidance = "\n\n🟡 Considering your assessment results, it might be helpful to speak with a counselor. In the meantime, let's work on some coping strategies together."
            urgent = False
        else:
            guidance = "\n\n🟢 Let's explore some healthy coping strategies that might help you feel better."
            urgent = False
        
        # Add personalized coping strategy
        coping_strategy = self.get_immediate_coping_strategy(message, session.phq9_score, session.gad7_score)
        
        full_response = f"{base_response}{guidance}\n\n💡 **Immediate Support:**\n{coping_strategy}\n\n💬 How does this resonate with you? What would you like to talk about?"
        
        return {
            "user_id": user_id,
            "message": full_response,
            "conversation_state": session.conversation_state.value,
            "current_risk_level": risk_level,
            "confidence": confidence,
            "requires_immediate_attention": urgent,
            "assessment_context": {
                "phq9_score": session.phq9_score,
                "gad7_score": session.gad7_score
            }
        }

    def get_immediate_coping_strategy(self, message: str, phq9_score: int, gad7_score: int) -> str:
        """Get immediate coping strategy based on message content and scores"""
        message_lower = message.lower()
        
        # Anxiety-focused strategies
        if any(word in message_lower for word in ["anxious", "worried", "panic", "nervous", "stressed"]) or gad7_score > phq9_score:
            strategies = [
                "🫁 Try the 4-7-8 breathing technique: Breathe in for 4 counts, hold for 7, exhale for 8 counts. Repeat 3-4 times.",
                "👋 Practice the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 things you can touch, 3 sounds you hear, 2 scents you smell, 1 thing you taste.",
                "🚶‍♀️ Take a brief walk outside or do gentle stretching to release physical tension.",
                "📝 Write down your worries in a 'worry journal' - sometimes getting them out of your head helps."
            ]
        
        # Depression-focused strategies  
        elif any(word in message_lower for word in ["sad", "depressed", "hopeless", "empty", "lonely"]) or phq9_score > gad7_score:
            strategies = [
                "✅ Try to do one small, achievable task - even something as simple as making your bed or having a glass of water.",
                "🤝 Reach out to one person you trust - a friend, family member, or counselor. Connection can help lift the weight you're carrying.",
                "💝 Practice self-compassion: speak to yourself as kindly as you would to a good friend going through the same situation.",
                "🌞 Consider spending 10 minutes outside or near a window. Natural light can help improve mood."
            ]
        
        # Sleep-related strategies
        elif any(word in message_lower for word in ["tired", "exhausted", "sleep", "insomnia"]):
            strategies = [
                "🌙 Establish a calming bedtime routine: dim lights, avoid screens, and try reading or gentle music 30 minutes before sleep.",
                "😌 Practice progressive muscle relaxation: tense and release each muscle group from your toes to your head.",
                "🗂️ If your mind is racing, try the 'mental filing cabinet' technique - imagine putting each worry in a file to deal with tomorrow.",
                "☕ Consider limiting caffeine after 2 PM and creating a comfortable sleep environment."
            ]
        
        # Academic stress strategies
        elif any(word in message_lower for word in ["exam", "study", "assignment", "grades", "academic"]):
            strategies = [
                "🧩 Break large tasks into smaller, manageable chunks. Focus on just the next small step.",
                "🍅 Use the Pomodoro Technique: 25 minutes focused work, 5-minute break. This makes overwhelming tasks feel manageable.",
                "📅 Create a realistic study schedule that includes breaks and self-care time.",
                "💎 Remember: your worth is not determined by your grades. You are valuable regardless of academic performance."
            ]
        
        # General strategies
        else:
            strategies = [
                "🫁 Take 5 deep breaths and remind yourself: 'This feeling is temporary and will pass.'",
                "😊 Do something small that brings you joy - listen to a favorite song, look at photos that make you smile, or call someone who cares about you.",
                "🛑 Practice the 'STOP' technique: Stop what you're doing, Take a breath, Observe how you're feeling, Proceed with kindness toward yourself.",
                "💪 Remember that seeking help is a sign of strength, not weakness. You deserve support and care."
            ]
        
        return random.choice(strategies)

    def interpret_phq9_score(self, score: int) -> Dict:
        """Interpret PHQ-9 score according to clinical guidelines"""
        if score <= 4:
            return {
                "severity": "Minimal Depression",
                "recommendation": "Continue with current self-care practices. Monitor symptoms regularly."
            }
        elif score <= 9:
            return {
                "severity": "Mild Depression", 
                "recommendation": "Consider lifestyle changes, stress management, and monitoring symptoms. Counseling may be beneficial."
            }
        elif score <= 14:
            return {
                "severity": "Moderate Depression",
                "recommendation": "Professional counseling recommended. Consider therapy and lifestyle interventions."
            }
        elif score <= 19:
            return {
                "severity": "Moderately Severe Depression",
                "recommendation": "Professional help strongly recommended. Therapy and possibly medication should be considered."
            }
        else:
            return {
                "severity": "Severe Depression",
                "recommendation": "Immediate professional intervention required. Please contact a mental health professional or emergency services."
            }

    def interpret_gad7_score(self, score: int) -> Dict:
        """Interpret GAD-7 score according to clinical guidelines"""
        if score <= 4:
            return {
                "severity": "Minimal Anxiety",
                "recommendation": "Continue with current coping strategies and self-care practices."
            }
        elif score <= 9:
            return {
                "severity": "Mild Anxiety",
                "recommendation": "Monitor symptoms and practice stress management techniques. Consider counseling if symptoms persist."
            }
        elif score <= 14:
            return {
                "severity": "Moderate Anxiety", 
                "recommendation": "Professional counseling recommended. Therapy can provide effective anxiety management strategies."
            }
        else:
            return {
                "severity": "Severe Anxiety",
                "recommendation": "Professional help strongly recommended. Immediate therapy and possibly medication should be considered."
            }

    def calculate_overall_risk(self, phq9_score: int, gad7_score: int) -> Dict:
        """Calculate overall risk level based on both assessments"""
        # Check for severe scores
        if phq9_score >= 20 or gad7_score >= 15:
            return {
                "level": "crisis",
                "actions": "Immediate professional intervention required. Contact emergency services or crisis helpline."
            }
        elif phq9_score >= 15 or gad7_score >= 10:
            return {
                "level": "high", 
                "actions": "Professional counseling strongly recommended within 1-2 weeks. Monitor closely."
            }
        elif phq9_score >= 10 or gad7_score >= 5:
            return {
                "level": "moderate",
                "actions": "Consider professional counseling. Implement coping strategies and monitor symptoms."
            }
        else:
            return {
                "level": "low",
                "actions": "Continue self-care practices. Stay connected with support systems."
            }

    def get_personalized_coping_strategies(self, phq9_score: int, gad7_score: int) -> List[str]:
        """Get personalized coping strategies based on assessment scores"""
        strategies = []
        
        # Depression-focused strategies
        if phq9_score >= 10:
            strategies.extend([
                "🏃‍♀️ Establish a daily routine, even a simple one, to provide structure and purpose",
                "🎯 Practice behavioral activation: schedule one small enjoyable or meaningful activity each day",
                "🧠 Challenge negative thoughts by asking 'Is this thought helpful? What would I tell a friend in my situation?'"
            ])
        
        # Anxiety-focused strategies  
        if gad7_score >= 10:
            strategies.extend([
                "🧘‍♀️ Practice daily mindfulness or meditation for 10-15 minutes",
                "⏰ Use the 'worry time' technique: set aside 20 minutes daily to worry, then redirect anxious thoughts",
                "💆‍♀️ Learn progressive muscle relaxation to manage physical anxiety symptoms"
            ])
        
        # General wellness strategies
        strategies.extend([
            "😴 Maintain regular sleep schedule (7-9 hours) and limit screen time before bed",
            "🏃‍♀️ Engage in regular physical activity, even just 15-20 minutes of walking daily",
            "👥 Stay connected with supportive friends and family - isolation worsens mental health",
            "🙏 Practice gratitude by writing down 3 things you're thankful for each day",
            "🚫 Limit alcohol and caffeine, which can worsen anxiety and depression symptoms"
        ])
        
        return strategies[:5]  # Return top 5 most relevant strategies

    def log_conversation(self, user_id: str, message: str, sender: str):
        """Log conversation for session tracking"""
        if user_id in self.sessions:
            self.sessions[user_id].conversation_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "sender": sender,
                "message": message
            })

    def get_session_analytics(self, user_id: str) -> Dict:
        """Get comprehensive session analytics for counselor dashboard"""
        if user_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[user_id]
        
        # Calculate session metrics
        total_messages = len(session.conversation_history)
        user_messages = [msg for msg in session.conversation_history if msg.get("sender") == "user"]
        
        # Risk progression analysis
        risk_progression = []
        for msg in user_messages:
            if msg.get("message"):
                risk, confidence = self.predict_risk_level(msg["message"])
                risk_progression.append({"timestamp": msg["timestamp"], "risk": risk, "confidence": confidence})
        
        # Assessment analysis
        phq9_domain_scores = {}
        if session.phq9_responses:
            for response in session.phq9_responses:
                domain = self.phq9_questions[response.question_id - 1]["domain"]
                phq9_domain_scores[domain] = response.response
        
        gad7_domain_scores = {}
        if session.gad7_responses:
            for response in session.gad7_responses:
                domain = self.gad7_questions[response.question_id - 1]["domain"]
                gad7_domain_scores[domain] = response.response
        
        return {
            "session_overview": {
                "user_id": f"ANON_{hash(user_id) % 10000}",
                "session_duration_hours": (datetime.datetime.now() - datetime.datetime.fromisoformat(session.session_start)).total_seconds() / 3600,
                "total_interactions": total_messages,
                "assessments_completed": len(session.phq9_responses) > 0 and len(session.gad7_responses) > 0
            },
            "clinical_assessments": {
                "phq9": {
                    "total_score": session.phq9_score,
                    "severity": self.interpret_phq9_score(session.phq9_score)["severity"],
                    "domain_scores": phq9_domain_scores,
                    "suicidal_ideation": session.phq9_responses[8].response if len(session.phq9_responses) > 8 else 0
                },
                "gad7": {
                    "total_score": session.gad7_score,
                    "severity": self.interpret_gad7_score(session.gad7_score)["severity"],
                    "domain_scores": gad7_domain_scores
                }
            },
            "risk_analysis": {
                "overall_risk": self.calculate_overall_risk(session.phq9_score, session.gad7_score),
                "crisis_indicators": any(self.detect_crisis(msg.get("message", "")) for msg in user_messages),
                "risk_progression": risk_progression
            },
            "recommendations": {
                "immediate_actions": self.calculate_overall_risk(session.phq9_score, session.gad7_score)["actions"],
                "referral_needed": session.phq9_score >= 15 or session.gad7_score >= 10 or any(resp.response > 0 for resp in session.phq9_responses[8:9]),
                "followup_timeline": "1-2 weeks" if session.phq9_score >= 10 or session.gad7_score >= 5 else "1 month"
            }
        }

    def export_clinical_report(self, user_id: str) -> str:
        """Export clinical report for healthcare providers"""
        analytics = self.get_session_analytics(user_id)
        
        if "error" in analytics:
            return json.dumps(analytics)
        
        # Create clinical report
        report = {
            "report_type": "Mental Health Screening Report",
            "generated_date": datetime.datetime.now().isoformat(),
            "patient_id": analytics["session_overview"]["user_id"],
            "screening_tools": {
                "PHQ-9": {
                    "score": f"{analytics['clinical_assessments']['phq9']['total_score']}/27",
                    "severity": analytics['clinical_assessments']['phq9']['severity'],
                    "suicidal_ideation": "Present" if analytics['clinical_assessments']['phq9']['suicidal_ideation'] > 0 else "Absent"
                },
                "GAD-7": {
                    "score": f"{analytics['clinical_assessments']['gad7']['total_score']}/21", 
                    "severity": analytics['clinical_assessments']['gad7']['severity']
                }
            },
            "clinical_impression": analytics["risk_analysis"]["overall_risk"]["level"],
            "recommendations": analytics["recommendations"],
            "crisis_risk": "HIGH" if analytics["risk_analysis"]["crisis_indicators"] else "LOW"
        }
        
        return json.dumps(report, indent=2)

    def save_session_data(self, filename: str = "session_data.json"):
        """Save all session data to file"""
        try:
            # Convert sessions to serializable format
            serializable_sessions = {}
            for user_id, session in self.sessions.items():
                serializable_sessions[user_id] = {
                    "user_id": session.user_id,
                    "name": session.name,
                    "language": session.language.value,
                    "conversation_state": session.conversation_state.value,
                    "phq9_score": session.phq9_score,
                    "gad7_score": session.gad7_score,
                    "session_start": session.session_start,
                    "conversation_history": session.conversation_history,
                    "phq9_responses": [asdict(resp) for resp in session.phq9_responses],
                    "gad7_responses": [asdict(resp) for resp in session.gad7_responses]
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_sessions, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Session data saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving session data: {e}")

# Demo and Testing Functions
def comprehensive_demo():
    """Comprehensive demonstration of the lightweight chatbot"""
    print("=" * 70)
    print("🧠 LIGHTWEIGHT MENTAL HEALTH CHATBOT WITH MANDATORY ASSESSMENTS 🧠")
    print("=" * 70)
    
    # Initialize chatbot
    print("\n🚀 Initializing chatbot...")
    chatbot = LightweightMentalHealthChatbot()
    
    # Create session
    session = chatbot.create_session("demo_user", "Alex Student", Language.ENGLISH)
    print(f"✅ Session created for user: {session.name}")
    
    # Simulate complete conversation flow
    print("\n🔄 Starting conversation simulation...")
    
    # Initial message
    print("\n" + "="*50)
    print("📝 INITIAL GREETING")
    print("="*50)
    response = chatbot.process_message("demo_user", "Hi, I'm feeling really stressed about my studies")
    print("🤖 Bot Response:")
    print(response["message"][:300] + "..." if len(response["message"]) > 300 else response["message"])
    
    # Simulate PHQ-9 responses (moderate depression)
    print("\n" + "="*50)
    print("🧠 PHQ-9 DEPRESSION ASSESSMENT")
    print("="*50)
    phq9_responses = [1, 2, 2, 2, 1, 1, 1, 0, 0]  # Score: 10 (moderate)
    
    for i, resp in enumerate(phq9_responses):
        print(f"Question {i+1}: User responds with {resp}")
        response = chatbot.process_message("demo_user", str(resp))
        if i == 8:  # Last question
            print(f"✅ PHQ-9 Complete! Moving to GAD-7...")
    
    # Simulate GAD-7 responses (mild anxiety) 
    print("\n" + "="*50)
    print("😰 GAD-7 ANXIETY ASSESSMENT")
    print("="*50)
    gad7_responses = [1, 1, 2, 1, 0, 1, 1]  # Score: 7 (mild)
    
    for i, resp in enumerate(gad7_responses):
        print(f"Question {i+1}: User responds with {resp}")
        response = chatbot.process_message("demo_user", str(resp))
        if i == 6:  # Last question
            print(f"✅ GAD-7 Complete! Assessment Results provided.")
            print(f"📊 PHQ-9 Score: {response['assessment_results']['phq9']['score']}/27")
            print(f"📊 GAD-7 Score: {response['assessment_results']['gad7']['score']}/21")
    
    # Continue with general support
    print("\n" + "="*50)
    print("💬 GENERAL SUPPORT CONVERSATION")
    print("="*50)
    support_messages = [
        "I'm really struggling with my workload and feel overwhelmed",
        "I have trouble sleeping and concentrating on studies", 
        "Thank you for the suggestions, they seem helpful"
    ]
    
    for i, msg in enumerate(support_messages, 1):
        print(f"\n👤 User Message {i}: {msg}")
        response = chatbot.process_message("demo_user", msg)
        print(f"🤖 Bot Response (Risk Level: {response.get('current_risk_level', 'N/A')}):")
        print(response['message'][:200] + "..." if len(response['message']) > 200 else response['message'])
    
    # Show session analytics
    print("\n" + "="*50)
    print("📈 SESSION ANALYTICS")
    print("="*50)
    analytics = chatbot.get_session_analytics("demo_user")
    print(f"📝 Total Interactions: {analytics['session_overview']['total_interactions']}")
    print(f"🧠 PHQ-9 Severity: {analytics['clinical_assessments']['phq9']['severity']}")
    print(f"😰 GAD-7 Severity: {analytics['clinical_assessments']['gad7']['severity']}")
    print(f"⚠️  Overall Risk: {analytics['risk_analysis']['overall_risk']['level'].upper()}")
    print(f"🚨 Crisis Detected: {analytics['risk_analysis']['crisis_indicators']}")
    
    # Export clinical report
    print("\n" + "="*50)
    print("🏥 CLINICAL REPORT")
    print("="*50)
    report = chatbot.export_clinical_report("demo_user")
    report_data = json.loads(report)
    print("📋 Clinical Report Generated:")
    print(f"🆔 Patient ID: {report_data['patient_id']}")
    print(f"🧠 PHQ-9: {report_data['screening_tools']['PHQ-9']['score']} - {report_data['screening_tools']['PHQ-9']['severity']}")
    print(f"😰 GAD-7: {report_data['screening_tools']['GAD-7']['score']} - {report_data['screening_tools']['GAD-7']['severity']}")
    print(f"🚨 Crisis Risk: {report_data['crisis_risk']}")
    print(f"💡 Referral Needed: {report_data['recommendations']['referral_needed']}")
    
    # Save session data
    print("\n" + "="*50)
    print("💾 SAVING SESSION DATA")
    print("="*50)
    chatbot.save_session_data("demo_session_data.json")
    
    print("\n🎉 Demo completed successfully!")
    return chatbot

def train_on_custom_dataset_demo():
    """Example of training on custom dataset"""
    print("\n" + "="*50)
    print("🎓 TRAINING ON CUSTOM DATASET")
    print("="*50)
    
    # Create example custom dataset
    custom_data = [
        ["text", "risk_level"],
        ["I can't handle the pressure anymore", "high"],
        ["Feeling great about my progress", "low"],
        ["I'm worried about failing my exams", "moderate"],
        ["I want to disappear forever", "crisis"],
        ["Having a good day today", "low"],
        ["I feel hopeless about everything", "high"],
        ["I'm stressed about deadlines", "moderate"],
        ["I want to hurt myself", "crisis"],
        ["Things are going well", "low"],
        ["I can't cope with college life", "high"]
    ]
    
    # Save as CSV for training
    with open("custom_mental_health_data.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(custom_data)
    
    print("📝 Created custom dataset: custom_mental_health_data.csv")
    
    # Initialize chatbot with custom dataset
    chatbot = LightweightMentalHealthChatbot("custom_mental_health_data.csv")
    
    print("✅ Chatbot trained on custom dataset!")
    print(f"📊 Training data size: {len(chatbot.training_data)}")
    
    # Test predictions
    test_messages = [
        "I'm feeling hopeless about everything",
        "I'm a bit stressed about exams", 
        "Having a wonderful day",
        "I want to end it all"
    ]
    
    print("\n🧪 Testing Risk Predictions:")
    for msg in test_messages:
        risk, confidence = chatbot.predict_risk_level(msg)
        print(f"📝 '{msg}' → Risk: {risk.upper()} (Confidence: {confidence:.2f})")
    
    return chatbot

def interactive_demo():
    """Interactive demo for real-time testing"""
    print("\n" + "="*50)
    print("🎮 INTERACTIVE CHATBOT DEMO")
    print("="*50)
    print("Type 'quit' to exit")
    
    chatbot = LightweightMentalHealthChatbot()
    
    # Get user details
    user_name = input("👤 Enter your name: ").strip()
    if not user_name:
        user_name = "Student"
    
    user_id = f"interactive_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = chatbot.create_session(user_id, user_name)
    
    print(f"\n✅ Session started for {user_name}")
    print("🤖 Chatbot: Hello! Let's begin with your mental health assessment.")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Thank you for using the mental health chatbot. Take care!")
            # Save session before exiting
            chatbot.save_session_data(f"interactive_session_{user_id}.json")
            break
        
        if not user_input:
            continue
        
        try:
            response = chatbot.process_message(user_id, user_input)
            
            print(f"\n🤖 Chatbot: {response['message']}")
            
            # Show additional info for crisis situations
            if response.get('requires_immediate_attention'):
                print("\n🚨 ALERT: This conversation indicates potential crisis situation!")
            
            # Show assessment progress
            if response.get('assessment_progress'):
                print(f"📊 Progress: {response['assessment_progress']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🤖 I'm sorry, there was an error. Please try again.")

if __name__ == "__main__":
    print("🧠 Mental Health Chatbot Demo Options:")
    print("1. Comprehensive Demo (automatic)")
    print("2. Custom Dataset Training Demo")
    print("3. Interactive Demo")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        comprehensive_demo()
    elif choice == "2":
        train_on_custom_dataset_demo()
    elif choice == "3":
        interactive_demo()
    else:
        print("Invalid choice. Running comprehensive demo...")
        comprehensive_demo()