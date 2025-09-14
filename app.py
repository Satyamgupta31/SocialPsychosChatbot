from flask import Flask, request, jsonify
from chatbot import LightweightMentalHealthChatbot, Language

app = Flask(__name__)
chatbot = LightweightMentalHealthChatbot()

@app.route("/start", methods=["POST"])
def start_session():
    data = request.get_json()
    user_id = data.get("user_id")
    name = data.get("name", "User")
    lang = data.get("language", "en")

    session = chatbot.create_session(user_id, name, Language(lang))
    return jsonify({"message": f"Session started for {name}", "user_id": user_id})

@app.route("/message", methods=["POST"])
def process_message():
    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message")

    response = chatbot.process_message(user_id, message)
    return jsonify(response)

@app.route("/analytics/<user_id>", methods=["GET"])
def get_analytics(user_id):
    analytics = chatbot.get_session_analytics(user_id)
    return jsonify(analytics)

@app.route("/report/<user_id>", methods=["GET"])
def get_report(user_id):
    report = chatbot.export_clinical_report(user_id)
    return jsonify(report)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
