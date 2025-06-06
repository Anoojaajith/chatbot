from flask import Flask, request, jsonify, render_template, session
import random
import psycopg2
import requests
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

file = 'data.pth'
data = torch.load(file)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Ankitha"


# Function to call car details API
def get_car_details(vin):
    api_url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValuesExtended/{vin}?format=json"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


# Function to format the car details, returning None if a detail is not found
def format_car_details(car_details):
    result = car_details['Results'][0]
    return {
        "Make": result.get('Make', None),  # None if not present
        "Model": result.get('Model', None),
        "Model Year": result.get('ModelYear', None)
       
    }

# Function to dynamically build the car details response, excluding missing data
def build_car_response(formatted_details):
    response_parts = []

    if formatted_details['Make']:
        response_parts.append(f"make: {formatted_details['Make']}")
    if formatted_details['Model']:
        response_parts.append(f"model: {formatted_details['Model']}")
    if formatted_details['Model Year']:
        response_parts.append(f"year: {formatted_details['Model Year']}")

    if response_parts:
        response = "Is your car:<br>" + "<br>".join(response_parts) + "<br>Please confirm. [Yes/No]"

    else:
        response = "Could not retrieve specific car details. Please confirm your car's make, model, and year."

    return response


# Database connection setup for different databases
def get_db_connection(db_name):
    if db_name == "postgres":
        conn = psycopg2.connect(
            host="localhost",  
            database="postgres",  
            user="postgres",  
            password="12345"
        )
    elif db_name == "parts":
        conn = psycopg2.connect(
            host="localhost",  
            database="parts",  
            user="postgres",  
            password="12345"
        )
    return conn

# Function to insert customer enquiry details into part_enquiry table
def save_part_enquiry(name, phone_number, car_make, car_model, car_year, part_name):
    conn = get_db_connection("postgres")
    cursor = conn.cursor()
    
    query = """
    INSERT INTO part_enquiry (name, phone_number, make, model, year, partname)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (name, phone_number, car_make, car_model, car_year, part_name))
    conn.commit()
    cursor.close()
    conn.close()




# Function to query part number from first database
def get_part_number(part_name):
    conn = get_db_connection("postgres")
    cursor = conn.cursor()

    query = "SELECT part_number FROM part_details WHERE part_name ILIKE %s"
    cursor.execute(query, (f"%{part_name}%",))
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    if result:
        return result[0]
    return None

#Function to get part_id using part_number from second database
def get_part_id(part_name):
    conn = get_db_connection("parts")
    cursor = conn.cursor()

    query = "SELECT part_id FROM stock_part WHERE LOWER(description) LIKE LOWER(%s)"
    cursor.execute(query, (f"%{part_name}%",))
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return result[0] if result else None

# Function to get stock and price using part_id from second database
def get_stock_and_price(part_id):
    conn = get_db_connection("parts")
    cursor = conn.cursor()

    query = "SELECT quantity, price FROM stock_supplier_parts WHERE part_id_id = %s"
    cursor.execute(query, (part_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return {"quantity": result[0], "price": result[1]} if result else None

def get_parts_by_vin(make=None, model=None, year=None):
    conn = get_db_connection("postgres")
    cursor = conn.cursor()
    
    # Base query
    query = "SELECT part_name FROM part_details WHERE 1=1"
    params = []
    
    # Add filters only if provided
    if make:
        query += " AND make = %s"
        params.append(make)
    if model:
        query += " AND model = %s"
        params.append(model)
    if year:
        query += " AND year = %s"
        params.append(year)
    
    cursor.execute(query, tuple(params))
    parts = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [{"part_name": part[0]} for part in parts] if parts else None


# List to store feedback temporarily
feedback_list = []

# Function to store feedback
def store_feedback(feedback):
    feedback_list.append(feedback)
    print(f"Collected feedback: {feedback}")


@app.route('/')
def index():
    return render_template('index.html')

# List of common greetings
greetings = ["hello", "hi", "hey", "greetings", "good day", "hei", "howdy", "helo", "heyo"]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    sentence = data['message'].lower()

    # Tokenize the sentence
    tokenized_sentence = tokenize(sentence)

    # Clear session when a greeting is detected to reset the conversation
    if any(word in tokenized_sentence for word in greetings):
        session.clear()  # Clear session to restart conversation
        session['awaiting_name'] = True
        return jsonify({"response":  "Hello! What's your name?"})

    # Name validation
    if 'awaiting_name' in session:
        name_parts = sentence.split()
        if all(part.isalpha() for part in name_parts) and 1 <= len(name_parts) <= 3:
            session['user_name'] = " ".join(part.capitalize() for part in name_parts)
            response = f"Nice to meet you, {session['user_name']}! Can you please share your phone number?"
            session.pop('awaiting_name')
            session['awaiting_phone'] = True
        else:
            response = "Please provide a valid name."
        return jsonify({"response": response})

    if 'awaiting_phone' in session:
        phone_number = ''.join([ch for ch in sentence if ch.isdigit()])
        if len(phone_number) == 10:
            session['phone_number'] = phone_number
            response = f"Thank you, {session['user_name']}! Now, can you please share your VIN number or car details?"
            session.pop('awaiting_phone')
        else:
            response = "Please provide a valid phone number."
        return jsonify({"response": response})
        
    # If user has already confirmed car details, check for yes/no response
    if 'awaiting_confirmation' in session:
        if 'yes' in tokenized_sentence:
            make, model, year = session['car_details'].values()
            available_parts = get_parts_by_vin(make, model, year)
            if available_parts:
                # Store available parts in session for confirmation follow-up
                session['available_parts'] = [part['part_name'].lower() for part in available_parts]
                response = f"Parts available for your {make} {model} {year}:<br>{available_parts[0]['part_name']}<br>Are you looking for this part? [Yes/No]"
                session['awaiting_part_confirmation'] = True  # Set flag for part confirmation follow-up
            else:
                response = "No parts are listed for your car model. Our sales team will follow up soon."
            session.pop('awaiting_confirmation')
        elif 'no' in tokenized_sentence:
            response = "Please provide your car's make, model, and year."
            session.pop('awaiting_confirmation')
        return jsonify({"response": response})

     # Check if user confirms specific part
    if 'awaiting_part_confirmation' in session:
        if 'yes' in tokenized_sentence:
            part_name = session['available_parts'][0]
            part_id = get_part_id(part_name)
            if part_id:
                stock_info = get_stock_and_price(part_id)
                if stock_info:
                    response = (f"The part '{part_name}' is available with a quantity of {stock_info['quantity']} "
                                f"and a price of ${stock_info['price']}.")
                    
                else:
                    response = f"Sorry, we couldn't retrieve stock and price information for '{part_name}'."
                    
            else:
                response = f"Sorry, we couldn't find the part '{part_name}' in our inventory."
            session.pop('awaiting_part_confirmation')
            
        elif 'no' in tokenized_sentence or part_name not in sentence:
            response = "Please specify the part name you're looking for."
            session['awaiting_specific_part_name'] = True  # Set flag to capture specific part name
            session.pop('awaiting_part_confirmation')
            
        return jsonify({"response": response})

    if 'awaiting_specific_part_name' in session:
        part_name = sentence
        save_part_enquiry(
            session.get('user_name'),
            session.get('phone_number'),
            session['car_details']['Make'],
            session['car_details']['Model'],
            session['car_details']['Model Year'],
            part_name
        )
        response = "Thank you. Our sales team will contact you as soon as possible regarding the part you requested."
        session.pop('awaiting_specific_part_name')
        return jsonify({"response": response})

    # If chatbot previously asked for make, model, and year
    if 'request_make_model' in session:
        session.pop('request_make_model')
        # Process make, model, year input and ask for spare parts requirements
        response = "Can you please share your spare parts requirements?"
        return jsonify({"response": response})

    # Handle "thank you" after part confirmation
    if 'awaiting_feedback' in session:
        if 'thank you' in tokenized_sentence or 'thanks' in tokenized_sentence:
            response = "Thank you for your feedback. Please provide a rating between 1 and 5."
            session.pop('awaiting_feedback', None)
            session['awaiting_rating'] = True  # Set flag to capture rating
            return jsonify({"response": response})

    # Handle rating input
    if 'awaiting_rating' in session:
        try:
            rating = int(sentence)
            if 1 <= rating <= 5:
                store_feedback(rating)
                response = "Thank you for your feedback!"
            else:
                response = "Please provide a rating between 1 and 5."
        except ValueError:
            response = "Please provide a valid number (1 to 5)."
        session.pop('awaiting_rating', None)
        return jsonify({"response": response})
        
    # Check if the message is a VIN number
    vin_number = next((word for word in tokenized_sentence if len(word) == 17 and word.isalnum()), None)

    if vin_number:
        car_details = get_car_details(vin_number)
        if car_details:
            formatted_details = format_car_details(car_details)
            response = build_car_response(formatted_details)  # Use dynamic response builder
            session['car_details'] = formatted_details 
            session['awaiting_confirmation'] = True  # Set flag to await yes/no confirmation
        else:
            response = "I couldn't find any car details for that VIN. Please provide your car details."
        return jsonify({"response": response})
    
    
    # Tokenize the sentence for intent classification
    x = bag_of_words(tokenized_sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    # Get model output
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])

                # Handle "get_name" intent
                if tag == 'get_name' and 'user_name' not in session:
                    name = sentence.capitalize()
                    if name.isalpha():
                        session['user_name'] = name
                        response = f"Nice to meet you, {session['user_name']}! Can you please share your VIN number or car details?"


                # Handle VIN retrieval and car details
                elif tag == 'get_vin_details':
                     vin_number = next((word for word in tokenized_sentence if len(word) == 17), None)
                     
                     if vin_number:
                        car_details = get_car_details(vin_number)
                        if car_details:
                            formatted_details = format_car_details(car_details)
                            response = build_car_response(formatted_details)  # Build response dynamically
                            session['car_details'] = formatted_details 
                            session['awaiting_confirmation'] = True  # Await yes/no confirmation
                        else:
                            response = "I couldn't find any car details for that VIN."
                     else:
                        response = "I didn't catch your VIN number. Please provide a valid 17-character VIN."
                    
                elif tag == 'part_inquiry':
                    part_name = ' '.join(tokenized_sentence[-2:])  # Capture multi-word part names
                    
                    if 'car_details' in session:
                        car_make = session['car_details']['Make']
                        car_model = session['car_details']['Model']
                        car_year = session['car_details']['Model Year']

                       # Fetch part number from the first database
                        part_number = get_part_number(part_name)
                        
                        if part_number:
                            # Fetch part_id, quantity, and price from the second database
                            part_id = get_part_id(part_number)
                            if part_id:
                                stock_info = get_stock_and_price(part_id)
                                if stock_info:
                                    response = (f"The part '{part_name}' for your car has a quantity of {stock_info['quantity']} "
                                                f"and the price is ${stock_info['price']}.")
                                else:
                                    response = f"Sorry, I couldn't find stock and price information for '{part_name}'."
                            else:
                                response = f"Sorry, I couldn't find the part ID for '{part_name}'."
                        else:
                            response = f"Sorry, I couldn't find the part number for '{part_name}'."
                    else:
                        response = "Please provide your car details first."  
                        
                # Clear session on goodbye
                if tag == "goodbye":
                    session.clear()
                    session['awaiting_feedback'] = True  # Set flag to wait for feedback
                    response = f"{response} Before you go, could you rate your experience from 1 to 5?"

                return jsonify({"response": response})
    else:
        return jsonify({"response": "I do not understand..."})

if __name__ == "__main__":
    app.run(debug=True)