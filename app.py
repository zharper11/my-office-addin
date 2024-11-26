from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pandas as pd
import numpy as np
import tiktoken
import json
import math
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up Flask app
app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        logging.info("Analysis request received")
        data = request.get_json()

        # Extract API parameters from the request payload
        api_key = data.get('apiKey')
        client = OpenAI(api_key=api_key)
        model = data.get('model', 'gpt-4o-mini')  # Default to 'gpt-4o-mini'
        instructions = data.get('instructions', """Your job is to sort [Item] into one of the following categories: [Categories]
                
                The additional context has been provided to help complete the task.
                "We need to sort each state into one of the following regional categories"
                You are only to return one of the categories and no other response. Please provide your best guess when there is no certain choice.""")

        # Validate API key
        if not api_key:
            logging.error("API key is missing.")
            return jsonify({"error": "API key is required."}), 400

        # Validate input data
        item_data = data.get('inputData', [])
        category_list = data.get('categories', [])

        if not item_data or all(not item for sublist in item_data for item in sublist):
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        if not category_list or all(not category for category in category_list):
            logging.error("Categories are empty.")
            return jsonify({"error": "Categories cannot be empty."}), 400

        # Initialize empty response data
        response_data = []

        # Token weighting dictionary for logit bias
        tokens_list = {}
        encoding = tiktoken.encoding_for_model(model)
        weighting = 5  # Define the weighting value

        # Create logit bias dictionary based on categories
        for category in category_list:
            token_ids = encoding.encode(category)
            for token_id in token_ids:
                tokens_list[token_id] = weighting

        # Loop through the input data
        for item in item_data:
            if isinstance(item, list) and item:
                item = item[0]
            elif not item:
                continue

            # Prepare prompt using first_run function
            filled_prompt = first_run(item, category_list, instructions)

            try:
                # Use the OpenAI ChatCompletion API
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": filled_prompt}
                    ],
                    logprobs=True,  # Enable logprobs for probability calculation
                    max_tokens=50,
                    temperature=0.5,
                    logit_bias=tokens_list  # Apply token weighting
                )

                # Extract response details
                response_text = response.choices[0].message.content
                logprobs_content = response.choices[0].logprobs

                # Calculate confidence using existing method if logprobs available
                confidence_score = None
                if logprobs_content:
                    confidence_score = calculate_confidence(logprobs_content, category_list, response_text)

                # Extract category from the response text
                extracted_category = None
                for category in category_list:
                    if category.lower() in response_text.lower():
                        extracted_category = category
                        break

                # Append the result
                response_data.append({
                    "item": item,
                    "probability": confidence_score,
                    "category": extracted_category or "Unknown"
                })

            except Exception as e:
                logging.error(f"Error during OpenAI API call for item '{item}': {e}")
                response_data.append({
                    "item": item,
                    "probability": None,
                    "category": "Error"
                })

        logging.info("Analysis complete")
        return jsonify({
            "results": response_data
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

# Function to calculate confidence score
def calculate_confidence(logprobs_content, category_list, response_text):
    # Initialize arrays to store summed probabilities for each category per position
    category_sums = {
        'Selected category': [],
        'Not-selected category': [],
        'Model deviation': [],
        'Selected category- Incorrect tokens': []
    }
    # Find the matching category row index directly
    response_text_lower = response_text.lower()

    for item in logprobs_content:
        # Store the probabilities for each category at each token position
        token_probs = {key: 0.0 for key in category_sums}

        for top_logprob in item.top_logprobs:
            token_lower = top_logprob.token.lower()

            probability = math.exp(top_logprob.logprob)

            if token_lower in response_text_lower:
                token_probs['Selected category'] += probability
            else:
                token_probs['Not-selected category'] += probability

        # Append the summed probabilities for each token position across all categories
        for category, prob in token_probs.items():
            category_sums[category].append(prob)

    # Ensure all probability sum lists are the same length by padding with zeros
    max_length = len(category_sums['Selected category'])
    for category in category_sums:
        category_sums[category] += [0.0] * (max_length - len(category_sums[category]))

    # Create a summary DataFrame for the total probabilities at each position
    summary_df = pd.DataFrame({
        'Category': list(category_sums.keys()),
        **{f'Position {i+1}': [category_sums[category][i] for category in category_sums] for i in range(max_length)}
    })

    # Calculate weighting for Model Deviation
    total_model_deviation = 0
    for i in range(max_length):
        total_model_deviation += (1 - total_model_deviation) * (summary_df.at[summary_df[summary_df['Category'] == 'Model deviation'].index[0], f'Position {i + 1}'])

    # Calculate entropy probabilities using log approach to avoid numerical underflow
    entropy_probs = []
    for i in range(max_length + 1):
        if i < max_length:
            # Start with log of Secondary Prediction at current position
            log_probability = math.log(summary_df.at[summary_df[summary_df['Category'] == 'Not-selected category'].index[0], f'Position {i + 1}'] + 1e-10)
            # Add log of all previous Primary Predictions, avoiding zero propagation
            for j in range(i):
                primary_prob = summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}']
                log_probability += math.log(primary_prob + 1e-10)
        else:
            # For the final step, just use the log product of all Primary Predictions
            log_probability = sum([math.log(summary_df.at[summary_df[summary_df['Category'] == 'Selected category'].index[0], f'Position {j + 1}'] + 1e-10) for j in range(max_length)])
        entropy_probs.append(math.exp(log_probability))

    # Normalize the entropy probabilities to ensure they sum to 1
    total_prob_sum = sum(entropy_probs)
    normalized_entropy_probs = [p / total_prob_sum for p in entropy_probs] if total_prob_sum > 0 else [1 / len(entropy_probs)] * len(entropy_probs)

    # Calculate entropy
    entropy = -sum([p * math.log2(p) for p in normalized_entropy_probs if p > 0])

    # Calculate maximum entropy (log2 of number of combinations)
    max_entropy = math.log2(max_length + 1)

    # Calculate total confidence based on entropy
    total_confidence = (1 - total_model_deviation) * (1 - entropy / max_entropy) if max_entropy > 0 else (1 - total_model_deviation)
    
    return total_confidence

# Function to create the filled prompt
def first_run(item, category_list, instructions):
    category_string = ", ".join(category_list)
    filled_prompt = instructions.replace("[Item]", item).replace("[Categories]", category_string)
    return filled_prompt

# Function to create retrain prompt
def retrain_run(item, response_text, confidence, category_list):
    retrain_prompt = "Retrain Prompt: Please adjust the categorization based on previous response and confidence."
    filled_prompt = retrain_prompt.replace("[Item]", item).replace("[Response]", response_text).replace("[Confidence]", str(confidence)).replace("[Categories]", ", ".join(category_list))
    return filled_prompt

if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(debug=True)
