from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up the Flask app
app= Flask(__name__)
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
        max_tokens = data.get('maxTokens', 50)  # Default to 50
        temperature = data.get('temperature', 0.5)  # Default to 0.5

        # Validate API key
        if not api_key:
            logging.error("API key is missing.")
            return jsonify({"error": "API key is required."}), 400

        # Validate input data
        item_data = data.get('inputData', [])
        category_list = data.get('categories', [])
        instructions = data.get('instructions', """Your at a data scientist data assistant and your job is to sort [Item] into one of the following categories: [Categories]

                The additional context has been provided to help complete the task.
                "We need to sort each state into one of the following regional categories"
                You are only to return one of the categories and no other response. Please provide your best guess when there is no certain choice.""")

        if not item_data or all(not item for sublist in item_data for item in sublist):
            logging.error("Input data is empty.")
            return jsonify({"error": "Input data cannot be empty."}), 400

        if not category_list or all(not category for category in category_list):
            logging.error("Categories are empty.")
            return jsonify({"error": "Categories cannot be empty."}), 400

        response_data = []

        for item in item_data:
            if isinstance(item, list) and item:
                item = item[0]
            elif not item:
                continue

            category_string = ", ".join(category_list)
            filled_prompt = f"{instructions}\nItem: {item}\nCategories: {category_string}"

            try:
                # Use the OpenAI ChatCompletion API with logprobs enabled
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": filled_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=True  # Enable logprobs for probability calculation
                )

                # Log the raw response for debugging
                logging.info(f"Raw Response: {response}")

                response_text = response.choices[0].message.content
                logprobs_content = response.choices[0].logprobs

                # Calculate the average log probability if logprobs are available
                probability = None
                if logprobs_content and hasattr(logprobs_content, "content"):
                    # Extract token logprobs from the logprobs content
                    token_logprobs = [
                        token_logprob.logprob for token_logprob in logprobs_content.content
                    ]
                    if token_logprobs:
                        # Average log probability
                        average_logprob = sum(token_logprobs) / len(token_logprobs)
                        probability = np.exp(average_logprob)  # Convert log probability to standard probability
                    else:
                        logging.warning("Token logprobs are empty.")
                else:
                    logging.warning("Logprobs content structure unexpected or missing.")

                # Extract category from the response text
                extracted_category = None
                for category in category_list:
                    if category.lower() in response_text.lower():
                        extracted_category = category
                        break

                response_data.append({
                    "item": item,
                    "probability": probability,
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
            "results": [
                {"item": entry["item"], "probability": entry["probability"], "category": entry["category"]}
                for entry in response_data
            ]
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(debug=True)
