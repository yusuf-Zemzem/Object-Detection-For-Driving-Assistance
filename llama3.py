# LLama3.py
from groq import Groq  # Ensure you have the Groq Python client installed

def generate_warning(detected_obj_names_string):
    """
    Generate a warning alert using Groq's API based on detected objects.

    Args:
        detected_obj_names_string (str): A string of detected object names.
        api_key (str): The API key for Groq client.

    Returns:
        str: The generated response or an error message.
    """
    client = Groq(api_key="Your APi Key"
)

    prompt = (
        f"Generate a warning alert for the detected objects: {detected_obj_names_string}. "
        "Format the response as follows, starting directly from the **Caution:** section:\n\n"
        "Caution: <Message about the detected object and road condition>\n\n"
        "For your safety follow this Actions:\n\n"
        " Step 1 .\n"
        " Step 2 .\n"
        " Step 3 .\n"
        "Do not include any introductory phrases like 'Here is...' or 'Below is...'."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"
