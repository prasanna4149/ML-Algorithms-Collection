# Import necessary libraries
import streamlit as st
import joblib
from PIL import Image
from spam_emails import spam_emails  # List of spam emails
from non_spam_emails import non_spam_emails  # List of non-spam emails


def classify_email(test_input):
    # Transform the input text into features using the TF-IDF vectorizer
    input_data_features = tfidf_vectorizer.transform([test_input])

    # Use the trained model to predict if the email is spam (1) or not (0)
    prediction = model.predict(input_data_features)

    # Return the prediction result (spam = 1, not spam = 0)
    return prediction[0]


# Navigation Bar for the Streamlit app
nav_selection = st.sidebar.radio("Navigation", ["Home", "Spam Email List", "Not Spam Email List"])

# Display the list of spam emails when "Spam Email List" is selected
if nav_selection == "Spam Email List":
    st.header("Spam Email List")  # Set the header for this section
    for email in spam_emails:  # Loop through each spam email
        st.code(email)  # Display each email as code

# Display the list of non-spam emails when "Not Spam Email List" is selected
elif nav_selection == "Not Spam Email List":
    st.header("Not Spam Email List")  # Set the header for this section
    for email in non_spam_emails:  # Loop through each non-spam email
        st.code(email)  # Display each email as code

# Home page: Email classification tool
else:
    # Display the header and description for the classification tool
    st.write('<h3>Email Spam Classification <span style="color:#EE7214;">(Logistic Regression)</span></h3>',
             unsafe_allow_html=True)
    st.caption(
        'Empower Your Inbox: Effortlessly Distinguish Spam from Legitimate Emails with our Email Spam Classification Tool')

    # Load the pre-trained Logistic Regression model
    model = joblib.load('email_model.pkl')

    # Load the TF-IDF vectorizer used during model training
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Create a text area for the user to input email content
    input_text = st.text_area("Enter an email text to classify:", max_chars=500)

    # Button to trigger classification when clicked
    if st.button("Classify"):
        if input_text:  # Check if the user has entered text
            # Classify the email using the classify_email function
            result = classify_email(input_text)

            # If the email is predicted as spam
            if result == 1:
                # Display an alert image for spam
                background = Image.open("assets/alert.png")
                col1, col2, col3 = st.columns(3)  # Create three columns for layout
                col2.image(background, use_column_width=True, width=10)  # Show the image in the middle column
                # Display the spam prediction result with red text
                st.write("Prediction:", "<b style='color:#FC4100;'>Spam Email</b>", unsafe_allow_html=True)

            # If the email is predicted as non-spam
            else:
                # Display a confirmation image for non-spam
                background = Image.open("assets/ok.png")
                col1, col2, col3 = st.columns(3)  # Create three columns for layout
                col2.image(background, use_column_width=True, width=10)  # Show the image in the middle column
                # Display the non-spam prediction result with green text
                st.write("Prediction:", "<b style='color:#65B741;'>Non-Spam Email</b>", unsafe_allow_html=True)

        # If no text is entered, show a warning message
        else:
            st.warning("Please enter some text to classify.")
