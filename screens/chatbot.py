import streamlit as st
from datetime import datetime
import numpy as np
from PIL import Image
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time
import re

# Simulate GPT-style response
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.04)

# Try to fetch Wikipedia summary
def get_wikipedia_summary(term):
    try:
        url = f"https://en.wikipedia.org/wiki/{term.replace(' ', '_')}"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        for p in soup.select("p"):
            text = p.get_text().strip()
            if len(text) > 50 and "refer to:" not in text.lower():
                return text
        return f"I couldn’t find a good summary, but you can [read more on Wikipedia]({url})"
    except Exception:
        return f"⚠️ Something went wrong. Try [Wikipedia]({url}) directly."

# Chatbot Screen
# This screen provides a chatbot interface for users to interact with the image classification and object detection models.
def show(image_classifier, yolo_model):
    st.header("🤖 Chatbot – Smart Image Insights")

    def get_formatted_timestamp():
        return datetime.now().strftime("%I:%M %p")

    # Session state setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_objects" not in st.session_state:
        st.session_state.last_objects = []
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "timestamp" in msg:
                st.caption(msg["timestamp"])
            if "image" in msg:
                st.image(msg["image"], caption=msg.get("caption", ""), use_container_width=True)

    st.divider()

    # Upload + Mode
    with st.form("image-form", clear_on_submit=True):
        file = st.file_uploader("Upload an image to classify or detect...", type=["jpg", "jpeg", "png"])
        view_mode = st.radio("Choose Analysis Mode", ["Main Object Prediction", "Detailed Object Detection"])
        submitted = st.form_submit_button("Submit")

    if submitted and file:
        img = Image.open(file).convert("RGB")
        np_img = np.array(img)
        timestamp = get_formatted_timestamp()

        st.session_state.messages.append({
            "role": "user", "content": f"Uploaded an image for **{view_mode}**.",
            "image": file, "caption": "Uploaded Image", "timestamp": timestamp
        })

        with st.chat_message("user"):
            st.image(file, caption="Uploaded Image", use_container_width=True)
            st.caption(timestamp)

        with st.chat_message("assistant"):
            st.markdown(f"🔎 Running **{view_mode}**...")
            st.caption(timestamp)

            # Main Object Prediction
            if view_mode == "Main Object Prediction":
                st.session_state.last_mode = "classification"
                st.session_state.last_objects = None  # clear detection context
                
                threshold = 0.5
                st.caption(f"🔧 Using default confidence threshold: `{threshold}`")
                preds = image_classifier.classify(img)
                preds = preds[preds["Pred_Prob"] >= threshold]

                if preds.empty:
                    response = "❌ No predictions above threshold. Try a clearer image."
                    st.warning(response)
                    st.session_state.messages.append({
                        "role": "assistant", "content": response, "timestamp": get_formatted_timestamp()
                    })
                    return

                # Displaying the results
                top_class = preds.iloc[0]
                confidence = top_class["Pred_Prob"]
                st.session_state.last_prediction = top_class["Class"]

                st.success(f"✅ **Prediction:** {top_class['Class']} ({confidence*100:.1f}%)")
                st.image(file, caption=f"Predicted: {top_class['Class']}", use_container_width=True)

                if confidence < 0.3:
                    st.warning("⚠️ Prediction has low confidence. Try re-uploading.")

                # Displaying the Wikipedia link
                wiki_link = f"https://en.wikipedia.org/wiki/{top_class['Class'].replace(' ', '_')}"
                st.info(f"🧠 Want to know more? [Learn about {top_class['Class']}]({wiki_link})")

                # Displaying the prediction summary
                st.markdown("### 🔬 Prediction Summary")
                st.markdown(f"**Top Class:** `{top_class['Class']}`  \n**Confidence:** `{confidence*100:.1f}%`")

                fig = px.bar(preds.sort_values("Pred_Prob", ascending=True),
                             x='Pred_Prob', y='Class',
                             orientation='h', color='Pred_Prob',
                             color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

                st.caption(f"💬 _Try asking_: **'What is a {top_class['Class'].lower()}'** or **'Tell me about it.'**")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Predicted **{top_class['Class']}** with confidence **{confidence:.2f}**.",
                    "image": file,
                    "caption": f"Predicted: {top_class['Class']}",
                    "timestamp": timestamp
                })

            # Detailed Object Detection
            elif view_mode == "Detailed Object Detection":
                st.session_state.last_mode = "detection"
                st.session_state.last_prediction = None  # clear classification context

                results = yolo_model(np_img)
                boxes = results[0].boxes
                classes = results[0].names
                annotated_img = results[0].plot()


                # Check if any objects are detected
                if boxes is None or len(boxes) == 0:
                    response = "⚠️ No objects detected."
                    st.warning(response)
                    st.session_state.messages.append({
                        "role": "assistant", "content": response, "timestamp": get_formatted_timestamp()
                    })
                    return

                # Filter boxes based on confidence and class
                class_counts = {}
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = classes[cls_id]
                    class_counts[label] = class_counts.get(label, 0) + 1

                st.session_state.last_objects = list(class_counts.keys())

                st.image(annotated_img, caption="Detected Objects", use_container_width=True)
                
                # Displaying the detected objects
                summary_lines = [f"- {count} **{label}(s)**" for label, count in class_counts.items()]
                summary = "### 🧾 Detected Objects Summary:\n" + "\n".join(summary_lines)
                st.markdown(summary)

                if len(class_counts) == 1:
                    only_label = list(class_counts.keys())[0].lower()
                    st.caption(f"💬 _Ask about it_: **'What is a {only_label}?'** or **'Tell me more.'**")
                else:
                    example_labels = list(class_counts.keys())[:2]  # take any 2 examples
                    suggestions = ", ".join(f"'What is a {label.lower()}?'" for label in example_labels)
                    st.caption(f"💬 _Try asking_: {suggestions}")


                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "image": annotated_img,
                    "caption": "Detected Objects",
                    "timestamp": get_formatted_timestamp()
                })

    # Follow-up input
    if prompt := st.chat_input("Ask a follow-up about the prediction or detection..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            cleaned_prompt = re.sub(r"[^\w\s]", "", prompt.lower())
            keywords = cleaned_prompt.split()
            mode = st.session_state.get("last_mode")
            response = ""
            found_match = None

            if mode == "detection" and st.session_state.get("last_objects"):
                last_objects = st.session_state["last_objects"]
                for obj in last_objects:
                    for word in keywords:
                        if word == obj.lower():
                            found_match = obj
                            break
                    if found_match:
                        break

                if not found_match and len(last_objects) == 1 and any(w in keywords for w in ["this", "it", "object", "thing"]):
                    found_match = last_objects[0]

                if found_match:
                    response = get_wikipedia_summary(found_match)
                else:
                    detected_list = ", ".join(f"`{obj}`" for obj in last_objects)
                    response = (
                        "❌ I didn’t detect that in the image. "
                        f"Try asking about one of these: {detected_list}"
                    )

            elif mode == "classification" and st.session_state.get("last_prediction"):
                prediction = st.session_state["last_prediction"].lower()
                match_found = any(word == prediction for word in keywords)

                if match_found or any(w in keywords for w in ["this", "it", "object", "thing"]):
                    response = get_wikipedia_summary(prediction)
                else:
                    response = (
                        f"❌ I only classified this image as `{prediction}`. "
                        "Try asking about that or re-upload a different image."
                    )

            else:
                response = "Please upload and analyze an image first."

            stream = stream_response(response)
            streamed = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": streamed})

