import streamlit as st
import plotly.express as px
from PIL import Image

# Image Classification Screen
# This screen allows users to upload an image and get classification predictions.
def show(image_classifier):
    st.header("📷 Image Classification")

    st.write("Upload an image to classify it using our deep learning model.")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    if st.button("Run Classification"):
        if uploaded_file is None:
            st.error("Please upload an image first.")
            return

        ## Displaying the uploaded image
        with st.spinner("Running classification..."):
            img = Image.open(uploaded_file).convert("RGB")
            preds = image_classifier.classify(img)
            preds = preds[preds["Pred_Prob"] >= threshold]

        if preds.empty:
            st.warning("❌ No predictions above threshold. Try a clearer image or lower the threshold.")
            return

        top_class = preds.iloc[0]
        confidence = top_class["Pred_Prob"]

        # Displaying the results
        st.success(f"✅ **Prediction:** {top_class['Class']} ({confidence*100:.1f}%)")
        st.image(img, caption=f"Predicted: {top_class['Class']}", use_container_width=True)

        #Low confidence warning
        if confidence < 0.3:
            st.warning("⚠️ Low confidence. Consider using a clearer image.")

        #Displaying the Wikipedia link
        wiki_link = f"https://en.wikipedia.org/wiki/{top_class['Class'].replace(' ', '_')}"
        st.markdown(f"[🧠 Learn more about {top_class['Class']} on Wikipedia]({wiki_link})")

        # Displaying the prediction summary
        st.markdown("### 🔬 Prediction Summary")
        st.markdown(f"**Top Class:** `{top_class['Class']}`  \n**Confidence:** `{confidence*100:.1f}%`")

        # Displaying the prediction probabilities for all classes
        fig = px.bar(preds.sort_values("Pred_Prob", ascending=True),
                     x='Pred_Prob', y='Class',
                     orientation='h', color='Pred_Prob',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        # Download button for predictions
        csv = preds.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, file_name='classification_predictions.csv')
