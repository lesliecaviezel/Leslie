# Program title: Storytelling App

# import part
import streamlit as st
from transformers import pipeline

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(caption):
    pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

    prompt = (
        "Write a child-friendly story for kids aged 3-10.\n"
        "Rules:\n"
        "- 50 to 100 words\n"
        "- Use short and simple sentences\n"
        "- Keep a warm, positive tone\n"
        "- No violence, horror, or scary elements\n"
        "- End with a gentle lesson about kindness, sharing, or courage\n\n"
        f"Image description: {caption}\n"
        "Story:"
    )

    out = pipe(
        prompt,
        max_new_tokens=130,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )[0]["generated_text"]

    # 某些模型会把 prompt 一起返回，做一次清理
    story = out.replace(prompt, "").strip()
    return story

# text2audio
def text2audio(story_text):
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = pipe(story_text)
    return audio_data


def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


        #Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(scenario)

        #Stage 2: Text to Story
        st.text('Generating a story...')
        story = text2story(scenario)
        st.write(story)

        #Stage 3: Story to Audio data
        st.text('Generating audio data...')
        audio_data =text2audio(story)

        # Play button
        if st.button("Play Audio"):
            # Get the audio array and sample rate
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]

            # Play audio directly using Streamlit
            st.audio(audio_array,
                     sample_rate=sample_rate)


if __name__ == "__main__":
    main()
