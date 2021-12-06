import streamlit as st
from char_cnn_model import CharCNN, preprocess_text
import torch
import warnings

def initialize_modules():
    warnings.filterwarnings("ignore")
    
    model = CharCNN(1014, 70, 2, net_type="small")
    model.load_state_dict(torch.load("./char-cnn-model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    with open("./default_options.txt", "r") as options_file:
        options = [(opt.split("-:-")[0], opt.split("-:-")[1][:-1]) for opt in options_file.readlines()]
    
    return model, options

def init_title():
    st.title("Character Level Convolutional Networks for Text Classification - Live Demo")
    
def init_inputs(options):
    default_select = st.selectbox("Reset to a default sentence", options, index=0, format_func=lambda opt_title: opt_title[0])

    sent = st.text_area("Enter a review of any imaginary product and find out if the review is positive or negative.", default_select[1], height=120, max_chars=1014)
    btn = st.button("Predict", key="enter", help="Click to run the model over your input.")
    return sent, btn, default_select

def init_outputs(sent, btn, model):
    st.write("How much the model thinks that this is a positively written review.")
    positive_sentiment = st.progress(0)
    st.write("This is how much the model considers the review to be negatively phrased.")
    negative_sentiment = st.progress(0)
    
    if btn:
        negative_sent, positive_sent = torch.nn.functional.softmax(model(preprocess_text(sent).unsqueeze(0)))[0]
        
        positive_sent = positive_sent.item()
        negative_sent = negative_sent.item()
        
        positive_sentiment.progress(positive_sent)
        negative_sentiment.progress(negative_sent)
        
        if positive_sent > negative_sent:
            st.success(f"The review is a positive one with {positive_sent * 100:.0f}% confidence")
        else:
            st.error(f"The review is a negative one with {negative_sent * 100:.0f}% confidence")

def initialize_structure(model, options):
    with st.container():
        init_title()
    with st.container():
        sent, btn, default_select = init_inputs(options)
    with st.container():
        init_outputs(sent, btn, model)

if __name__ == '__main__':
    model, options = initialize_modules()
    
    initialize_structure(model, options)
    
    
