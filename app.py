# Core Packages
import streamlit as st
import helper
from load_css import local_css

local_css("css/style.css")

# NLP Packages
import numpy as np
import csv
import spacy_streamlit
import spacy
nlp = spacy.load('en_core_web_sm')

def main():
	st.markdown(title_temp, unsafe_allow_html=True)

	menu = ['Detection', 'Counselling', 'Are you Mentally Ill?']
	choice = st.sidebar.selectbox('Menu', menu)

	if choice == 'Detection':
		st.subheader('Detecting Offensive Words')
		user_input = st.text_input("Message: ")
		words = user_input.split()
		clean = []
		foul = []
		for word in words:
			word = word.lower()
			temp = helper.calc_thresold(word)
			if (temp['bad']>temp['good']) and temp['bad']>0.65:
				foul.append([word, temp['bad']])
			else:
				clean.append(word)

		if st.button('Send Message'):
			t = "<div>"
			for word in foul:
				t += "<span class='highlight red'>{}<span class='bold'>{}</span></span>  ".format(str(word[0]), str(word[1]))
			t+="</div>"
			st.markdown(t, unsafe_allow_html=True)
			st.text_area("Cleaned Message:", value=" ".join(clean), height=100, max_chars=None, key=None)
		else:
			st.text_area("Cleaned Message:", value='Type message and start the detection by clicking the "Send Message" button.', height=100, max_chars=None, key=None)
		
	elif choice == 'Counselling':
		st.subheader('Counselling on Mental Health')
		user_input = st.text_input("Question: ")
		if len(user_input)>0 and user_input[-1] == '?':
			user_input = user_input[:-1]

		if st.button("Submit"):
			st.text_area("Answer:", value=helper.match_answer(user_input), height=300, max_chars=None, key=None)
			audio_file = open('audio.mp3', 'rb')
			audio_bytes = audio_file.read()
			st.audio(audio_bytes, format='audio/ogg')
		else:
			st.text_area("Answer:", value="Ask any question related to mental health.", height=300, max_chars=None, key=None)


	elif choice == 'Are you Mentally Ill?':
		st.subheader('Check if you are Mentally ill')
		user_input = st.text_input("Tell us how your life is going: ")

		if st.button("Check"):
			result = np.argmax(helper.get_prediction(user_input))
			if result==1:
				st.warning("Mentally ill")
				st.warning("LIFE ISN'T AS SERIOUS AS YOUR MIND MAKES IT OUT TO BE.")
			else:
				st.success("not Mentally ill")
				st.success("YAY, YOUR LIFE SOUNDS FUN.")


title_temp = """
<link href="//maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>

<div class="border-top my-3"></div>
<div class="row featurette">
	<div class="col-md-7 order-md-2">
		<h2 class="featurette-heading">सच्ची <span class="text-muted">Awaaz.</span></h2>
		<p class="lead">Learn to STOP Abusing!!</p>
		<p class="lead">- Kshitij Mohan</p>
	</div>
	<div class="col-md-5 order-md-1">
		<img src="https://daily.jstor.org/wp-content/uploads/2020/01/the_theory_of_cuss_word_relativity_1050x700.jpg" class="img-fluid logo" alt="logo">
	</div>
</div>
<div class="border-top my-3"></div>
"""

if __name__ == '__main__':
	main()