from puppeteer import MessageObservation
import transformers
#have to "import transformers" on the previous line otherwise the next line will raise segmentation fault error.
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity_score(premises, observations):
	max_score = 0
	for observation in observations:
		if isinstance(observation, MessageObservation):
			msg = observation.text
			msg_embedding = model.encode(msg, convert_to_tensor=True)
			for pm in premises:
				pm_embedding = model.encode(pm, convert_to_tensor=True)
				cosine_score = util.cos_sim(pm_embedding, msg_embedding).item()
				if cosine_score > max_score:
					max_score = cosine_score
	return max_score
