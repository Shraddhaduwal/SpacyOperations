import spacy

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from utils.util import timeit
from utils.csv_writer import create_csv, create_csv_list, create_csv_dictionary

@timeit
def clean_text(text):
	"""Words without stop_words, spaces, punctuations"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "textcat", "..."])
	nlp.max_length = 10000000
	doc = nlp(text)
	text_without_stop_words = [words for words in doc if words.is_stop==False if words.is_punct==False\
							   if words.is_space==False]
	print(text_without_stop_words)
	return text_without_stop_words

@timeit
def total_nouns(text):
	"""Noun words, total no of nouns, nouns frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat", "..."])
	nlp.max_length = 3000000
	doc = nlp(text)
	nouns = [token.string for token in doc if token.pos_=="NOUN"]
	nouns_frequency = {}
	for noun in nouns:
		if noun in nouns_frequency:
			nouns_frequency[noun] += 1
		else:
			nouns_frequency[noun] = 1

	list_of_tuples = sorted(nouns_frequency.items(), reverse=True, key=lambda x:x[1])
	top_ten_noun = list_of_tuples[:10]

	# print(max(nouns_frequency, key=nouns_frequency.get))
	return len(nouns), nouns, nouns_frequency, top_ten_noun

@timeit
def total_adjectives(text):
	"""Adjectives, total no of adjectives, adjective frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	doc = nlp(text)
	sents = doc.sents
	sentences_with_adj = []
	adj_count = {}

	for sent in sents:
		for token in sent:
			if token.pos_=="ADJ":
				sentences_with_adj.append(sent)
				break


	for sent in sentences_with_adj:
		for token in sent:
			if token.pos_=="ADJ":
				if token in adj_count:
					adj_count[token] += 1
				else:
					adj_count[token] = 1

	print(adj_count)




	# print(sentences_with_adj)
	# adjectives = [token for token in doc if token.pos_=="ADJ"]
	# adjective_frequency = {}
	# for adj in adjectives:
	# 	if adj in adjective_frequency:
	# 		adjective_frequency[adj] += 1
	# 	else:
	# 		adjective_frequency[adj] = 1
	#
	# print(max(adjective_frequency, key=adjective_frequency.get))
	#
	# list_of_tuples = sorted(adjective_frequency.items(), reverse=True, key=lambda x:x[1])
	# top_ten_adjectives = list_of_tuples[:10]

	# return len(adjectives), adjectives, adjective_frequency, top_ten_adjectives

@timeit
def total_verbs(text):
	"""Verbs, total no of verbs, verb frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat", "..."])
	doc = nlp(text)
	verbs = [token for token in doc if token.pos_=="VERB"]
	verb_frequency = {}
	for verb in verbs:
		if verb in verb_frequency:
			verb_frequency[verb] += 1
		else:
			verb_frequency[verb] = 1

	list_of_tuples = sorted(verb_frequency.items(), reverse=True, key=lambda x:x[1])
	top_ten_verbs = list_of_tuples[:10]

	return len(verbs), verbs, verb_frequency, top_ten_verbs

@timeit
def noun_noun_phrase(text):
	"""Noun-Noun phrases and their frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "..."])
	doc = nlp(text)
	noun_and_noun_phrase = [chunk for chunk in doc.noun_chunks] # noun_chunks is the chunk consists of noun-noun phrase

	noun_phrase_frequency = {}
	for phrase in noun_and_noun_phrase:
		if phrase in noun_phrase_frequency:
			noun_phrase_frequency[phrase] += 1
		else:
			noun_phrase_frequency[phrase] = 1

	favorite_noun_noun_phrase = max(noun_phrase_frequency, key=noun_phrase_frequency.get)

	return noun_and_noun_phrase, noun_phrase_frequency, favorite_noun_noun_phrase

@timeit
def noun_adj_phrase(text):
	"""Gives Noun-adjective phrases from the text and their frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "..."])
	matcher = Matcher(nlp.vocab)
	doc = nlp(text)

	pattern = [{"POS": "NOUN"}, {"POS": "ADJ"}]
	matcher.add("NOUN_ADJ_PATTERN", None, pattern)
	matches = matcher(doc)
	print("Total matches found:", len(matches))

	noun_adj_pairs = [doc[start:end].text for match_id, start, end in matches]
	noun_adj_frequency = {}
	for phrase in noun_adj_pairs:
		if phrase in noun_adj_frequency:
			noun_adj_frequency[phrase] += 1
		else:
			noun_adj_frequency[phrase] = 1

	favorite_noun_adj_phrase = max(noun_adj_frequency, key=noun_adj_frequency.get)

	return noun_adj_pairs, noun_adj_frequency, favorite_noun_adj_phrase

@timeit
def adj_noun_phrase(text):
	"""Gives adjective-noun phrases from the text and their frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "..."])
	matcher = Matcher(nlp.vocab)
	doc = nlp(text)

	pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}]
	matcher.add("ADJ_NOUN_PATTERN", None, pattern)
	matches = matcher(doc)
	print("Total match found:", len(matches))

	adj_noun_pairs = [doc[start:end].text for match_id, start, end in matches]
	adj_noun_frequency = {}
	for phrase in adj_noun_pairs:
		if phrase in adj_noun_frequency:
			adj_noun_frequency[phrase] += 1
		else:
			adj_noun_frequency[phrase] = 1

	favorite_adj_noun_phrase = max(adj_noun_frequency, key=adj_noun_frequency.get)

	return adj_noun_pairs, adj_noun_frequency, favorite_adj_noun_phrase

@timeit
def sentences_with_two_or_more_nouns(text):
	"""Sentences with two or more nouns"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	matcher = Matcher(nlp.vocab)
	matched_sentences = []

	def collect_sents(matcher, doc, i, matches):
		match_id, start, end = matches[i]
		span = doc[start:end]
		sents = span.sent
		matched_sentences.append(sents.text)

	pattern = [{"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "*"}]
	matcher.add("SENTENCES_WITH_2_OR_MORE_NOUNS", collect_sents, pattern)
	doc = nlp(text)
	matches = matcher(doc)

	return matched_sentences

@timeit
def sentences_with_two_or_more_adj(text):
	"""Sentences with two or more adjectives"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	matcher = Matcher(nlp.vocab)
	matched_sentences = []

	def collect_sents(matcher, doc, i, matches):
		match_id, start, end = matches[i]
		span = doc[start:end]
		sents = span.sent
		matched_sentences.append(sents.text)

	pattern = [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "ADJ", "OP": "*"}]
	matcher.add("SENTENCES_WITH_2_OR_MORE_ADJ", collect_sents, pattern)
	doc = nlp(text)
	matches = matcher(doc)

	return matched_sentences

@timeit
def sentences_with_two_or_more_verbs(text):
	"""Sentences with two or more verbs"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	matcher = Matcher(nlp.vocab)
	matched_sentences = []

	def collect_sents(matcher, doc, i, matches):
		match_id, start, end = matches[i]
		span = doc[start:end]
		sents = span.sent
		matched_sentences.append(sents.text)

	pattern = [{"POS": "VERB"}, {"POS": "VERB"}, {"POS": "VERB", "OP": "*"}]
	matcher.add("SENTENCES_WITH_2_OR_MORE_VERB", collect_sents, pattern)
	doc = nlp(text)
	matches = matcher(doc)

	return matched_sentences

@timeit
def sentences_without_noun(text):
	"""Sentences without noun"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	doc = nlp(text)
	total_sentences = list(doc.sents)
	sentences = []
	for sentence in total_sentences:
		for token in sentence:
			if token.pos_ is not "NOUN":
				sentences.append(token)

	return sentences

@timeit
def sentences_without_adjective(text):
	"""Sentences without adjectives"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	doc = nlp(text)
	total_sentences = list(doc.sents)
	sentences = []
	for sentence in total_sentences:
		for token in sentence:
			if token.pos_ is not "ADJ":
				sentences.append(token)

	return sentences

@timeit
def sentences_without_verbs(text):
	"""Sentences without verbs"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	doc = nlp(text)
	total_sentences = list(doc.sents)
	sentences = []
	for sentence in total_sentences:
		for token in sentence:
			if token.pos_ is not "VERB":
				sentences.append(token)

	return sentences

@timeit
def favorite(noun, adj, verb):
	"""Gives the favorite among nouns, adjectives and verbs"""
	if noun>adj and noun>verb:
		result = "Tolstoy favorite among nouns, adjectives and verbs is noun"
	elif adj>noun and adj>verb:
		result = "Tolstoy favorite among nouns, adjectives and verbs is adjective"
	else:
		result = "Tolstoy favorite among nouns, adjectives and verbs is verb"

	return result


@timeit
def person_names(text):
	"""Person names and their frequencies"""
	nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat", "..."])
	doc = nlp(text)
	names = [ent.text for ent in doc.ents if ent.label_=="PERSON"]
	name_frequency = {}
	for name in names:
		if name in name_frequency:
			name_frequency[name] += 1
		else:
			name_frequency[name] = 1
	print("Tolstoy's favorite character is:", max(name_frequency, key=name_frequency.get))

	return names, name_frequency

@timeit
def tense(text):
	"""Total present tense, past tense and future tense sentences"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
	doc = nlp(text)
	sents = doc.sents

	sentences_with_verbs, present_tense_sentences, past_tense_sentences, future_tense_sentences = [], [], [], []

	for sent in sents:
		for token in sent:
			if token.pos_=="VERB":
				sentences_with_verbs.append(sent)
				break

	for sent in sentences_with_verbs:
		for token in sent:
			if token.tag_ in ["VBZ", "VBP", "VBG"]:
				present_tense_sentences.append(sent)
				break

	for sent in sentences_with_verbs:
		for token in sent:
			if token.tag_ in ["VBD", "VBN"]:
				past_tense_sentences.append(sent)
				break

	for sent in sentences_with_verbs:
		for token in sent:
			if token.tag_ in ["VBC", "VBF"]:
				future_tense_sentences.append(sent)
				break

	return present_tense_sentences, past_tense_sentences, future_tense_sentences

def paragraph_splitting(text):
	"""Splits the text into paragraphs"""
	nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "textcat"])
	doc = nlp(text)

	start = 0
	paragraph_list= []
	adj_in_paragraph = {}
	for token in doc:
		if token.is_space and token.text.count("\n") > 1:
			paragraph = doc[start:token.i]
			adjectives = [token for token in paragraph if token.pos_ == "ADJ"]
			for adj in adjectives:
				if adj in adj_in_paragraph:
					adj_in_paragraph[adj] += 1
				else:
					adj_in_paragraph[adj] = 1
			start = token.i
			paragraph_list.append(paragraph)


if __name__ == '__main__':
	stop_words = list(STOP_WORDS)
	text_data = open("Data/2600-0.txt", "r", encoding="utf-8").read().lower()[:90000]

	# create_csv_list("clean_text.csv", "Words without stop_words, spaces, punctuations", clean_text(text_data))

	# noun_count, noun_list, noun_frequency, ten_noun_words= total_nouns(text_data)
	# create_csv("noun_count.csv", "Total number of nouns", noun_count)
	# create_csv_list("noun_list.csv", "list of nouns", noun_list)
	# create_csv_dictionary("noun_frequency.csv", "Noun Frequencies", "Nouns", "Frequencies", noun_frequency)
	# create_csv_dictionary("top_ten_noun_frequency.csv", "Favorite Ten Nouns", "Nouns", "Frequencies", ten_noun_words)
	#
	#
	# adj_count,adj_list, adj_frequency, ten_adj_words = total_adjectives(text_data)
	# create_csv("adj_count.csv", "Total number of adjectives", adj_count)
	# create_csv_list("adj_list.csv", "list of adjectives", adj_list)
	# create_csv_dictionary("adjective_frequency.csv", "Adjective Frequencies", "Adjectives", "Frequencies", adj_frequency)
	# create_csv_dictionary("top_ten_adjective_frequency.csv", "Favorite Ten Adjectives", "Adjectives", "Frequencies", ten_adj_words)
	#
	# verb_count, verb_list, verbs_frequency, ten_verb_words = total_verbs(text_data)
	# create_csv("verb_count.csv", "Total number of verbs", verb_count)
	# create_csv_list("verb_list.csv", "list of verbs", verb_list)
	# create_csv_dictionary("verb_frequency.csv", "Verbs Frequencies", "Verbs", "Frequencies", verbs_frequency)
	# create_csv_dictionary("top_ten_verb_frequency.csv", "Favorite Ten Verbs", "Verbs", "Frequencies", ten_verb_words)
	#
	# noun_noun_phrase_list, noun_noun_phrase_frequency, favorite_phrase = noun_noun_phrase(text_data)
	# create_csv("favorite_noun_noun_phrase.csv", "Tolstoy's favorite noun-noun phrase", favorite_phrase)
	# create_csv_list("noun_noun_phrase_list.csv", "list of noun-noun phrases", noun_noun_phrase_list)
	# create_csv_dictionary("noun_noun_phrase_frequency.csv", "noun-noun phrase frequencies", "noun-noun phrases", "frequencies", noun_noun_phrase_frequency)
	#
	# noun_adj_phrase_list, noun_adj_phrase_frequency, favorite_phrase = noun_adj_phrase(text_data)
	# create_csv("favorite_noun_adj_phrase.csv", "Tolstoy's favorite noun-adj phrase", favorite_phrase)
	# create_csv_list("noun_adj_phrase_list.csv", "list of noun-adj phrases", noun_adj_phrase_list)
	# create_csv_dictionary("noun_adj_phrase_frequency.csv", "noun-adj phrase frequencies", "noun-adj phrases", "frequencies", noun_adj_phrase_frequency)
	#
	# adj_noun_phrase_list, adj_noun_phrase_frequency, favorite_phrase = adj_noun_phrase(text_data)
	# create_csv("favorite_adj_noun_phrase.csv", "Tolstoy's favorite adj-noun phrase", favorite_phrase)
	# create_csv_list("adj_noun_phrase_phrase_list.csv", "list of adj-noun phrases", adj_noun_phrase_list)
	# create_csv_dictionary("adj_noun_phrase_frequency.csv", "adj-noun phrase frequencies", "adj-noun phrases", "frequencies", adj_noun_phrase_frequency)
	#
	# create_csv_list("sentences_with_two_or_more_nouns.csv", "list of sentences with two or more nouns", sentences_with_two_or_more_nouns(text_data))
	# create_csv_list("sentences_with_two_or_more_adj.csv", "list of sentences with two or more adj", sentences_with_two_or_more_adj(text_data))
	# create_csv_list("sentences_with_two_or_more_verbs.csv", "list of sentences with two or more verbs", sentences_with_two_or_more_verbs(text_data))
	#
	# create_csv_list("sentences_without_noun.csv", "list of sentences without nouns", sentences_without_noun(text_data))
	# create_csv_list("sentences_without_adj.csv", "list of sentences without adj", sentences_without_adjective(text_data))
	# create_csv_list("sentences_without_verbs.csv", "list of sentences without verbs", sentences_without_verbs(text_data))
	#
	# name_list, names_frequency = person_names(text_data)
	# create_csv_list("name_list.csv", "list of person names", name_list)
	# create_csv_dictionary("name_frequency.csv", "Person Name Frequencies", "Person Names", "Frequencies", names_frequency)
	#
	# create_csv("favorite.csv", "Favorite among nouns, adjectives and verbs", favorite(noun_count, adj_count, verb_count))
	#
	# present, past, future = tense(text_data)
	# create_csv_list("present_tense_sentences.csv", "Present tense sentences", present)
	# create_csv_list("past_tense_sentences.csv", "Past tense sentences", past)
	# create_csv_list("future_tense_sentences.csv", "Future tense sentences", future)

	total_adjectives(text_data)