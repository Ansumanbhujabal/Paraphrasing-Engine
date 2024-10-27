from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings("ignore")
from Paraphrasing_test import paraphrase_pg


# Load a translation model (English to French, and French to English)
src_lang = "en"  # source language
tgt_lang = "fr"  # target language

en_to_fr_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
fr_to_en_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"

# Load English to French tokenizer and model
en_to_fr_tokenizer = MarianTokenizer.from_pretrained(en_to_fr_model_name)
en_to_fr_model = MarianMTModel.from_pretrained(en_to_fr_model_name)

# Load French to English tokenizer and model
fr_to_en_tokenizer = MarianTokenizer.from_pretrained(fr_to_en_model_name)
fr_to_en_model = MarianMTModel.from_pretrained(fr_to_en_model_name)

# Translate from English to French
def translate_to_french(text):
    tokens = en_to_fr_tokenizer(text, return_tensors="pt", truncation=True)
    translated = en_to_fr_model.generate(**tokens)
    return en_to_fr_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Translate from French back to English
def translate_to_english(text):
    tokens = fr_to_en_tokenizer(text, return_tensors="pt", truncation=True)
    translated = fr_to_en_model.generate(**tokens)
    return fr_to_en_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def backtranslate(corpus):
  corpus_in_french=translate_to_french(corpus)
  corpus_in_english=translate_to_english(corpus_in_french)
  return (corpus_in_english)


# Original_Text = """
# It is the duty of the government of India, of every government in
# every state of India, of every sportsman who visits her forests—
# indeed, it is the duty of every Indian—to make a supreme effort to
# save the country’s wild creatures from extinction.
# Game and bird sanctuaries have been formed, it is true. But this
# is not nearly enough. Rules are printed that cannot be enforced.
# For one thing, the forest guards and watches are not paid enough.
# For another, corruption is rife.
# In the state of Mysore as a whole, and also in the district of
# Salem belonging to Madras state, tigers and panthers are now
# almost extinct, wiped out by the villagers who use a poison
# supplied to them almost free of charge by the local governments as
# an insecticide to protect their crops. This poison they smear on the
# flesh of the kills made by tigers and panthers. These animals
# invariably return, eat the doctored meat and die within a few
# yards. So do the jackals, hyaenas, vultures and crows that follow
# them.

# """




# paraphrased_text = paraphrase_pg(Original_Text)
# text_in_french = translate_to_french(paraphrased_text)
# back_translated_text = translate_to_english(text_in_french)
# print("--------------------------------------------->>>>>>>>>>>>>>>>>")
# print(f"Original_text:{paraphrased_text}")
# print("--------------------------------------------->>>>>>>>>>>>>>>>>")
# print(f"text_in_french:{text_in_french}")
# print("--------------------------------------------->>>>>>>>>>>>>>>>>")
# print(f"Back-translated Text1: {back_translated_text}")
# print("--------------------------------------------->>>>>>>>>>>>>>>>>")
# print(f"Back-translated Text2: {backtranslate(paraphrased_text)}")




