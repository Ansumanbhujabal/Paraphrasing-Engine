from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the pre-trained Pegasus model and tokenizer
tokenizer_pg = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model_pg = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")

# Function to clean the input text
def clean_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except punctuation marks
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

# Function to break down structured data like skills or education into smaller chunks
def chunk_large_sentences(sentence, max_len=100):
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 > max_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(word)
        current_len += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function to paraphrase text using Pegasus
def paraphrase_pg(corpus, max_len=100):
    paraphrased_sentences = []
    sentences = corpus.split(".")  # Split text by period to get individual sentences
    for sentence in sentences:
        sentence = sentence.strip()  # Clean up leading/trailing spaces
        if not sentence:
            continue
        
        # If the sentence is too long, break it into smaller chunks
        sentence_chunks = chunk_large_sentences(sentence, max_len)
        
        for chunk in sentence_chunks:
            try:
                # Encode input sentence and generate paraphrase
                input_ids = tokenizer_pg.encode(chunk, return_tensors='pt', truncation=True, max_length=max_len)
                paraphrase_ids = model_pg.generate(input_ids, num_beams=5, max_length=max_len, early_stopping=True)
                paraphrase = tokenizer_pg.decode(paraphrase_ids[0], skip_special_tokens=True)
                paraphrased_sentences.append(paraphrase)  # Add paraphrase to list
            except Exception as e:
                print(f"Error paraphrasing sentence: '{chunk}'\nError: {str(e)}")
                continue
    return "\n".join(paraphrased_sentences)  # Join paraphrased sentences into a single string

# Sample input text
# text = """
# Ophelia rushes to her father, telling him that Hamlet arrived at her door the prior night half-undressed and behaving erratically. Polonius blames love for Hamlet's madness and resolves to inform Claudius and Gertrude. As he enters to do so, the king and queen are welcoming Rosencrantz and Guildenstern, two student acquaintances of Hamlet, to Elsinore. The royal couple has requested that the two students investigate the cause of Hamlet's mood and behaviour. Additional news requires that Polonius wait to be heard: messengers from Norway inform Claudius that the king of Norway has rebuked Prince Fortinbras for attempting to re-fight his father's battles. The forces that Fortinbras had conscripted to march against Denmark will instead be sent against Poland, though they will pass through Danish territory to get there.

# Polonius tells Claudius and Gertrude his theory regarding Hamlet's behaviour, and then speaks to Hamlet in a hall of the castle to try to learn more. Hamlet feigns madness and subtly insults Polonius all the while. When Rosencrantz and Guildenstern arrive, Hamlet greets his "friends" warmly but quickly discerns that they are there to spy on him for Claudius. Hamlet admits that he is upset at his situation but refuses to give the true reason, instead remarking "What a piece of work is a man". Rosencrantz and Guildenstern tell Hamlet that they have brought along a troupe of actors that they met while travelling to Elsinore. Hamlet, after welcoming the actors and dismissing his friends-turned-spies, asks them to deliver a soliloquy about the death of King Priam and Queen Hecuba at the climax of the Trojan War. Hamlet then asks the actors to stage The Murder of Gonzago, a play featuring a death in the style of his father's murder. Hamlet intends to study Claudius's reaction to the play, and thereby determine the truth of the ghost's story of Claudius's guilt.

# Act III
# Polonius forces Ophelia to return Hamlet's love letters to the prince while he and Claudius secretly watch in order to evaluate Hamlet's reaction. Hamlet is walking alone in the hall as the King and Polonius await Ophelia's entrance. Hamlet muses on thoughts of life versus death. When Ophelia enters and tries to return Hamlet's things, Hamlet accuses her of immodesty and cries "get thee to a nunnery", though it is unclear whether this, too, is a show of madness or genuine distress. His reaction convinces Claudius that Hamlet is not mad for love. Shortly thereafter, the court assembles to watch the play Hamlet has commissioned. After seeing the Player King murdered by his rival pouring poison in his ear, Claudius abruptly rises and runs from the room; for Hamlet, this is proof of his uncle's guilt.


# Hamlet mistakenly stabs Polonius (Artist: Coke Smyth, 19th century).
# Gertrude summons Hamlet to her chamber to demand an explanation. Meanwhile, Claudius talks to himself about the impossibility of repenting, since he still has possession of his ill-gotten goods: his brother's crown and wife. He sinks to his knees. Hamlet, on his way to visit his mother, sneaks up behind him but does not kill him, reasoning that killing Claudius while he is praying will send him straight to heaven while his father's ghost is stuck in purgatory. In the queen's bedchamber, Hamlet and Gertrude fight bitterly. Polonius, spying on the conversation from behind a tapestry, calls for help as Gertrude, believing Hamlet wants to kill her, calls out for help herself.

# Hamlet, believing it is Claudius, stabs wildly, killing Polonius, but he pulls aside the curtain and sees his mistake. In a rage, Hamlet brutally insults his mother for her apparent ignorance of Claudius's villainy, but the ghost enters and reprimands Hamlet for his inaction and harsh words. Unable to see or hear the ghost herself, Gertrude takes Hamlet's conversation with it as further evidence of madness. After begging the queen to stop sleeping with Claudius, Hamlet leaves, dragging Polonius's corpse away.

# Act IV
# Hamlet jokes with Claudius about where he has hidden Polonius's body, and the king, fearing for his life, sends Rosencrantz and Guildenstern to accompany Hamlet to England with a sealed letter to the English king requesting that Hamlet be executed immediately.

# Unhinged by grief at Polonius's death, Ophelia wanders Elsinore. Laertes arrives back from France, enraged by his father's death and his sister's madness. Claudius convinces Laertes that Hamlet is solely responsible, but a letter soon arrives indicating that Hamlet has returned to Denmark, foiling Claudius's plan. Claudius switches tactics, proposing a fencing match between Laertes and Hamlet to settle their differences. Laertes will be given a poison-tipped foil, and, if that fails, Claudius will offer Hamlet poisoned wine as a congratulation. Gertrude interrupts to report that Ophelia has drowned, though it is unclear whether it was suicide or an accident caused by her madness.
# """

# # Clean the input text and paraphrase it
# cleaned_text = clean_text(text)
# paraphrased_output = paraphrase_pg(cleaned_text)

# # Print the paraphrased output
# print(paraphrased_output)
