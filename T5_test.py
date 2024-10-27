from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain import LLMChain, PromptTemplate
import warnings
warnings.filterwarnings("ignore")

# Load pre-trained T5 model and tokenizer from Hugging Face
model_name = "t5-large"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def paraphrase_text(text, max_length=512):
    # Add the task prefix for paraphrasing
    input_text = f"paraphrase: {text}"
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    # Generate the paraphrased output
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)

    # Decode the generated output
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return paraphrased_text


# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Paraphrase the following text:\n\n{text}"
)

# Define an LLMChain with the paraphrasing function
def paraphrase_chain(input_text):
    paraphrased_text = paraphrase_text(input_text)
    return paraphrased_text


input_text ="""Electronics and Communications Engineering (ECE) is a dynamic subject that involves the design, development, and maintenance of electronic systems and communication networks used in a variety of sectors, including telecommunications, information technology, healthcare, and military. ECE engineers serve critical roles by using their skills in electrical circuits, communication protocols, and digital signal processing.
ECE engineers are in great demand because of their ability to invent and optimize communication technology. Their expertise includes building modern electronic gadgets, guaranteeing a strong network architecture, and integrating cutting-edge technology for flawless data transmission and reception. This breadth of abilities qualifies them for a variety of professions within various industries, including telecommunications professionals, network engineers, and research scientists.
The salary for electronics and communication engineering experts reflects their specialised expertise and vital role in technological progress. ECE engineering positions provide excellent salary packages that are frequently commensurate with the complexity of their tasks and the importance of their contributions to industry innovation.
The scope of electronic communication engineering is expanding as global connection and digital transformation increase need for competent personnel. Emerging disciplines such as IoT (Internet of Things), 5G network development, and smart technologies provide several prospects for electronics and communication engineers. ECE engineers are leading the way in determining the future of communication systems via their experience in building efficient, secure, and high-performance electronic solutions.
In essence, ECE engineers make substantial contributions to technology developments while also affecting how societies interact and communicate in a rapidly linked world. Their broad talents and understanding of complex electrical systems make them vital for generating innovation and tackling changing issues in the digital age. """

paraphrased_output = paraphrase_chain(input_text)
print(f"Original Text: {input_text}")
print(f"Paraphrased Text: {paraphrased_output}")
