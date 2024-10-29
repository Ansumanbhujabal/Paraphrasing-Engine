# from Paraphrasing_test import paraphrase_pg
# /teamspace/studios/this_studio/Paraphrasing-Engine/scripts/Paraphrasing_test.py
from Paraphrasing_test import paraphrase_pg
from Back_Translation import backtranslate
from NLI_Validation import nli_for_corpus
from Document_Processing import extract_text
from Summarizer import summarize_text
import warnings
warnings.filterwarnings("ignore")



text="""
OBJECTIVE
Aspiring Computer Science graduate with a strong foundation in machine learning and Python programming,
seeking a challenging role to leverage my technical skills and academic knowledge. Eager to contribute to innovative
projects and gain hands-on experience in developing and deploying data-driven solutions. Passionate about
continuous learning and collaborating within dynamic teams to solve real-world problems.
EDUCATION
Bachelor of Computer Science, Prala Maharaja Engineering College Expected 2025
Relevant Coursework: ML, Python, Non-relational Database
SKILLS
Programming Language Python, JavaScript, Shell Script
Cloud Technologies AWS
Monitoring Grafana, Prometheus
Database MongoDB
Machine Learning Scikit-Learn, NumPy, Pandas, BeautifulSoup, Model Development
EXPERIENCE
Software Intern May 2024 - Till Date
Invest4Edu Hyderabad, IN
• Designed and implemented a web crawler for scrapping educational data from various Institutes published over
the internet applying various policies and guidelines.
• Part of research group and elite team member for designing EDUGPT one of its kind AI framework which
emulates the student question and proposes correct answer. Able to achieve 19% success rate.
• Developed a pipeline for building Datasets from huge amount of texts from PDFs
PROJECTS
Mindflow. Mind-Flow is a comprehensive system that combines cutting-edge technology, doctor expertise, and
user-centric features to provide a holistic approach to mental health treatment.! (Check it here)
Technology Used Back-end: Python and Data Handler: MongoDB, Monitoring: Grafana & Influx-DB
EXTRA-CURRICULAR ACTIVITIES
• Actively write Medium when I solve a problem that has no beginner friendly approach on internet.
EXECUTIVE MANAGEMENT

• Admin for a Technology Community comprising over 60 individuals from various colleges, responsible for pub-
lishing technical guild reports based on group discussions.

"""


paraphrased_text = paraphrase_pg(text)
back_translated_text=backtranslate(paraphrased_text)
summarized_text1=summarize_text(back_translated_text)
summarized_text2=summarize_text(paraphrased_text)
print(f"-------------------------------------------------------------------")
print(f"original------------->{text}")
print(f"-------------------------------------------------------------------")
print("\n")
print(f"paraphrased_text------------->{paraphrased_text}")
print(f"-------------------------------------------------------------------")
print("\n")
print(f"back_translated_text------------->{back_translated_text}")
print(f"-------------------------------------------------------------------")
print("\n")
print(summarized_text1)
print(f"-------------------------------------------------------------------")
print("\n")
print(summarized_text2)
print(f"-------------------------------------------------------------------")
print("\n")
nli_score1=nli_for_corpus(text,paraphrased_text)
print(nli_score1)
print(f"-------------------------------------------------------------------")
print("\n")
nli_score2=nli_for_corpus(text,back_translated_text)
print(nli_score2)
print(f"-------------------------------------------------------------------")
print("\n")


