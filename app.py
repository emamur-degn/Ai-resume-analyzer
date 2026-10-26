import re
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer", page_icon="📄", layout="wide"
)


# Function to extract text from PDF or Image
def extract_text_from_file(uploaded_file):
  file_type = uploaded_file.name.split(".")[-1].lower()
  text = ""

  if file_type == "pdf":
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
      text += page.get_text()
  elif file_type in ["png", "jpg", "jpeg", "jfif"]:
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)

  return text


# Function to generate a quick summary/overview of the resume
def generate_resume_summary(text):
  lines = [line.strip() for line in text.split("\n") if line.strip()]
  if not lines:
    return "রেজুমি থেকে কোনো তথ্য পড়া যায়নি।"

  # Word count & key features
  word_count = len(text.split())
  preview = " ".join(lines[:5]) + "..." if len(lines) > 5 else " ".join(lines)

  summary = f"""
    **📝 রেজুমি সামারি (Resume Overview):**
    - **মোট শব্দ সংখ্যা:** {word_count} টি শব্দ
    - **প্রাথমিক বিষয়বস্তু/হেডার:** {preview}
    """
  return summary


# Function to calculate match score using NLP (TF-IDF & Cosine Similarity)
def calculate_match_score(resume_text, jd_text):
  if not resume_text.strip() or not jd_text.strip():
    return 0.0
  vectorizer = TfidfVectorizer().fit_transform([resume_text, jd_text])
  vectors = vectorizer.toarray()
  cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
  return round(cosine_sim * 100, 2)


# Function to find missing keywords/skills
def find_missing_skills(resume_text, jd_text):
  stop_words = set([
      "and",
      "the",
      "to",
      "a",
      "of",
      "in",
      "for",
      "is",
      "on",
      "that",
      "by",
      "this",
      "with",
      "i",
      "you",
      "it",
      "not",
      "or",
      "be",
      "are",
      "from",
      "at",
      "as",
      "your",
      "all",
      "have",
      "new",
      "more",
      "an",
      "was",
      "we",
      "will",
      "my",
      "one",
      "has",
  ])

  jd_words = set(
      [
          word
          for word in re.findall(r"\b[a-zA-Z]{3,}\b", jd_text.lower())
          if word not in stop_words
      ]
  )
  resume_words = set(
      [
          word
          for word in re.findall(r"\b[a-zA-Z]{3,}\b", resume_text.lower())
          if word not in stop_words
      ]
  )

  missing = list(jd_words - resume_words)
  return missing[:10]


# UI Design with Streamlit
st.title("📄 AI Resume Analyzer (ATS-Style)")
st.markdown(
    "আপনার রেজুমি (Resume) আপলোড করুন এবং জব ডেসক্রিপশন (Job Description)"
    " দিন। সিস্টেমটি ATS স্কোর এবং মিসিং স্কিলগুলো খুঁজে বের করবে।"
)

col1, col2 = st.columns(2)

resume_text = ""

with col1:
  st.subheader("1. Resume আপলোড করুন (PDF / Image)")
  uploaded_file = st.file_uploader(
      "আপনার রেজুমি সিলেক্ট করুন", type=["pdf", "png", "jpg", "jpeg", "jfif"]
  )

  # Auto-extract and display resume description upon upload
  if uploaded_file is not None:
    try:
      resume_text = extract_text_from_file(uploaded_file)
      st.info("📌 রেজুমি সফলভাবে আপলোড হয়েছে!")

      # Display Resume Summary/Description right under upload
      with st.expander("📄 আপলোড করা রেজুমির বিবরণ (Resume Description)", expanded=True):
        st.markdown(generate_resume_summary(resume_text))
        st.text_area(
            "এক্সট্র্যাক্ট করা মূল টেক্সট (Extracted Text):",
            resume_text,
            height=150,
        )
    except Exception as e:
      st.error(
          f"ফাইল পড়তে সমস্যা হয়েছে। দয়া করে সঠিক PDF বা ইমেজ আপলোড করুন। Error: {e}"
      )

with col2:
  st.subheader("2. Job Description দিন")
  job_description = st.text_area(
      "এখানে চাকরির বিবরণী (Job Description) পেস্ট করুন...", height=200
  )

st.markdown("---")

if st.button("🚀 Analyze Resume", type="primary"):
  if uploaded_file is not None and job_description.strip() != "":
    with st.spinner("রেজুমি অ্যানালাইসিস করা হচ্ছে..."):
      score = calculate_match_score(resume_text, job_description)
      missing_skills = find_missing_skills(resume_text, job_description)

    st.success("অ্যানালাইসিস সম্পন্ন হয়েছে!")

    # Display Results
    st.header("📊 অ্যানালাইসিস রেজাল্ট")
    col_score, col_details = st.columns([1, 2])

    with col_score:
      st.metric(
          label="🎯 ATS Match Score",
          value=f"{score}%",
          delta="Good Match" if score > 60 else "Needs Improvement",
      )

    with col_details:
      st.subheader("💡 উন্নতির পরামর্শ ও মিসিং স্কিলস:")
      if missing_skills:
        st.write(
            "নিচের কিওয়ার্ড বা স্কিলগুলো আপনার রেজুমিতে পাওয়া যায়নি, এগুলো"
            " যুক্ত করলে স্কোর বাড়তে পারে:"
        )
        for skill in missing_skills:
          st.markdown(f"- `{skill}`")
      else:
        st.write(
            "দারুণ! আপনার রেজুমিতে জব ডেসক্রিপশনের বেশিরভাগ কিওয়ার্ড মিলে গেছে।"
        )
  else:
    st.error(
        "দয়া করে একটি রেজুমি (PDF/Image) আপলোড করুন এবং জব ডেসক্রিপশন বক্সে"
        " টেক্সট লিখুন।"
    )