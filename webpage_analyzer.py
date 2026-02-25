import requests
from bs4 import BeautifulSoup

def analyze_webpage(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")

    text = soup.get_text().lower()

    features = {}
    features["faq"] = int("faq" in text)

    headings = soup.find_all(["h1","h2","h3"])
    features["questions"] = sum(1 for h in headings if "?" in h.text)

    paragraphs = soup.find_all("p")
    features["short_para"] = sum(1 for p in paragraphs if len(p.text.split()) < 60)

    return features

def compute_score(features):
    score = 0
    if features["faq"]: score += 30
    score += min(features["questions"]*5,25)
    score += min(features["short_para"]*2,45)
    return min(score,100)

def recommendations(features):
    rec = []
    if not features["faq"]:
        rec.append("Add FAQ schema markup.")
    if features["questions"] < 3:
        rec.append("Use more question-based headings.")
    if features["short_para"] < 5:
        rec.append("Provide short 40-60 word answers.")
    return rec